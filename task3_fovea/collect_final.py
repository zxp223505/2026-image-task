# -*- coding: utf-8 -*-
"""collect_final.py -- leak-free predictions with NO epoch selection.

Why this exists
---------------
Picking "the epoch with the lowest val RMSE" and then scoring on those same 16
val images is optimistic: measured on hm1b, selected-best gave 27.62 while every
fixed epoch gave 33-40 (mean 36.7), and averaging all fixed-epoch snapshots gave
32.44 (best honest).  So the submission is built from the *average of fixed-epoch
snapshots*, which needs no validation look at all.

Outputs mem3_<tag>.npz with, for each of the gated / flat extraction variants:
    oof_{g,f}  (N,2)  original-px predictions for every train image (fold model only)
    te_{g,f}   (M,2)  average over all 5 fold models x all snapshots
    oof_conf, te_conf normalized heatmap entropy
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
import hm2
from hm1 import (FoveaDS, FoveaNet, soft_argmax, split_folds, N_FOLDS, CKPT,
                 CELL2CANON, GATE)
from hm2 import PatchDS, CELL2CANON2, PATCH
from common import train_items, test_items, WORK


def ckpt_path(tag, k, snap):
    return os.path.join(CKPT, f'{tag}_fold{k}' + ('' if snap is None else f'_s{snap}') + '.pt')


def load_model(tag, k, snap):
    ck = torch.load(ckpt_path(tag, k, snap), map_location='cuda', weights_only=False)
    m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def n_snaps(tag, k):
    n = 0
    while os.path.exists(ckpt_path(tag, k, n)):
        n += 1
    return n


def entropy_of(h):
    n = h.shape[0]
    f = h.flatten(2)
    mx = f.max(2, keepdim=True).values
    f = torch.where(f >= mx - GATE, f, torch.full_like(f, -1e30))
    p = torch.softmax(f, 2)
    ent = -(p * torch.log(p + 1e-30)).sum(2)[:, 0]
    w = h.shape[-1]
    return (ent / np.log(w * w)).cpu().numpy()


@torch.no_grad()
def run_views(model, ds):
    """(gated, flat, conf) coordinates in the dataset's cell units."""
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    G, F, C = [], [], []
    for x, _, _ in dl:
        g = np.zeros((x.shape[0], 2)); fl = np.zeros((x.shape[0], 2))
        c = np.zeros(x.shape[0])
        for mirror in (False, True):
            xx = torch.flip(x, dims=[3]) if mirror else x
            h, _ = model(xx.cuda())
            if mirror:
                h = torch.flip(h, dims=[3])
            g += soft_argmax(h).cpu().numpy() / 2
            fl += soft_argmax(h, gate=0.0).cpu().numpy() / 2
            c += entropy_of(h) / 2
        G.append(g); F.append(fl); C.append(c)
    return np.concatenate(G), np.concatenate(F), np.concatenate(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--base', default=None, help='stage-1 member whose centres stage 2 uses')
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--snap_lo', type=int, default=0, help='first snapshot index used')
    ap.add_argument('--dog', action='store_true')
    a = ap.parse_args()
    hm1.DOG = a.dog

    items, folds = split_folds()
    te = test_items()
    n = len(items)
    oG = np.zeros((n, 2)); oF = np.zeros((n, 2)); oC = np.zeros(n)
    tG = np.zeros((len(te), 2)); tF = np.zeros((len(te), 2)); tC = np.zeros(len(te))

    if a.stage == 1:
        def predict(m, geos, split):
            g, f, c = run_views(m, FoveaDS(geos, split, train=False))
            return g * CELL2CANON, f * CELL2CANON, c
        centers = None
    else:
        base = np.load(os.path.join(WORK, f'mem3_{a.base}.npz'))
        # PatchDS / infer2 work in the CANONICAL frame
        o_cent, t_cent = base['oof_canon'], base['te_canon']

        def predict(m, geos, split, cen=None):
            g, f, c = run_views(m, PatchDS(geos, cen, split, train=False))
            cc = np.clip(np.asarray(cen, float), PATCH / 2, hm1.CANON - PATCH / 2)
            o = cc - PATCH / 2
            return o + g * CELL2CANON2, o + f * CELL2CANON2, c

    used = 0
    for k in range(N_FOLDS):
        snaps = list(range(a.snap_lo, n_snaps(a.tag, k)))
        assert snaps, f'no snapshots for {a.tag} fold {k}'
        ns = len(snaps)
        va = [items[i] for i in folds[k]]
        accG = np.zeros((len(va), 2)); accF = np.zeros((len(va), 2)); accC = np.zeros(len(va))
        for s in snaps:
            m = load_model(a.tag, k, s)
            if a.stage == 1:
                g, f, c = predict(m, va, 'train')
            else:
                g, f, c = predict(m, va, 'train', o_cent[folds[k]])
            accG += g / ns; accF += f / ns; accC += c / ns
            del m; torch.cuda.empty_cache()
        oG[folds[k]] = accG; oF[folds[k]] = accF; oC[folds[k]] = accC
        used = ns
        print(f'  fold{k}: averaged {ns} snapshots {snaps}', flush=True)

    for k in range(N_FOLDS):
        snaps = list(range(a.snap_lo, n_snaps(a.tag, k)))
        ns = len(snaps)
        accG = np.zeros((len(te), 2)); accF = np.zeros((len(te), 2)); accC = np.zeros(len(te))
        for s in snaps:
            m = load_model(a.tag, k, s)
            if a.stage == 1:
                g, f, c = predict(m, te, 'test')
            else:
                g, f, c = predict(m, te, 'test', t_cent)
            accG += g / ns / N_FOLDS; accF += f / ns / N_FOLDS; accC += c / ns / N_FOLDS
            del m; torch.cuda.empty_cache()
        tG += accG; tF += accF; tC += accC

    r_g = hm1.rmse(oG, items)
    r_f = hm1.rmse(oF, items)
    r_m = hm1.rmse(np.column_stack([oG[:, 0], oF[:, 1]]), items)
    print(f'\n{a.tag}  [{used} snapshots/fold, NO epoch selection]')
    print(f'  gated  OOF {r_g[0]:6.2f} (x {r_g[1]:.2f} y {r_g[2]:.2f})')
    print(f'  flat   OOF {r_f[0]:6.2f} (x {r_f[1]:.2f} y {r_f[2]:.2f})')
    print(f'  gx+fy  OOF {r_m[0]:6.2f}')

    out = os.path.join(WORK, f'mem3_{a.tag}.npz')
    np.savez(out,
             oof_g=np.array([items[i].to_orig(*oG[i]) for i in range(n)]),
             oof_f=np.array([items[i].to_orig(*oF[i]) for i in range(n)]),
             te_g=np.array([te[i].to_orig(*tG[i]) for i in range(len(te))]),
             te_f=np.array([te[i].to_orig(*tF[i]) for i in range(len(te))]),
             oof_conf=oC, te_conf=tC,
             oof_canon=oG, te_canon=tG)
    print('wrote', out)


if __name__ == '__main__':
    main()

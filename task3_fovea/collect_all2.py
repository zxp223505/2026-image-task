# -*- coding: utf-8 -*-
"""collect_all2.py -- like collect_all.py but stores BOTH extraction variants.

Rationale: the optic disc sits ~0.7R away from the fovea almost purely
horizontally, so a weak secondary disc peak contaminates the *x* coordinate but
not *y*.  Storing the gated and the ungated coordinate for every member lets the
final assembler pick per axis (chosen by cross-fitting, see final3.py).
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
import hm2
from hm1 import FoveaDS, FoveaNet, soft_argmax, split_folds, N_FOLDS, CKPT, CELL2CANON, GATE
from hm2 import PatchDS, CELL2CANON2, PATCH
from common import train_items, test_items, WORK


def load_model(tag, k):
    ck = torch.load(os.path.join(CKPT, f'{tag}_fold{k}.pt'), map_location='cuda',
                    weights_only=False)
    m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


@torch.no_grad()
def views(model, ds):
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    g, f = [], []
    for x, _, _ in dl:
        cg = np.zeros((x.shape[0], 2)); cf = np.zeros((x.shape[0], 2))
        for fl in (False, True):
            xx = torch.flip(x, dims=[3]) if fl else x
            h, _ = model(xx.cuda())
            if fl:
                h = torch.flip(h, dims=[3])
            a = soft_argmax(h).cpu().numpy()          # gated (hm1.GATE)
            b = soft_argmax(h, gate=0.0).cpu().numpy()  # ungated
            cg += a / 2; cf += b / 2
        g.append(cg); f.append(cf)
    return np.concatenate(g), np.concatenate(f)


def ent(h_):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm2')
    ap.add_argument('--base', default='hm1')
    ap.add_argument('--stage', type=int, default=2)
    a = ap.parse_args()

    items, folds = split_folds()
    te = test_items()
    og = np.zeros((len(items), 2)); of = np.zeros((len(items), 2))
    tg = np.zeros((len(te), 2)); tf = np.zeros((len(te), 2))

    if a.stage == 1:
        def run(m, geos, split):
            cg, cf = views(m, FoveaDS(geos, split, train=False))
            return cg * CELL2CANON, cf * CELL2CANON
        for k in range(N_FOLDS):
            m = load_model(a.tag, k)
            va = [items[i] for i in folds[k]]
            g, f = run(m, va, 'train')
            og[folds[k]] = g; of[folds[k]] = f
            del m; torch.cuda.empty_cache()
        for k in range(N_FOLDS):
            m = load_model(a.tag, k)
            g, f = run(m, te, 'test')
            tg += g / N_FOLDS; tf += f / N_FOLDS
            del m; torch.cuda.empty_cache()
    else:
        base = np.load(os.path.join(WORK, f'oof_{a.base}.npz'))
        oc, tc = base['oof_hm'], base['te_hm']

        def run(m, geos, centers, split):
            cen = np.clip(np.asarray(centers, float), PATCH / 2, hm1.CANON - PATCH / 2)
            cg, cf = views(m, PatchDS(geos, centers, split, train=False))
            o = cen - PATCH / 2
            return o + cg * CELL2CANON2, o + cf * CELL2CANON2
        for k in range(N_FOLDS):
            m = load_model(a.tag, k)
            va = [items[i] for i in folds[k]]
            g, f = run(m, va, oc[folds[k]], 'train')
            og[folds[k]] = g; of[folds[k]] = f
            del m; torch.cuda.empty_cache()
        for k in range(N_FOLDS):
            m = load_model(a.tag, k)
            g, f = run(m, te, tc, 'test')
            tg += g / N_FOLDS; tf += f / N_FOLDS
            del m; torch.cuda.empty_cache()

    print(f'{a.tag} gated OOF {hm1.rmse(og, items)[0]:.2f}   flat OOF {hm1.rmse(of, items)[0]:.2f}')
    print(f'  mixed (gated x, flat y): '
          f'{hm1.rmse(np.column_stack([og[:, 0], of[:, 1]]), items)[0]:.2f}')
    print(f'  mixed (flat x, gated y): '
          f'{hm1.rmse(np.column_stack([of[:, 0], og[:, 1]]), items)[0]:.2f}')
    out = os.path.join(WORK, f'mem2_{a.tag}.npz')
    np.savez(out,
             oof_g=np.array([items[i].to_orig(*og[i]) for i in range(len(items))]),
             oof_f=np.array([items[i].to_orig(*of[i]) for i in range(len(items))]),
             te_g=np.array([te[i].to_orig(*tg[i]) for i in range(len(te))]),
             te_f=np.array([te[i].to_orig(*tf[i]) for i in range(len(te))]))
    print('wrote', out)


if __name__ == '__main__':
    main()

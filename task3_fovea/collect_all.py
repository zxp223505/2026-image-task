# -*- coding: utf-8 -*-
"""collect_all.py -- one pass that produces, for a whole pipeline member:

  oof_pred (N,2)  leak-free fovea prediction in ORIGINAL px for every train image
  oof_conf (N,)   heatmap entropy (0=very confident .. 1=flat)
  te_pred  (M,2)  5-fold-ensemble prediction for every test image
  te_conf  (M,)

stage1: full canonical frame; stage2: high-res patch around the stage-1 estimate.
Stage-2 centres always come from the *stored* stage-1 member predictions, so the
crop centre used at test time is exactly the one a real submission would use.
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
import hm2
from hm1 import FoveaDS, FoveaNet, soft_argmax, split_folds, N_FOLDS, CKPT, CELL2CANON, GATE
from hm2 import PatchDS, infer2, CELL2CANON2, OUTSZ2, PATCH
from common import train_items, test_items, WORK

H1 = hm1.OUTSZ
H2 = OUTSZ2


def load_model(tag, k, snap=None):
    name = f'{tag}_fold{k}.pt' if snap is None else f'{tag}_fold{k}_s{snap}.pt'
    ck = torch.load(os.path.join(CKPT, name), map_location='cuda', weights_only=False)
    m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def load_models(tag, k, nsnap):
    """best checkpoint + any saved snapshots, for prediction averaging."""
    out = [load_model(tag, k, None)]
    for s in range(nsnap):
        try:
            out.append(load_model(tag, k, s))
        except FileNotFoundError:
            break
    return out


def entropy_from(h):
    """h (N,1,H,W) gated softmax -> normalised entropy (N,)"""
    n = h.shape[0]
    f = h.flatten(2)
    mx = f.max(2, keepdim=True).values
    f = torch.where(f >= mx - GATE, f, torch.full_like(f, -1e30))
    p = torch.softmax(f, 2)
    ent = -(p * torch.log(p + 1e-30)).sum(2)[:, 0]
    return (ent / np.log(H1 * H1)).cpu().numpy()


@torch.no_grad()
def stage1_out(model, geos, split):
    ds = FoveaDS(geos, split, train=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    coords, conf = [], []
    for x, _, _ in dl:
        hs = []
        for fl in (False, True):
            xx = torch.flip(x, dims=[3]) if fl else x
            h, _ = model(xx.cuda())
            if fl:
                h = torch.flip(h, dims=[3])
            hs.append(h)
        coords.append(0.5 * (soft_argmax(hs[0]) + soft_argmax(hs[1])).cpu().numpy())
        conf.append(0.5 * (entropy_from(hs[0]) + entropy_from(hs[1])))
    c = np.concatenate(coords) * CELL2CANON
    return c, np.concatenate(conf)


@torch.no_grad()
def stage2_out(model, geos, centers, split):
    ds = PatchDS(geos, centers, split, train=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    offs, conf = [], []
    for x, _, _ in dl:
        hs = []
        for fl in (False, True):
            xx = torch.flip(x, dims=[3]) if fl else x
            h, _ = model(xx.cuda())
            if fl:
                h = torch.flip(h, dims=[3])
            hs.append(h)
        offs.append(0.5 * (soft_argmax(hs[0]) + soft_argmax(hs[1])).cpu().numpy())
        conf.append(0.5 * (entropy_from(hs[0]) + entropy_from(hs[1])))
    off = np.concatenate(offs) * CELL2CANON2
    cen = np.clip(np.asarray(centers, float), PATCH / 2, hm1.CANON - PATCH / 2)
    coords = cen - PATCH / 2 + off
    return coords, np.concatenate(conf)


def entropy_scaled(e, h):
    return e / np.log(h * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm2')
    ap.add_argument('--base', default='hm1')
    ap.add_argument('--stage', type=int, default=2)
    ap.add_argument('--nsnap', type=int, default=0)
    a = ap.parse_args()

    items, folds = split_folds()
    te = test_items()
    oof = np.zeros((len(items), 2))
    oconf = np.zeros(len(items))
    tep = np.zeros((len(te), 2))
    tconf = np.zeros(len(te))
    runner = stage1_out if a.stage == 1 else stage2_out
    nmod = 0

    if a.stage == 1:
        pairs = [(k, [items[i] for i in folds[k]]) for k in range(N_FOLDS)]
        for k, va in pairs:
            ms = load_models(a.tag, k, a.nsnap)
            acc = np.zeros((len(va), 2))
            accc = np.zeros(len(va))
            for m in ms:
                c, cf = runner(m, va, 'train')
                acc += c / len(ms); accc += cf / len(ms)
                del m; torch.cuda.empty_cache()
            oof[folds[k]] = acc
            oconf[folds[k]] = accc
            nmod = len(ms)
        for k in range(N_FOLDS):
            ms = load_models(a.tag, k, a.nsnap)
            acc = np.zeros((len(te), 2))
            accc = np.zeros(len(te))
            for m in ms:
                c, cf = runner(m, te, 'test')
                acc += c / len(ms); accc += cf / len(ms)
                del m; torch.cuda.empty_cache()
            tep += acc / N_FOLDS
            tconf += accc / N_FOLDS
    else:
        base = np.load(os.path.join(WORK, f'oof_{a.base}.npz'))
        oc, tc = base['oof_hm'], base['te_hm']
        for k in range(N_FOLDS):
            ms = load_models(a.tag, k, a.nsnap)
            va = [items[i] for i in folds[k]]
            acc = np.zeros((len(va), 2)); accc = np.zeros(len(va))
            for m in ms:
                c, cf = runner(m, va, oc[folds[k]], 'train')
                acc += c / len(ms); accc += cf / len(ms)
                del m; torch.cuda.empty_cache()
            oof[folds[k]] = acc
            oconf[folds[k]] = accc
            nmod = len(ms)
        for k in range(N_FOLDS):
            ms = load_models(a.tag, k, a.nsnap)
            acc = np.zeros((len(te), 2)); accc = np.zeros(len(te))
            for m in ms:
                c, cf = runner(m, te, tc, 'test')
                acc += c / len(ms); accc += cf / len(ms)
                del m; torch.cuda.empty_cache()
            tep += acc / N_FOLDS
            tconf += accc / N_FOLDS
    print(f'averaged {nmod} checkpoint(s) per fold')

    # to original px for the submission frame
    oof_o = np.array([items[i].to_orig(*oof[i]) for i in range(len(items))])
    tep_o = np.array([te[i].to_orig(*tep[i]) for i in range(len(te))])
    out = os.path.join(WORK, f'mem_{a.tag}.npz')
    np.savez(out, oof_canon=oof, oof_pred=oof_o, oof_conf=oconf,
             te_canon=tep, te_pred=tep_o, te_conf=tconf)
    s = hm1.rmse(oof, items)
    print(f'{a.tag} OOF {s[0]:.2f} (x {s[1]:.2f} y {s[2]:.2f})  '
          f'conf {oconf.min():.3f}..{oconf.max():.3f}')
    print('wrote', out)


if __name__ == '__main__':
    main()

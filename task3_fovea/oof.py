# -*- coding: utf-8 -*-
"""oof.py -- collect leak-free predictions for every train image + ensemble
predictions for the test images, so later stages / calibration can be trained
and scored without any leakage.

For train image i in fold f we use ONLY models of fold f (trained without i).
For test images we average all fold models.  Optional snapshot averaging.
"""
import os, argparse
import numpy as np
import torch

import hm1
from hm1 import FoveaNet, infer, split_folds, N_FOLDS, CKPT, CELL2CANON
from common import train_items, test_items, WORK


def load_state(tag, k, snap=False):
    if snap:
        p = os.path.join(CKPT, f'{tag}_fold{k}_s{snap}.pt')
    else:
        p = os.path.join(CKPT, f'{tag}_fold{k}.pt')
    ck = torch.load(p, map_location='cuda', weights_only=False)
    m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def collect(tag='hm1', tta=True, nsnap=0, save=True):
    items, folds = split_folds()
    fold_of = np.zeros(len(items), int)
    for k in range(N_FOLDS):
        fold_of[folds[k]] = k

    te = test_items()
    keys = ['best'] + [f's{i}' for i in range(nsnap)]
    oof_hm = np.zeros((len(items), 2))
    oof_rg = np.zeros((len(items), 2))
    te_hm = np.zeros((len(te), 2))
    te_rg = np.zeros((len(te), 2))
    nk = 0
    for key in keys:
        try:
            acc_o = np.zeros((len(items), 2))
            acc_ot = np.zeros((len(items), 2))
            acc_t = np.zeros((len(te), 2))
            acc_tt = np.zeros((len(te), 2))
            for k in range(N_FOLDS):
                m = load_state(tag, k, None if key == 'best' else int(key[1:]))
                va = [items[i] for i in folds[k]]
                h, r = infer(m, va, 'train', tta=tta)
                acc_o[folds[k]] = h
                acc_ot[folds[k]] = r
                h, r = infer(m, te, 'test', tta=tta)
                acc_t += h / N_FOLDS
                acc_tt += r / N_FOLDS
                del m
                torch.cuda.empty_cache()
            s = hm1.rmse(acc_o, items)[0]
            print(f'  {tag} {key:5s} OOF {s:6.2f}', flush=True)
            oof_hm += acc_o
            oof_rg += acc_ot
            te_hm += acc_t
            te_rg += acc_tt
            nk += 1
        except FileNotFoundError:
            print('  (no snapshot %s)' % key)
            break
    oof_hm /= nk; oof_rg /= nk; te_hm /= nk; te_rg /= nk

    s_hm = hm1.rmse(oof_hm, items)
    s_rg = hm1.rmse(oof_rg, items)
    print(f'{tag} AVG({nk}) heatmap OOF {s_hm[0]:.2f} (x {s_hm[1]:.1f} y {s_hm[2]:.1f})')
    print(f'{tag} AVG({nk}) regress OOF {s_rg[0]:.2f} (x {s_rg[1]:.1f} y {s_rg[2]:.1f})')

    if save:
        out = os.path.join(WORK, f'oof_{tag}.npz')
        np.savez(out, oof_hm=oof_hm, oof_rg=oof_rg, te_hm=te_hm, te_rg=te_rg,
                 fold_of=fold_of, nsnap=nk)
        print('wrote', out)
    return oof_hm, te_hm


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm1')
    ap.add_argument('--nsnap', type=int, default=0, help='#snapshots to also average')
    ap.add_argument('--no_tta', action='store_true')
    ap.add_argument('--mask_disc', type=float, default=0.0)
    a = ap.parse_args()
    hm1.MASK_DISC = a.mask_disc
    collect(a.tag, tta=not a.no_tta, nsnap=a.nsnap)

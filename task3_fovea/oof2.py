# -*- coding: utf-8 -*-
"""oof2.py -- collect leak-free stage-2 predictions (val = fold model, test = 5-fold
average) using the stage-1 out-of-fold predictions as crop centres.
Writes oof_<tag>.npz with the same layout as oof.py so final.py can use it.
"""
import os, argparse
import numpy as np
import torch

import hm1
from hm1 import split_folds, N_FOLDS, CKPT, FoveaNet
from hm2 import infer2, PATCH
from common import train_items, test_items, WORK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm2')
    ap.add_argument('--base', default='hm1')
    ap.add_argument('--mask_disc', type=float, default=0.0)
    a = ap.parse_args()
    hm1.MASK_DISC = a.mask_disc

    base = np.load(os.path.join(WORK, f'oof_{a.base}.npz'))
    oof_centers = base['oof_hm']
    te_centers = base['te_hm']

    items, folds = split_folds()
    te = test_items()
    oof = np.zeros((len(items), 2))
    te_acc = np.zeros((len(te), 2))

    for k in range(N_FOLDS):
        ck = torch.load(os.path.join(CKPT, f'{a.tag}_fold{k}.pt'), map_location='cuda',
                        weights_only=False)
        m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
        m.load_state_dict(ck['state'])
        m.eval()
        va = [items[i] for i in folds[k]]
        oof[folds[k]] = infer2(m, va, oof_centers[folds[k]], 'train')
        te_acc += infer2(m, te, te_centers, 'test') / N_FOLDS
        del m
        torch.cuda.empty_cache()

    s = hm1.rmse(oof, items)
    st = hm1.rmse(te_acc, []) if False else None
    print(f'{a.tag} OOF {s[0]:.2f} (x {s[1]:.2f} y {s[2]:.2f})')
    for k in range(N_FOLDS):
        sk = hm1.rmse(oof[folds[k]], [items[i] for i in folds[k]])
        print('  fold%d %.1f' % (k, sk[0]), end='')
    print()
    out = os.path.join(WORK, f'oof_{a.tag}.npz')
    np.savez(out, oof_hm=oof, te_hm=te_acc, oof_rg=oof, te_rg=te_acc)
    print('wrote', out)


if __name__ == '__main__':
    main()

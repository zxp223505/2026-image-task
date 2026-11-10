# -*- coding: utf-8 -*-
"""tta_eval.py -- multi-scale TTA for stage 2, evaluated honestly on the 5-fold OOF.

Zooming the patch by z means the network sees the fovea at a different scale;
averaging several scales is a standard, label-free way to harden a keypoint
regressor, and it can only help cases where one scale happens to fall into a
wrong local basin (which is what our low-contrast failures look like).

The canonical conversion must divide by z, because D is scaled by ZOOM:
    cell -> canonical px = STRIDE * PATCH / (SIZE2 * z)
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
from hm1 import FoveaNet, split_folds, CKPT, GATE
from hm2 import PatchDS, PATCH, SIZE2, STRIDE
from common import train_items, test_items, WORK


def load(tag, k, snap=None):
    name = f'{tag}_fold{k}' + ('' if snap is None else f'_s{snap}') + '.pt'
    ck = torch.load(os.path.join(CKPT, name), map_location='cuda', weights_only=False)
    # infer the input-channel count from the checkpoint so the same script works
    # for both the plain (3ch) and DoG (4ch) families
    hm1.DOG = (ck['state']['stem.0.weight'].shape[1] == 4)
    m = FoveaNet('resnet18').cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


def nsnaps(tag, k):
    n = 0
    while os.path.exists(os.path.join(CKPT, f'{tag}_fold{k}_s{n}.pt')):
        n += 1
    return n


@torch.no_grad()
def pred(m, geos, centers, split, flat=True):
    """canonical-frame coordinates for the given zoom (hm1.ZOOM)"""
    ds = PatchDS(geos, centers, split, train=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    out = []
    for x, _, _ in dl:
        acc = np.zeros((x.shape[0], 2))
        for mirror in (False, True):
            xx = torch.flip(x, dims=[3]) if mirror else x
            h, _ = m(xx.cuda())
            if mirror:
                h = torch.flip(h, dims=[3])
            acc += hm1.soft_argmax(h, gate=0.0 if flat else None).cpu().numpy() / 2
        out.append(acc)
    cell2canon = STRIDE * PATCH / (SIZE2 * hm1.ZOOM)
    off = np.concatenate(out) * cell2canon
    cen = np.clip(np.asarray(centers, float), PATCH / 2, hm1.CANON - PATCH / 2)
    return cen - PATCH / 2 + off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='dog2')
    ap.add_argument('--base', default='dogA')
    ap.add_argument('--zooms', default='1.0,1.1,0.9')
    ap.add_argument('--flat', type=int, default=1)
    a = ap.parse_args()

    items, folds = split_folds()
    base = np.load(os.path.join(WORK, f'mem3_{a.base}.npz'))
    oc = base['oof_canon']
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    zooms = [float(z) for z in a.zooms.split(',')]

    def sc(p):
        e = p - gt
        return float(np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2))

    results = {}
    for z in zooms:
        hm1.ZOOM = z
        pred_o = np.zeros((len(items), 2))
        for k in range(5):
            ns = nsnaps(a.tag, k)
            acc = np.zeros((len(folds[k]), 2))
            for s in range(ns):
                m = load(a.tag, k, s)
                acc += pred(m, [items[i] for i in folds[k]], oc[folds[k]], 'train',
                            bool(a.flat)) / ns
                del m; torch.cuda.empty_cache()
            pred_o[folds[k]] = acc
        results[z] = pred_o
        print('zoom %.2f  OOF %.2f' % (z, hm1.rmse(pred_o, items)[0]))

    hm1.ZOOM = 1.0
    if len(zooms) > 1:
        comb = np.mean([results[z] for z in zooms], axis=0)
        print('averaged %.2f' % hm1.rmse(comb, items)[0])
        for z in zooms:
            if z != 1.0:
                print('1.0 + %.2f   %.2f' % (z, hm1.rmse(0.5 * (results[1.0] + results[z]), items)[0]
                                             if 1.0 in results else 0))

    # save the all-zoom average at zoom 1.0 frame so later stages can use it
    np.savez(os.path.join(WORK, f'tta_{a.tag}.npz'),
             oof=np.mean([results[z] for z in zooms], axis=0))
    print('saved tta_%s.npz' % a.tag)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""late_avg.py -- how much of my OOF score is epoch-selection bias?

Every fold model was chosen as "the epoch with the lowest val RMSE" and then
scored on those same 16 val images, which is optimistically biased.  hm1b has
snapshots at fixed epochs 30..100, so we can compare:

  selected-best           (what I reported)         -> biased
  fixed epoch E           (no selection at all)     -> honest
  average of late epochs  (no selection, low var)   -> honest, usually best

This decides whether the submission should use selected checkpoints, and it
gives an honest number to quote.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
from hm1 import FoveaDS, FoveaNet, split_folds, N_FOLDS, CKPT
from common import train_items, WORK

TAG = 'hm1b'
EARLY = {0: 30, 1: 40, 2: 50, 3: 60, 4: 70, 5: 80, 6: 90, 7: 100}


def load(tag, k, snap=None):
    name = f'{tag}_fold{k}.pt' if snap is None else f'{tag}_fold{k}_s{snap}.pt'
    ck = torch.load(os.path.join(CKPT, name), map_location='cuda', weights_only=False)
    m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


@torch.no_grad()
def coords(m, geos):
    ds = FoveaDS(geos, 'train', train=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    out = []
    for x, _, _ in dl:
        acc = np.zeros((x.shape[0], 2))
        for fl in (False, True):
            xx = torch.flip(x, dims=[3]) if fl else x
            h, _ = m(xx.cuda())
            if fl:
                h = torch.flip(h, dims=[3])
            acc += hm1.soft_argmax(h).cpu().numpy() / 2
        out.append(acc)
    return np.concatenate(out) * hm1.CELL2CANON


def main():
    items, folds = split_folds()
    nsnap = len(EARLY)
    # coords[snap][fold] -> array over that fold's val items
    coords_s = {}
    for s in range(nsnap):
        coords_s[s] = []
        for k in range(N_FOLDS):
            m = load(TAG, k, s)
            coords_s[s].append(coords(m, [items[i] for i in folds[k]]))
            del m
            torch.cuda.empty_cache()
        print(f'  snapshot {s} (ep{EARLY[s]}) done', flush=True)

    coords_best = []
    for k in range(N_FOLDS):
        m = load(TAG, k, None)
        coords_best.append(coords(m, [items[i] for i in folds[k]]))
        del m
        torch.cuda.empty_cache()

    def pooled(per_fold):
        pred = np.zeros((len(items), 2))
        for k in range(N_FOLDS):
            pred[folds[k]] = per_fold[k]
        return hm1.rmse(pred, items)

    print('\n%-34s %8s %8s %8s' % ('protocol', 'RMSE', 'x', 'y'))
    s = pooled(coords_best)
    print('%-34s %8.2f %8.2f %8.2f   <-- biased (selected on val)' %
          ('selected best epoch', s[0], s[1], s[2]))
    for s_ in range(nsnap):
        r = pooled(coords_s[s_])
        print('%-34s %8.2f %8.2f %8.2f' % (f'fixed epoch {EARLY[s_]}', r[0], r[1], r[2]))

    for grp, nm in (((5, 6), 'avg ep80+90'), ((6, 7), 'avg ep90+100'),
                    ((5, 6, 7), 'avg ep80+90+100'), ((4, 5, 6, 7), 'avg ep70..100'),
                    (tuple(range(nsnap)), 'avg all 30..100')):
        per = [sum(coords_s[s_][k] for s_ in grp) / len(grp) for k in range(N_FOLDS)]
        r = pooled(per)
        print('%-34s %8.2f %8.2f %8.2f' % (nm, r[0], r[1], r[2]))


if __name__ == '__main__':
    main()

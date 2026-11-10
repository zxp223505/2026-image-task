# -*- coding: utf-8 -*-
"""tune_gate.py -- choose the extraction rule against the real pipeline.

Three ways to combine the two TTA views:
  coord : gated expectation per view, then average the (x,y)  [what infer() does]
  heat  : average the two heatmaps, then one gated expectation
  none  : no gating (baseline)
"""
import os, argparse
import numpy as np

from hm1 import split_folds, rmse, CELL2CANON
from common import WORK

H = 128


def gated(h, delta):
    n = h.shape[0]
    f = h.reshape(n, -1).astype(np.float64)
    if delta:
        mx = f.max(1, keepdims=True)
        f = np.where(f >= mx - delta, f, -1e30)
    e = np.exp(f - f.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    idx = np.arange(H * H)
    xs = (idx % H + 0.5)
    ys = (idx // H + 0.5)
    return np.stack([(p * xs).sum(1), (p * ys).sum(1)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm1')
    a = ap.parse_args()
    d = np.load(os.path.join(WORK, f'heat2_{a.tag}.npz'))
    v0, v1 = d['oof0'], d['oof1']
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])

    print('%-22s %8s %8s %8s' % ('rule', 'RMSE', 'x', 'y'))
    rows = []
    for mode in ('none', 'coord', 'heat'):
        for delta in (None, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            if mode == 'none':
                if delta is not None:
                    continue
            if mode == 'coord':
                c = 0.5 * (gated(v0, delta) + gated(v1, delta))
            elif mode == 'heat':
                c = gated(0.5 * (v0.astype(np.float64) + v1.astype(np.float64)), delta)
            else:
                c = 0.5 * (gated(v0, None) + gated(v1, None))
            s = rmse(c * CELL2CANON, items)
            nm = f'{mode} delta={delta}'
            rows.append((s[0], nm, mode, delta))
            print('%-22s %8.2f %8.2f %8.2f' % (nm, s[0], s[1], s[2]))

    rows.sort()
    print('\nBEST:', rows[0][1], '%.2f' % rows[0][0])

    # per-fold check for the winner vs baseline
    _, _, mode, delta = rows[0]
    if mode == 'coord':
        cbest = 0.5 * (gated(v0, delta) + gated(v1, delta))
    elif mode == 'heat':
        cbest = gated(0.5 * (v0.astype(np.float64) + v1.astype(np.float64)), delta)
    else:
        cbest = 0.5 * (gated(v0, None) + gated(v1, None))
    cbase = 0.5 * (gated(v0, None) + gated(v1, None))
    print('\nfold    base   best')
    for k in range(5):
        sub = [items[i] for i in folds[k]]
        print('  %d  %6.2f %6.2f' % (k, rmse(cbase[folds[k]] * CELL2CANON, sub)[0],
                                     rmse(cbest[folds[k]] * CELL2CANON, sub)[0]))


if __name__ == '__main__':
    main()

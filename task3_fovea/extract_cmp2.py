# -*- coding: utf-8 -*-
"""extract_cmp2.py -- fine-tune the peak-gated heatmap extraction rule."""
import os
import numpy as np
from scipy import ndimage

from common import WORK
from hm1 import split_folds, rmse, CELL2CANON, OUTSZ

H = OUTSZ
_EPS = 1e-12


def _softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def ext_gate(h, delta):
    n = h.shape[0]
    flat = h.reshape(n, -1)
    mx = flat.max(1, keepdims=True)
    g = np.where(flat >= mx - delta, flat, -1e30)
    p = _softmax(g)
    idx = np.arange(H * H)
    xs = (idx % H + 0.5)
    ys = (idx // H + 0.5)
    return np.stack([(p * xs).sum(1), (p * ys).sum(1)], 1)


def ext_comp(h, delta):
    """argmax, then the connected component of the peak above (max-delta); centroid."""
    n = h.shape[0]
    out = np.zeros((n, 2))
    for i in range(n):
        hi = h[i]
        m = hi.max()
        mask = hi >= m - delta
        lab, _ = ndimage.label(mask)
        am = int(np.argmax(hi))
        ay, ax = divmod(am, H)
        comp = lab[ay, ax]
        if comp == 0:
            out[i] = (ax + 0.5, ay + 0.5)
            continue
        sel = (lab == comp)
        w = np.exp(hi - m) * sel
        yy, xx = np.nonzero(sel)
        ws = w[yy, xx]
        out[i] = ((xx + 0.5) * ws).sum() / ws.sum(), ((yy + 0.5) * ws).sum() / ws.sum()
    return out


def main():
    d = np.load(os.path.join(WORK, 'heat_hm1.npz'))
    oof = d['oof']
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])

    res = []
    print('%-14s %8s %8s %8s' % ('rule', 'RMSE', 'x', 'y'))
    for dl in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        s = rmse(ext_gate(oof, dl) * CELL2CANON, items)
        res.append((s[0], f'gate{dl}'))
        print('%-14s %8.2f %8.2f %8.2f' % (f'gate{dl}', s[0], s[1], s[2]))
    for dl in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        s = rmse(ext_comp(oof, dl) * CELL2CANON, items)
        res.append((s[0], f'comp{dl}'))
        print('%-14s %8.2f %8.2f %8.2f' % (f'comp{dl}', s[0], s[1], s[2]))
    res.sort()
    print('\nBEST', res[0][1], '%.2f' % res[0][0])

    # blend of gate4 and comp
    a = ext_gate(oof, 4.0)
    b = ext_comp(oof, 4.0)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = rmse((w * a + (1 - w) * b) * CELL2CANON, items)
        print('  mix gate4*%.2f + comp4*%.2f -> %.2f' % (w, 1 - w, s[0]))


if __name__ == '__main__':
    main()

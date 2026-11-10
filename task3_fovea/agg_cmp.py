# -*- coding: utf-8 -*-
"""agg_cmp.py -- compare fixed-weight blending with robust per-image aggregation.

Motivation: on the hardest images a *single* member can be the outlier (e.g. hm1b
on 0072/0016) while the others agree.  A weighted mean still drags towards it; a
per-image median discards it.
"""
import itertools
import numpy as np
import os

from common import WORK
from hm1 import split_folds

SPEC = [('hm1', 'gg'), ('hm1b', 'gf'), ('hm2', 'gf'), ('hm2b', 'gg')]


def load(tag, var):
    d = np.load(os.path.join(WORK, f'mem2_{tag}.npz'))
    oof = np.column_stack([d['oof_g' if var[0] == 'g' else 'oof_f'][:, 0],
                           d['oof_g' if var[1] == 'g' else 'oof_f'][:, 1]])
    return oof


def score(p, gt):
    e = p - gt
    rx = float(np.sqrt(np.mean(e[:, 0] ** 2)))
    ry = float(np.sqrt(np.mean(e[:, 1] ** 2)))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2.0)), rx, ry


def main():
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    names = [t for t, _ in SPEC]
    P = np.stack([load(t, v) for t, v in SPEC])          # (M, N, 2)

    print('%-28s %8s %8s %8s' % ('aggregation', 'RMSE', 'x', 'y'))
    s = score(P.mean(0), gt)
    print('%-28s %8.2f %8.2f %8.2f' % ('mean all 4', s[0], s[1], s[2]))
    s = score(np.median(P, axis=0), gt)
    print('%-28s %8.2f %8.2f %8.2f' % ('median all 4', s[0], s[1], s[2]))

    # 3-member subsets (median is a true order statistic there)
    for sub in itertools.combinations(range(len(SPEC)), 3):
        nm = '+'.join(names[i] for i in sub)
        s = score(np.median(P[list(sub)], axis=0), gt)
        print('%-28s %8.2f %8.2f %8.2f' % ('median ' + nm, s[0], s[1], s[2]))
        s = score(P[list(sub)].mean(0), gt)
        print('%-28s %8.2f %8.2f %8.2f' % ('mean   ' + nm, s[0], s[1], s[2]))

    # weighted mean, best over a coarse grid (reference)
    best = (1e18, None)
    grid = np.arange(0, 1.0001, 0.1)
    for w in itertools.product(grid, repeat=4):
        if abs(sum(w) - 1) > 1e-9:
            continue
        v = score(np.tensordot(np.array(w), P, axes=1), gt)[0]
        if v < best[0]:
            best = (v, w)
    print('\n%-28s %8.2f' % ('weighted mean (grid)', best[0]), best[1])

    # weighted median / trimmed mean over all 4
    srt = np.sort(P, axis=0)
    s = score(0.5 * (srt[1] + srt[2]), gt)               # median 4
    print('%-28s %8.2f %8.2f %8.2f' % ('mean of middle 2', s[0], s[1], s[2]))
    s = score((P.sum(0) - P.min(0) - P.max(0)) / 2.0, gt)  # drop min & max per axis
    print('%-28s %8.2f %8.2f %8.2f' % ('trimmed mean (drop 1+1)', s[0], s[1], s[2]))

    # per-image: which is better, mean-of-4 or median-of-4?
    m = np.linalg.norm(P.mean(0) - gt, axis=1)
    md = np.linalg.norm(np.median(P, axis=0) - gt, axis=1)
    print('\nimages where median beats mean by >20px:')
    for i in np.argsort(-(m - md))[:6]:
        print('  %-9s mean %6.1f  median %6.1f' %
              (os.path.basename(items[i].path), m[i], md[i]))


if __name__ == '__main__':
    main()

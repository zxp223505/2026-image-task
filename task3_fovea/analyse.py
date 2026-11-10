# -*- coding: utf-8 -*-
"""analyse.py -- per-image OOF error analysis in ORIGINAL pixel coordinates."""
import os, sys
import numpy as np

from common import WORK, test_items
from hm1 import split_folds, rmse


def load(tag):
    return np.load(os.path.join(WORK, f'oof_{tag}.npz'))


def to_orig_arr(p, items):
    out = np.zeros_like(p)
    for i, g in enumerate(items):
        out[i] = g.to_orig(p[i, 0], p[i, 1])
    return out


def main(tags):
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    fold_of = np.zeros(len(items), int)
    for k in range(5):
        fold_of[folds[k]] = k

    for t in tags:
        d = load(t)
        p = to_orig_arr(d['oof_hm'], items)
        e = p - gt
        dist = np.linalg.norm(e, axis=1)
        s = rmse(d['oof_hm'], items)
        print(f'\n=== {t}: pooled RMSE {s[0]:.2f} (x {s[1]:.2f} y {s[2]:.2f}) ===')
        print('  median %.1f  p75 %.1f  p90 %.1f  max %.1f'
              % (np.median(dist), np.percentile(dist, 75), np.percentile(dist, 90), dist.max()))
        print('  bias  x %+.1f  y %+.1f   resid std x %.1f y %.1f'
              % (e[:, 0].mean(), e[:, 1].mean(), e[:, 0].std(), e[:, 1].std()))
        print('  pred std x %.1f (gt %.1f)   y %.1f (gt %.1f)'
              % (p[:, 0].std(), gt[:, 0].std(), p[:, 1].std(), gt[:, 1].std()))
        order = np.argsort(-dist)
        print('  worst:')
        for i in order[:10]:
            print('    %-9s fold%d  ex %7.1f  ey %7.1f  dist %7.1f'
                  % (os.path.basename(items[i].path), fold_of[i], e[i, 0], e[i, 1], dist[i]))
        # per-fold pooled
        for k in range(5):
            print('    fold%d rmse %.1f' % (k, rmse(d['oof_hm'][folds[k]],
                                                    [items[i] for i in folds[k]])[0]), end='')
        print()

    # ---- calibration scan on the LAST tag (the one we would use) ----
    d = load(tags[-1])
    p = d['oof_hm']
    print('\n--- calibration (in-sample, 2 params/axis) ---')
    pc = np.zeros_like(p)
    for ax, nm in ((0, 'x'), (1, 'y')):
        A = np.column_stack([p[:, ax], np.ones(len(p))])
        coef, *_ = np.linalg.lstsq(A, gt[:, ax], rcond=None)
        pc[:, ax] = p[:, ax] * coef[0] + coef[1]
        print(f'  {nm}: a={coef[0]:.4f} b={coef[1]:+.2f}')
    print(f'  raw {rmse(p, items)[0]:.2f}  ->  calibrated {rmse(pc, items)[0]:.2f}')
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()
    print('--- shrink x only (y raw) ---')
    for a in (0.85, 0.9, 0.95, 1.0, 1.05):
        q = p.copy(); pc2 = to_orig_arr(q, items)
        pc2[:, 0] = mx + a * (pc2[:, 0] - mx)
        qq = np.zeros_like(p)
        for i, g in enumerate(items):
            qq[i] = g.to_canon(pc2[i, 0], pc2[i, 1])
        print(f'  sx={a:.2f} -> {rmse(qq, items)[0]:.2f}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['hm1'])

# -*- coding: utf-8 -*-
"""anat_constraint.py -- clamp the prediction to the anatomically plausible
region relative to the optic disc.

The disc sits ~0.70 R from the fovea, almost purely horizontally, and the fovea
is always temporal to it.  Measured on train targets:
    |dx|/R = 0.704 +- 0.060,  dy/R = +0.092 +- 0.064
So any prediction whose disc-relative offset is far outside that is a gross
failure.  Projecting it back onto the plausible range turns a ~190 px disaster
into a ~60 px error -- and the worst 1-2 OOF images carry ~70% of the total
squared error, so this can matter a lot.
"""
import os
import numpy as np

from common import WORK
from hm1 import split_folds


def score(p, gt):
    e = p - gt
    rx = np.sqrt(np.mean(e[:, 0] ** 2))
    ry = np.sqrt(np.mean(e[:, 1] ** 2))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2)), float(rx), float(ry)


def rel(pred, items):
    """disc-relative offsets in units of the fundus radius."""
    out = np.zeros_like(pred)
    for i, g in enumerate(items):
        out[i, 0] = (pred[i, 0] - g.disc_x) / g.fov_r
        out[i, 1] = (pred[i, 1] - g.disc_y) / g.fov_r
    return out


def clamp(pred, items, ax_lo, ax_hi, ay_lo, ay_hi):
    """clip the disc-relative offset into [lo,hi] while keeping the predicted side"""
    out = pred.copy()
    r = rel(pred, items)
    for i, g in enumerate(items):
        dx, dy = r[i]
        mag = abs(dx)
        mag = np.clip(mag, ax_lo, ax_hi)
        dx = np.sign(dx if dx != 0 else 1.0) * mag
        dy = np.clip(dy, ay_lo, ay_hi)
        out[i, 0] = g.disc_x + dx * g.fov_r
        out[i, 1] = g.disc_y + dy * g.fov_r
    return out


def main():
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    d = np.load(os.path.join(WORK, 'final_oof4.npz'))
    pred = d['oof']

    gtr = rel(gt, items)
    print('train GT disc-relative (units of R):')
    print('  |dx| %.3f..%.3f  mean %.3f std %.3f'
          % (np.abs(gtr[:, 0]).min(), np.abs(gtr[:, 0]).max(),
             np.abs(gtr[:, 0]).mean(), np.abs(gtr[:, 0]).std()))
    print('  dy   %+.3f..%+.3f  mean %+.3f std %.3f'
          % (gtr[:, 1].min(), gtr[:, 1].max(), gtr[:, 1].mean(), gtr[:, 1].std()))
    pr = rel(pred, items)
    print('\npredicted disc-relative:')
    print('  |dx| %.3f..%.3f   dy %+.3f..%+.3f'
          % (np.abs(pr[:, 0]).min(), np.abs(pr[:, 0]).max(), pr[:, 1].min(), pr[:, 1].max()))
    bad = np.abs(pr[:, 0]) < 0.5
    print('  predictions with |dx|/R < 0.5 (implausible): %d' % bad.sum())
    for i in np.nonzero(bad)[0]:
        print('    %-9s |dx|/R %.3f dy/R %+.3f  err %.0f px'
              % (os.path.basename(items[i].path), abs(pr[i, 0]), pr[i, 1],
                 np.linalg.norm(pred[i] - gt[i])))

    print('\nbase                     %8.2f %8.2f %8.2f' % (score(pred, gt)[0], *score(pred, gt)[1:]))
    for ax_lo in (0.45, 0.50, 0.55):
        for ax_hi in (0.85, 0.95):
            for ay_lo, ay_hi in ((-0.10, 0.26), (-0.20, 0.35)):
                p = clamp(pred, items, ax_lo, ax_hi, ay_lo, ay_hi)
                s = score(p, gt)
                print('clamp |dx|/R in [%.2f,%.2f] dy in [%+.2f,%+.2f]  %8.2f %8.2f %8.2f'
                      % (ax_lo, ax_hi, ay_lo, ay_hi, s[0], s[1], s[2]))


if __name__ == '__main__':
    main()

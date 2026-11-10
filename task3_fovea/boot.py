# -*- coding: utf-8 -*-
"""boot.py -- what score would this submission plausibly get on the leaderboard?

The competition scores 20 test images (10 public + 10 private) with
RMSE = sqrt(mean(ex^2) + mean(ey^2)) / 2 style pooling.  Our estimate comes from
80 out-of-fold train images, but the *test* score is a statistic over only 20
(and for the final ranking, 10) images.  Resampling our per-image errors shows
how much the leaderboard can legitimately differ from the OOF number.
"""
import os, sys
import numpy as np

from common import WORK


def rmse_of(ex, ey):
    return np.sqrt((np.mean(ex ** 2) + np.mean(ey ** 2)) / 2.0)


def main():
    d = np.load(os.path.join(WORK, sys.argv[1] if len(sys.argv) > 1 else 'final_oof4.npz'))
    oof, gt = d['oof'], d['gt']
    e = oof - gt
    ex, ey = e[:, 0], e[:, 1]
    rng = np.random.default_rng(0)

    print('full 80-image OOF RMSE          : %.2f' % rmse_of(ex, ey))
    print('mean per-image distance         : %.1f px' % np.mean(np.linalg.norm(e, axis=1)))
    print('median per-image distance       : %.1f px' % np.median(np.linalg.norm(e, axis=1)))

    for n, label in ((20, 'test set (public+private)'), (10, 'private LB only')):
        vals = np.empty(20000)
        for i in range(20000):
            idx = rng.integers(0, len(ex), n)
            vals[i] = rmse_of(ex[idx], ey[idx])
        q = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f'\n--- resampled {n} images ({label}), 20000 draws ---')
        print('  median (best guess) %.1f' % q[2])
        print('  50%% central range  %.1f .. %.1f' % (q[1], q[3]))
        print('  90%% central range  %.1f .. %.1f' % (q[0], q[4]))
        print('  P(score < 20) = %.2f' % float((vals < 20).mean()))
        print('  P(score < 18) = %.2f' % float((vals < 18).mean()))

    # how much do the worst images matter?
    dist = np.linalg.norm(e, axis=1)
    order = np.argsort(-dist)
    tot = (np.sum(ex ** 2) + np.sum(ey ** 2)) / 2
    print('\ncontribution to the pooled squared error:')
    for r in (1, 2, 3, 4, 5):
        idx = order[:r]
        part = (np.sum(ex[idx] ** 2) + np.sum(ey[idx] ** 2)) / 2
        print(f'  worst {r} image(s): {100 * part / tot:5.1f}% of total')

    # what if the worst images were merely average?
    for r in (3, 4, 5):
        ex2, ey2 = ex.copy(), ey.copy()
        idx = order[:r]
        ex2[idx] = np.median(np.abs(ex))
        ey2[idx] = np.median(np.abs(ey))
        print(f'  if worst {r} were median-quality -> OOF {rmse_of(ex2, ey2):.2f}')


if __name__ == '__main__':
    main()

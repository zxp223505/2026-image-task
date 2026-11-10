# -*- coding: utf-8 -*-
"""disagree.py -- use stage-1/stage-2 disagreement as an uncertainty signal.

The entropy shrink cannot detect our worst failures: on 0072/0049/0024 the model
is *confidently* wrong (entropy 0.50-0.68 vs a 0.43 median).  But the two stages
often disagree on exactly those images, and disagreement is a classic ensemble
uncertainty measure.  Test whether shrinking on disagreement beats shrinking on
entropy, or on both.
"""
import os
import numpy as np

from common import WORK
from hm1 import split_folds, N_FOLDS


def score(p, gt):
    e = p - gt
    rx = np.sqrt(np.mean(e[:, 0] ** 2)); ry = np.sqrt(np.mean(e[:, 1] ** 2))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2)), float(rx), float(ry)


def unorm(v):
    return np.clip((v - v.min()) / max(v.max() - v.min(), 1e-9), 0, 1)


def shrink(p, u, mx, my, s_hi, s_lo):
    s = s_hi + (s_lo - s_hi) * u
    o = p.copy()
    o[:, 0] = mx + s * (p[:, 0] - mx)
    o[:, 1] = my + s * (p[:, 1] - my)
    return o


def main():
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()

    s2 = np.load(os.path.join(WORK, 'mem3_dog2.npz'))      # stage 2 (best member)
    s1 = np.load(os.path.join(WORK, 'mem3_dogA.npz'))      # stage 1
    p2 = s2['oof_f']
    p1 = s1['oof_g']
    ent = s2['oof_conf']

    # per-image disagreement in units of the fundus radius
    R = np.array([g.fov_r for g in items])
    dis = np.linalg.norm(p1 - p2, axis=1) / R

    e = np.linalg.norm(p2 - gt, axis=1)
    print('corr(entropy, err)      = %.3f' % np.corrcoef(ent, e)[0, 1])
    print('corr(disagreement, err) = %.3f' % np.corrcoef(dis, e)[0, 1])
    order = np.argsort(-e)
    print('\nworst 6 images:')
    for i in order[:6]:
        print('  %-9s err %6.1f   entropy %.3f   disagreement %.1f px'
              % (os.path.basename(items[i].path), e[i], ent[i], dis[i] * R[i]))
    print('\nmedian disagreement %.1f px, p90 %.1f px'
          % (np.median(dis * R), np.percentile(dis * R, 90)))

    print('\n%-30s %8s' % ('scheme', 'OOF'))
    print('%-30s %8.2f' % ('no shrink', score(p2, gt)[0]))

    for u, nm in ((unorm(ent), 'entropy'), (unorm(dis), 'disagreement'),
                  (0.5 * unorm(ent) + 0.5 * unorm(dis), 'both 50/50'),
                  (0.3 * unorm(ent) + 0.7 * unorm(dis), 'both 30/70')):
        best = (1e9, None)
        for s_lo in np.arange(0.60, 1.001, 0.02):
            v = score(shrink(p2, u, mx, my, 1.02, s_lo), gt)[0]
            if v < best[0]:
                best = (v, s_lo)
        print('%-30s %8.2f   (best s_lo %.2f)' % (nm, best[0], best[1]))


if __name__ == '__main__':
    main()

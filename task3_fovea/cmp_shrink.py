# -*- coding: utf-8 -*-
"""cmp_shrink.py -- fair comparison of shrink signals (both cross-fitted)."""
import os
import numpy as np

from common import WORK
from hm1 import split_folds, N_FOLDS


def main():
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()

    s2 = np.load(os.path.join(WORK, 'mem3_dog2.npz'))
    s1 = np.load(os.path.join(WORK, 'mem3_dogA.npz'))
    p2, ent = s2['oof_f'], s2['oof_conf']
    R = np.array([g.fov_r for g in items])
    dis = np.linalg.norm(s1['oof_g'] - p2, axis=1) / R

    def sc(p, g):
        e = p - g
        return float(np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2))

    def sh(p, u, slo, shi=1.02):
        s = shi + (slo - shi) * u
        o = p.copy()
        o[:, 0] = mx + s * (p[:, 0] - mx)
        o[:, 1] = my + s * (p[:, 1] - my)
        return o

    def un(v):
        return np.clip((v - v.min()) / max(v.max() - v.min(), 1e-9), 0, 1)

    print('%-10s %10s %14s' % ('signal', 'in-sample', 'cross-fitted'))
    print('%-10s %10.2f %14s' % ('none', sc(p2, gt), '-'))
    grid = np.arange(0.6, 1.001, 0.02)
    for nm, u in (('entropy', un(ent)), ('disagree', un(dis)),
                  ('avg', 0.5 * un(ent) + 0.5 * un(dis))):
        ins = min(sc(sh(p2, u, s), gt) for s in grid)
        p = p2.copy()
        for k in range(N_FOLDS):
            ti = np.concatenate([folds[f] for f in range(N_FOLDS) if f != k])
            b = min(((sc(sh(p2[ti], u[ti], s), gt[ti]), s) for s in grid))
            p[folds[k]] = sh(p2[folds[k]], u[folds[k]], b[1])
        print('%-10s %10.2f %14.2f' % (nm, ins, sc(p, gt)))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""assemble_disagree.py -- final submission with disagreement-based shrinkage.

Uncertainty signal = distance between the stage-1 and stage-2 predictions
(in units of the fundus radius).  On the honest OOF this correlates with the
true error better than the heatmap entropy (0.65 vs 0.52) and, crucially, it
does flag the "confidently wrong" images that entropy misses (0024: 123 px
disagreement despite high confidence).

The mapping (min/max normalisation and s_lo) is calibrated on the 5-fold
out-of-fold predictions and then applied unchanged to the all-data test
predictions.  A 4-folds-fit / 1-fold-score estimate is printed as the honest
score.
"""
import os, csv, argparse
import numpy as np

from common import WORK, ROOT, test_items, train_items
from hm1 import split_folds, N_FOLDS

SAMPLE = os.path.join(ROOT, 'sample_submission.csv')
OUTDIR = os.path.join(os.path.dirname(WORK), 'out')


def score(p, gt):
    e = p - gt
    rx = np.sqrt(np.mean(e[:, 0] ** 2)); ry = np.sqrt(np.mean(e[:, 1] ** 2))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2)), float(rx), float(ry)


def shrink(p, u, mx, my, s_hi, s_lo):
    s = s_hi + (s_lo - s_hi) * u
    o = p.copy()
    o[:, 0] = mx + s * (p[:, 0] - mx)
    o[:, 1] = my + s * (p[:, 1] - my)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1', default='dogA')
    ap.add_argument('--s2', default='dog2')
    ap.add_argument('--full1', default='d1')
    ap.add_argument('--full2', default='d2')
    ap.add_argument('--out', default='submission.csv')
    a = ap.parse_args()

    items, folds = split_folds()
    te = test_items()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()

    # ---- OOF calibration (5-fold, leak-free) ----
    m1 = np.load(os.path.join(WORK, f'mem3_{a.s1}.npz'))
    m2 = np.load(os.path.join(WORK, f'mem3_{a.s2}.npz'))
    p1, p2 = m1['oof_g'], m2['oof_f']
    R = np.array([g.fov_r for g in items])
    dis = np.linalg.norm(p1 - p2, axis=1) / R
    lo, hi = dis.min(), dis.max()
    u = np.clip((dis - lo) / (hi - lo), 0, 1)

    print(f'OOF: stage2 alone {score(p2, gt)[0]:.2f}')
    best = (1e9, None)
    for s_lo in np.arange(0.6, 1.001, 0.02):
        v = score(shrink(p2, u, mx, my, 1.02, s_lo), gt)[0]
        if v < best[0]:
            best = (v, s_lo)
    # honest: pick s_lo on 4 folds, score the 5th
    p_cf = p2.copy()
    for k in range(N_FOLDS):
        ti = np.concatenate([folds[f] for f in range(N_FOLDS) if f != k])
        b = (1e9, None)
        for s_lo in np.arange(0.6, 1.001, 0.02):
            v = score(shrink(p2[ti], u[ti], mx, my, 1.02, s_lo), gt[ti])[0]
            if v < b[0]:
                b = (v, s_lo)
        p_cf[folds[k]] = shrink(p2[folds[k]], u[folds[k]], mx, my, 1.02, b[1])
    print(f'OOF: + shrink, s_lo={best[1]:.2f} -> {best[0]:.2f}   '
          f'(cross-fitted {score(p_cf, gt)[0]:.2f})')

    # ---- data-driven choice: does one global constant beat the shrink? ----
    const = np.tile([mx, my], (len(gt), 1))
    print(f'reference: constant-mean predictor {score(const, gt)[0]:.2f}')

    # ---- apply to the all-data test predictions ----
    f1 = np.load(os.path.join(WORK, f'full_{a.full1}.npz'))
    f2 = np.load(os.path.join(WORK, f'full_{a.full2}.npz'))
    t1, t2 = f1['test_orig'], f2['test_flat']
    Rt = np.array([g.fov_r for g in te])
    dis_t = np.linalg.norm(t1 - t2, axis=1) / Rt
    u_t = np.clip((dis_t - lo) / (hi - lo), 0, 1)
    print(f'test disagreement: median {np.median(dis_t * Rt):.1f} px  '
          f'max {np.max(dis_t * Rt):.1f} px   (OOF median {np.median(dis * R):.1f} px)')

    out = shrink(t2, u_t, mx, my, 1.02, best[1])
    print(f'test x {out[:, 0].min():.0f}..{out[:, 0].max():.0f}   '
          f'y {out[:, 1].min():.0f}..{out[:, 1].max():.0f}')

    # cross-check against the previously shipped file
    prev = os.path.join(OUTDIR, 'submission_v9.csv')
    if os.path.exists(prev):
        old = {}
        with open(prev) as f:
            for r in csv.DictReader(f):
                n, ax = r['ImageID'].split('_Fovea_')
                old.setdefault(int(n), {})[ax] = float(r['value'])
        diffs = np.array([np.hypot(p[0] - old[int(os.path.basename(g.path).split('.')[0])]['X'],
                                   p[1] - old[int(os.path.basename(g.path).split('.')[0])]['Y'])
                          for g, p in zip(te, out)])
        print(f'disagreement with v9: median {np.median(diffs):.1f} px, max {diffs.max():.1f} px')

    num2 = {int(os.path.basename(g.path).split('.')[0]): (float(p[0]), float(p[1]))
            for g, p in zip(te, out)}
    with open(SAMPLE) as f:
        rd = csv.reader(f); header = next(rd); rows = list(rd)
    outp = os.path.join(OUTDIR, a.out)
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r[0]
            ns, ax = iid.split('_Fovea_')
            fx, fy = num2[int(ns)]
            w.writerow([iid, '%.6f' % (fx if ax == 'X' else fy)])
    print('wrote', outp, len(rows), 'rows')


if __name__ == '__main__':
    main()

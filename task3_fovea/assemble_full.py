# -*- coding: utf-8 -*-
"""assemble_full.py -- build the submission from the all-data models.

Uses the stage-2 member (flat extraction, which won on the honest OOF) and the
confidence shrink whose parameters were validated on the honest 5-fold OOF
(s_hi=1.02, s_lo=0.80).  Also cross-checks against the verified 5-fold
predictions so a silently broken run cannot slip through.
"""
import os, csv, argparse
import numpy as np

from common import WORK, ROOT, test_items
from hm1 import split_folds

SAMPLE = os.path.join(ROOT, 'sample_submission.csv')
OUTDIR = os.path.join(os.path.dirname(WORK), 'out')


def unorm(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def score(p, gt):
    e = p - gt
    rx = np.sqrt(np.mean(e[:, 0] ** 2)); ry = np.sqrt(np.mean(e[:, 1] ** 2))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2)), float(rx), float(ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', default='f2')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--s_hi', type=float, default=1.02)
    ap.add_argument('--s_lo', type=float, default=0.88)
    ap.add_argument('--compare', default='submission_v8.csv')
    a = ap.parse_args()

    d = np.load(os.path.join(WORK, f'full_{a.stage2}.npz'))
    te = test_items()
    pred = d['test_flat']                     # flat won on stage 2
    conf = d['test_conf']
    print(f"stage-2 all-data members: {int(d['n_models'])} snapshots")

    items, _ = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()

    u = unorm(conf, conf.min(), conf.max())
    s = a.s_hi + (a.s_lo - a.s_hi) * u
    out = pred.copy()
    out[:, 0] = mx + s * (pred[:, 0] - mx)
    out[:, 1] = my + s * (pred[:, 1] - my)

    # ---- cross-check against the verified 5-fold (v7) predictions ----
    v7 = os.path.join(OUTDIR, a.compare)
    if os.path.exists(v7):
        old = {}
        with open(v7) as f:
            for r in csv.DictReader(f):
                n, ax = r['ImageID'].split('_Fovea_')
                old.setdefault(int(n), {})[ax] = float(r['value'])
        diffs = []
        for g, p in zip(te, out):
            n = int(os.path.basename(g.path).split('.')[0])
            diffs.append(np.hypot(p[0] - old[n]['X'], p[1] - old[n]['Y']))
        diffs = np.array(diffs)
        print(f'disagreement with the 5-fold submission: median {np.median(diffs):.1f} px, '
              f'max {diffs.max():.1f} px')
        if diffs.max() > 250:
            print('  !! large disagreement - check the run before trusting it')

    print(f'test x {out[:, 0].min():.0f}..{out[:, 0].max():.0f}   '
          f'y {out[:, 1].min():.0f}..{out[:, 1].max():.0f}')
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

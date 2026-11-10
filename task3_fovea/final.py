# -*- coding: utf-8 -*-
"""final.py -- blend stages, calibrate, and write the submission CSV.

Everything runs in ORIGINAL image pixels (that is what the leaderboard scores),
fitted only on leak-free out-of-fold predictions.
"""
import os, csv, argparse, itertools
import numpy as np

from common import WORK, ROOT, test_items
from hm1 import split_folds, N_FOLDS

SAMPLE = os.path.join(ROOT, 'sample_submission.csv')
OUTDIR = os.path.join(os.path.dirname(WORK), 'out')
os.makedirs(OUTDIR, exist_ok=True)


def to_orig(p, items):
    out = np.zeros_like(p, dtype=np.float64)
    for i, g in enumerate(items):
        out[i] = g.to_orig(p[i, 0], p[i, 1])
    return out


def score_rows(pred, gt):
    """Leaderboard metric: RMSE over the X/Y split rows."""
    e = pred - gt
    rx = float(np.sqrt(np.mean(e[:, 0] ** 2)))
    ry = float(np.sqrt(np.mean(e[:, 1] ** 2)))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2.0)), rx, ry


def fit_linear(pred, gt):
    A = np.column_stack([pred, np.ones(len(pred))])
    coef, *_ = np.linalg.lstsq(A, gt, rcond=None)
    return float(coef[0]), float(coef[1])


def cross_fitted(oof, gt, folds):
    out = np.zeros_like(oof)
    for k in range(N_FOLDS):
        ti = np.concatenate([folds[f] for f in range(N_FOLDS) if f != k])
        va = folds[k]
        for ax in range(2):
            a, b = fit_linear(oof[ti, ax], gt[ti, ax])
            out[va, ax] = oof[va, ax] * a + b
    return out


def load_tag(tag, items, te_items):
    p = os.path.join(WORK, f'oof_{tag}.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return to_orig(d['oof_hm'], items), to_orig(d['te_hm'], te_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='hm1,hm2')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--calib', default='auto', choices=['auto', 'linear', 'shrink', 'none'])
    ap.add_argument('--step', type=float, default=0.05)
    args = ap.parse_args()

    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    te_items = test_items()

    names, oofs, tes = [], [], []
    for t in args.tags.split(','):
        r = load_tag(t, items, te_items)
        if r is None:
            print('!! missing tag', t)
            continue
        names.append(t); oofs.append(r[0]); tes.append(r[1])
        s = score_rows(r[0], gt)
        print(f'  {t:8s} OOF {s[0]:7.2f}   (x {s[1]:6.2f}  y {s[2]:6.2f})')
    if not names:
        raise SystemExit('no tags found')

    if len(oofs) == 1:
        oof, te = oofs[0], tes[0]
        print('\nblend: single model')
    else:
        grid = np.arange(0, 1.0001, args.step)
        best = (1e18, None)
        for ws in itertools.product(grid, repeat=len(oofs)):
            if abs(sum(ws) - 1) > 1e-9:
                continue
            p = sum(w * o for w, o in zip(ws, oofs))
            s = score_rows(p, gt)[0]
            if s < best[0]:
                best = (s, ws)
        ws = best[1]
        print('\nblend weights', {n: round(w, 2) for n, w in zip(names, ws)},
              '-> OOF %.2f' % best[0])
        oof = sum(w * o for w, o in zip(ws, oofs))
        te = sum(w * o for w, o in zip(ws, tes))

    s = score_rows(oof, gt)
    print(f'BLEND OOF {s[0]:.2f}  (x {s[1]:.2f} y {s[2]:.2f})')
    dist = np.linalg.norm(oof - gt, axis=1)
    print('  median %.1f  p75 %.1f  p90 %.1f  max %.1f' %
          (np.median(dist), np.percentile(dist, 75), np.percentile(dist, 90), dist.max()))
    print('  pred std x %.1f (gt %.1f)  y %.1f (gt %.1f)'
          % (oof[:, 0].std(), gt[:, 0].std(), oof[:, 1].std(), gt[:, 1].std()))

    # ---------- calibration ----------
    mode = args.calib
    if mode == 'auto':
        mode = 'linear'
    if mode == 'linear':
        cf = cross_fitted(oof, gt, folds)
        sc = score_rows(cf, gt)
        print(f'\ncalib linear (5-fold cross-fitted, honest) {sc[0]:.2f} '
              f'(x {sc[1]:.2f} y {sc[2]:.2f})')
        ax_, bx = fit_linear(oof[:, 0], gt[:, 0])
        ay_, by = fit_linear(oof[:, 1], gt[:, 1])
        print(f'  x * {ax_:.4f} {bx:+.2f}   y * {ay_:.4f} {by:+.2f}')
        oof_c = np.column_stack([oof[:, 0] * ax_ + bx, oof[:, 1] * ay_ + by])
        te_c = np.column_stack([te[:, 0] * ax_ + bx, te[:, 1] * ay_ + by])
    elif mode == 'shrink':
        mx, my = gt[:, 0].mean(), gt[:, 1].mean()
        best = (1e18, None)
        for sx in np.arange(0.80, 1.21, 0.01):
            for sy in np.arange(0.80, 1.21, 0.01):
                p = np.column_stack([mx + sx * (oof[:, 0] - mx), my + sy * (oof[:, 1] - my)])
                v = score_rows(p, gt)[0]
                if v < best[0]:
                    best = (v, (sx, sy))
        sx, sy = best[1]
        print(f'\nshrink sx={sx:.2f} sy={sy:.2f} -> OOF {best[0]:.2f}')
        oof_c = np.column_stack([mx + sx * (oof[:, 0] - mx), my + sy * (oof[:, 1] - my)])
        te_c = np.column_stack([mx + sx * (te[:, 0] - mx), my + sy * (te[:, 1] - my)])
    else:
        oof_c, te_c = oof, te

    s = score_rows(oof_c, gt)
    print(f'FINAL OOF (in-sample) {s[0]:.2f}  (x {s[1]:.2f} y {s[2]:.2f})')
    dist = np.linalg.norm(oof_c - gt, axis=1)
    print('  worst:', [(os.path.basename(items[i].path), round(float(dist[i]), 1))
                       for i in np.argsort(-dist)[:6]])

    # ---------- write ----------
    num2pred = {}
    for g, p in zip(te_items, te_c):
        num2pred[int(os.path.basename(g.path).split('.')[0])] = (float(p[0]), float(p[1]))
    with open(SAMPLE) as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    outp = os.path.join(OUTDIR, args.out)
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r[0]
            num_s, axis = iid.split('_Fovea_')
            fx, fy = num2pred[int(num_s)]
            w.writerow([iid, '%.6f' % (fx if axis == 'X' else fy)])
    print('\nwrote', outp, len(rows), 'rows')
    # sanity
    print('test pred ranges: x %.0f..%.0f  y %.0f..%.0f'
          % (te_c[:, 0].min(), te_c[:, 0].max(), te_c[:, 1].min(), te_c[:, 1].max()))


if __name__ == '__main__':
    main()

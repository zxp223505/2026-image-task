# -*- coding: utf-8 -*-
"""final2.py -- assemble the submission from collected pipeline members.

Steps
  1. per-member blend weights (searched on OOF)
  2. normalise each member's heatmap entropy -> uncertainty u in [0,1]
  3. confidence-dependent shrink towards the training mean
        s(u) = s_hi + (s_lo - s_hi) * u
     (s_hi for the most confident images, s_lo for the least)
  4. honest estimate: weights + shrink fitted on 4 folds, scored on the 5th
  5. refit on all 80 OOF points, apply to the test predictions, write the CSV

Shrinking more where the model is unsure is the Bayes-optimal move for squared
error, and the heatmap entropy is a good uncertainty proxy (r=0.70 with |error|).
"""
import os, csv, argparse, itertools
import numpy as np

from common import WORK, ROOT, test_items
from hm1 import split_folds, N_FOLDS

SAMPLE = os.path.join(ROOT, 'sample_submission.csv')
OUTDIR = os.path.join(os.path.dirname(WORK), 'out')
os.makedirs(OUTDIR, exist_ok=True)


def score(pred, gt):
    e = pred - gt
    rx = float(np.sqrt(np.mean(e[:, 0] ** 2)))
    ry = float(np.sqrt(np.mean(e[:, 1] ** 2)))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2.0)), rx, ry


def load_members(tags):
    out = []
    for t in tags:
        p = os.path.join(WORK, f'mem_{t}.npz')
        if not os.path.exists(p):
            print('!! missing', p)
            continue
        d = np.load(p)
        out.append(dict(tag=t, oof=d['oof_pred'], te=d['te_pred'],
                        ooc=d['oof_conf'], tec=d['te_conf']))
    return out


def unorm(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def blend(mem, ws):
    oof = sum(w * m['oof'] for w, m in zip(ws, mem))
    te = sum(w * m['te'] for w, m in zip(ws, mem))
    return oof, te


def uncertainty(mem, ws):
    us = []
    for m in mem:
        lo, hi = m['ooc'].min(), m['ooc'].max()
        us.append((unorm(m['ooc'], lo, hi), unorm(m['tec'], lo, hi)))
    uo = sum(w * u[0] for w, u in zip(ws, us))
    ut = sum(w * u[1] for w, u in zip(ws, us))
    return uo, ut


def shrink(pred, u, mx, my, s_hi, s_lo):
    s = s_hi + (s_lo - s_hi) * u
    out = pred.copy()
    out[:, 0] = mx + s * (pred[:, 0] - mx)
    out[:, 1] = my + s * (pred[:, 1] - my)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='hm1,hm2')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--step', type=float, default=0.05)
    ap.add_argument('--damp', type=float, default=1.0,
                    help='damp the shrink 0..1 (1 = full fitted, 0 = none)')
    a = ap.parse_args()

    mem = load_members(a.tags.split(','))
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    te_items = test_items()
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()
    names = [m['tag'] for m in mem]

    for m in mem:
        s = score(m['oof'], gt)
        print(f"  {m['tag']:6s} OOF {s[0]:7.2f}  (x {s[1]:6.2f} y {s[2]:6.2f})")

    # ---------------- weight search ----------------
    grid = np.arange(0, 1.0001, a.step)
    best = (1e18, None)
    for ws in itertools.product(grid, repeat=len(mem)):
        if abs(sum(ws) - 1) > 1e-9:
            continue
        p, _ = blend(mem, ws)
        v = score(p, gt)[0]
        if v < best[0]:
            best = (v, ws)
    ws = best[1]
    print('\nweights', {n: round(float(w), 2) for n, w in zip(names, ws)},
          '-> raw blend OOF %.2f' % best[0])

    oof, te = blend(mem, ws)
    uo, ut = uncertainty(mem, ws)

    # ---------------- honest cross-fitted selection ----------------
    # per-fold best weights are computed once, then reused for every candidate
    fold_ws = []
    for k in range(N_FOLDS):
        ti = np.concatenate([folds[f] for f in range(N_FOLDS) if f != k])
        b2 = (1e18, None)
        for w in np.arange(0, 1.0001, 0.05):
            ws2 = ([w, 1 - w] if len(mem) == 2 else list(ws))
            pv, _ = blend(mem, ws2)
            v = score(pv[ti], gt[ti])[0]
            if v < b2[0]:
                b2 = (v, ws2)
        fold_ws.append(b2[1])

    def honest(s_hi, s_lo):
        p = oof.copy()
        for k in range(N_FOLDS):
            ws2 = fold_ws[k]
            pv, _ = blend(mem, ws2)
            uv, _ = uncertainty(mem, ws2)
            p[folds[k]] = shrink(pv[folds[k]], uv[folds[k]], mx, my, s_hi, s_lo)
        return score(p, gt)[0]

    print('\nshrink candidates, scored honestly (weights refit on 4 folds):')
    cand = []
    for s_hi in (1.0, 1.02, 1.05):
        for s_lo in (0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.00):
            v = honest(s_hi, s_lo)
            ins = score(shrink(oof, uo, mx, my, s_hi, s_lo), gt)[0]
            cand.append((v, s_hi, s_lo))
            print('  s_hi=%.2f s_lo=%.2f -> honest %.2f   (in-sample %.2f)'
                  % (s_hi, s_lo, v, ins))
    cand.sort()
    print('best by honest estimate:', cand[0])

    # ---------------- final ----------------
    s_hi, s_lo = cand[0][1], cand[0][2]
    if a.damp < 1.0:
        s_hi = 1.0 + a.damp * (s_hi - 1.0)
        s_lo = 1.0 + a.damp * (s_lo - 1.0)
        print(f'\ndamped shrink to s_hi={s_hi:.3f} s_lo={s_lo:.3f} (damp={a.damp})')
    oof_f = shrink(oof, uo, mx, my, s_hi, s_lo)
    te_f = shrink(te, ut, mx, my, s_hi, s_lo)

    s = score(oof_f, gt)
    print(f'FINAL OOF {s[0]:.2f} (x {s[1]:.2f} y {s[2]:.2f})')
    d = np.linalg.norm(oof_f - gt, axis=1)
    print('  median %.1f p75 %.1f p90 %.1f max %.1f'
          % (np.median(d), np.percentile(d, 75), np.percentile(d, 90), d.max()))
    print('  worst:', [(os.path.basename(items[i].path), round(float(d[i]), 1))
                       for i in np.argsort(-d)[:6]])

    num2pred = {}
    for g, p in zip(te_items, te_f):
        num2pred[int(os.path.basename(g.path).split('.')[0])] = (float(p[0]), float(p[1]))
    with open(SAMPLE) as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    outp = os.path.join(OUTDIR, a.out)
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r[0]
            num_s, axis = iid.split('_Fovea_')
            fx, fy = num2pred[int(num_s)]
            w.writerow([iid, '%.6f' % (fx if axis == 'X' else fy)])
    print('\nwrote', outp, len(rows), 'rows')
    print('test x %.0f..%.0f  y %.0f..%.0f'
          % (te_f[:, 0].min(), te_f[:, 0].max(), te_f[:, 1].min(), te_f[:, 1].max()))


if __name__ == '__main__':
    main()

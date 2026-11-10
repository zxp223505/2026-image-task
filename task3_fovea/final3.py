# -*- coding: utf-8 -*-
"""final3.py -- assemble the submission from mem2_<tag>.npz members.

Each member provides two extraction variants per axis:
    g = peak-gated soft-argmax   (robust to a secondary peak far away)
    f = plain soft-argmax
`--var tag=xy` chooses per member, e.g. hm2=gf means gated for x, flat for y.

Then:
  * blend weights        (cross-fitted: chosen on 4 folds, scored on the 5th)
  * confidence shrink    (idem; heatmap entropy drives how much to pull to the mean)
  * final refit on all 80 OOF points, applied to the test predictions
"""
import os, csv, argparse, itertools
import numpy as np

from common import WORK, ROOT, test_items
from hm1 import split_folds, N_FOLDS
from final2 import score, unorm, shrink

SAMPLE = os.path.join(ROOT, 'sample_submission.csv')
OUTDIR = os.path.join(os.path.dirname(WORK), 'out')
os.makedirs(OUTDIR, exist_ok=True)


def load_members(specs):
    out = []
    for s in specs:
        tag, var = s.split('=')
        d = np.load(os.path.join(WORK, f'mem2_{tag}.npz'))
        oof = np.column_stack([d['oof_g' if var[0] == 'g' else 'oof_f'][:, 0],
                               d['oof_g' if var[1] == 'g' else 'oof_f'][:, 1]])
        te = np.column_stack([d['te_g' if var[0] == 'g' else 'te_f'][:, 0],
                              d['te_g' if var[1] == 'g' else 'te_f'][:, 1]])
        # confidence: reuse the gated member's stored entropy if present
        p = os.path.join(WORK, f'mem_{tag}.npz')
        if os.path.exists(p):
            e = np.load(p)
            oc, tc = e['oof_conf'], e['te_conf']
        else:
            oc = np.zeros(len(oof)); tc = np.zeros(len(te))
        out.append(dict(tag=tag, var=var, oof=oof, te=te, ooc=oc, tec=tc))
    return out


def blend(mem, ws):
    return (sum(w * m['oof'] for w, m in zip(ws, mem)),
            sum(w * m['te'] for w, m in zip(ws, mem)))


def uncertainty(mem, ws):
    us = []
    for m in mem:
        lo, hi = m['ooc'].min(), m['ooc'].max()
        us.append((unorm(m['ooc'], lo, hi), unorm(m['tec'], lo, hi)))
    return (sum(w * u[0] for w, u in zip(ws, us)),
            sum(w * u[1] for w, u in zip(ws, us)))


def best_w(mem, gt, idx, step=0.05):
    grid = np.arange(0, 1.0001, step)
    best = (1e18, None)
    for ws in itertools.product(grid, repeat=len(mem)):
        if abs(sum(ws) - 1) > 1e-9:
            continue
        p = sum(w * m['oof'][idx] for w, m in zip(ws, mem))
        v = score(p, gt[idx])[0]
        if v < best[0]:
            best = (v, ws)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--var', required=True,
                    help='comma list tag=xy, x/y in {g,f}, e.g. hm1=gg,hm2=gf')
    ap.add_argument('--out', default='submission.csv')
    a = ap.parse_args()

    mem = load_members(a.var.split(','))
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    te_items = test_items()
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()

    for m in mem:
        s = score(m['oof'], gt)
        print(f"  {m['tag']:6s} {m['var']}  OOF {s[0]:7.2f} (x {s[1]:6.2f} y {s[2]:6.2f})")

    ws = best_w(mem, gt, np.arange(len(gt)))
    oof, te = blend(mem, ws)
    uo, ut = uncertainty(mem, ws)
    print('\nweights', {m['tag']: round(float(w), 2) for m, w in zip(mem, ws)},
          '-> raw blend %.2f' % score(oof, gt)[0])

    # honest: weights refit on 4 folds; shrink chosen by the honest score
    fold_ws = []
    for k in range(N_FOLDS):
        ti = np.concatenate([folds[f] for f in range(N_FOLDS) if f != k])
        fold_ws.append(best_w(mem, gt, ti))
    fold_blend = []
    for k in range(N_FOLDS):
        pv, _ = blend(mem, fold_ws[k])
        uv, _ = uncertainty(mem, fold_ws[k])
        fold_blend.append((pv, uv))

    def honest(s_hi, s_lo):
        p = oof.copy()
        for k in range(N_FOLDS):
            pv, uv = fold_blend[k]
            p[folds[k]] = shrink(pv[folds[k]], uv[folds[k]], mx, my, s_hi, s_lo)
        return score(p, gt)[0]

    print('\nshrink selected on the honest estimate:')
    cand = []
    for s_hi in (1.0, 1.02):
        for s_lo in (0.85, 0.88, 0.90, 0.92, 0.95, 1.00):
            v = honest(s_hi, s_lo)
            cand.append((v, s_hi, s_lo))
            print('  s_hi=%.2f s_lo=%.2f -> honest %.2f  (in-sample %.2f)'
                  % (s_hi, s_lo, v, score(shrink(oof, uo, mx, my, s_hi, s_lo), gt)[0]))
    cand.sort()
    best = cand[0]
    print('chosen:', best)

    oof_f = shrink(oof, uo, mx, my, best[1], best[2])
    te_f = shrink(te, ut, mx, my, best[1], best[2])
    s = score(oof_f, gt)
    print(f'\nHONEST OOF ESTIMATE {best[0]:.2f}   (in-sample {s[0]:.2f}, x {s[1]:.2f} y {s[2]:.2f})')
    d = np.linalg.norm(oof_f - gt, axis=1)
    print('  median %.1f p75 %.1f p90 %.1f max %.1f'
          % (np.median(d), np.percentile(d, 75), np.percentile(d, 90), d.max()))
    print('  worst:', [(os.path.basename(items[i].path), round(float(d[i]), 1))
                       for i in np.argsort(-d)[:6]])

    num2pred = {int(os.path.basename(g.path).split('.')[0]): (float(p[0]), float(p[1]))
                for g, p in zip(te_items, te_f)}
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
    np.savez(os.path.join(WORK, 'final_oof.npz'), oof=oof_f, gt=gt, te=te_f,
             conf=uo, honest=np.array([best[0]]),
             s_hi=np.array([best[1]]), s_lo=np.array([best[2]]))
    print('saved final_oof.npz (per-image OOF errors for uncertainty analysis)')


if __name__ == '__main__':
    main()

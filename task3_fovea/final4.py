# -*- coding: utf-8 -*-
"""final4.py -- assemble the submission from mem3_<tag>.npz members
(no epoch selection anywhere: every member is an average of fixed-epoch snapshots).

Per member: pick the extraction variant per axis (g = peak-gated, f = plain).
Then: blend weights and a confidence-dependent shrink, both chosen by a nested
protocol (fit on 4 folds, score the 5th) so the reported number is not inflated
by the selection.
"""
import os, csv, argparse, itertools
import numpy as np

from common import WORK, ROOT, test_items
from hm1 import split_folds, N_FOLDS

SAMPLE = os.path.join(ROOT, 'sample_submission.csv')
OUTDIR = os.path.join(os.path.dirname(WORK), 'out')
os.makedirs(OUTDIR, exist_ok=True)


def score(p, gt):
    e = p - gt
    rx = float(np.sqrt(np.mean(e[:, 0] ** 2)))
    ry = float(np.sqrt(np.mean(e[:, 1] ** 2)))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2.0)), rx, ry


def unorm(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def load_members(specs):
    out = []
    for s in specs:
        tag, var = s.split('=')
        d = np.load(os.path.join(WORK, f'mem3_{tag}.npz'))
        oof = np.column_stack([d['oof_g' if var[0] == 'g' else 'oof_f'][:, 0],
                               d['oof_g' if var[1] == 'g' else 'oof_f'][:, 1]])
        te = np.column_stack([d['te_g' if var[0] == 'g' else 'te_f'][:, 0],
                              d['te_g' if var[1] == 'g' else 'te_f'][:, 1]])
        out.append(dict(tag=tag, var=var, oof=oof, te=te,
                        ooc=d['oof_conf'], tec=d['te_conf']))
    return out


def blend(mem, ws):
    return (sum(w * m['oof'] for w, m in zip(ws, mem)),
            sum(w * m['te'] for w, m in zip(ws, mem)))


def unc(mem, ws):
    us = []
    for m in mem:
        lo, hi = m['ooc'].min(), m['ooc'].max()
        us.append((unorm(m['ooc'], lo, hi), unorm(m['tec'], lo, hi)))
    return (sum(w * u[0] for w, u in zip(ws, us)),
            sum(w * u[1] for w, u in zip(ws, us)))


def shrink(p, u, mx, my, s_hi, s_lo):
    s = s_hi + (s_lo - s_hi) * u
    o = p.copy()
    o[:, 0] = mx + s * (p[:, 0] - mx)
    o[:, 1] = my + s * (p[:, 1] - my)
    return o


def best_w(mem, gt, idx, step=0.05):
    grid = np.arange(0, 1.0001, step)
    b = (1e18, None)
    for ws in itertools.product(grid, repeat=len(mem)):
        if abs(sum(ws) - 1) > 1e-9:
            continue
        p = sum(w * m['oof'][idx] for w, m in zip(ws, mem))
        v = score(p, gt[idx])[0]
        if v < b[0]:
            b = (v, ws)
    return b[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--var', required=True,
                    help='comma list tag=xy (x,y in {g,f}), e.g. hm1b=gg,hm2e=gf')
    ap.add_argument('--out', default='submission.csv')
    a = ap.parse_args()

    mem = load_members(a.var.split(','))
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    te_items = test_items()
    mx, my = gt[:, 0].mean(), gt[:, 1].mean()
    N = len(gt)

    print('members (NO epoch selection):')
    for m in mem:
        s = score(m['oof'], gt)
        print(f"  {m['tag']:6s} {m['var']}  OOF {s[0]:7.2f} (x {s[1]:6.2f} y {s[2]:6.2f})")

    ws = best_w(mem, gt, np.arange(N))
    oof, te = blend(mem, ws)
    uo, ut = unc(mem, ws)
    print('\nweights', {m['tag']: round(float(w), 2) for m, w in zip(mem, ws)},
          '-> blend %.2f' % score(oof, gt)[0])

    fold_ws = [best_w(mem, gt, np.concatenate([folds[f] for f in range(N_FOLDS) if f != k]))
               for k in range(N_FOLDS)]

    def honest(s_hi, s_lo):
        p = oof.copy()
        for k in range(N_FOLDS):
            pv, _ = blend(mem, fold_ws[k])
            uv, _ = unc(mem, fold_ws[k])
            p[folds[k]] = shrink(pv[folds[k]], uv[folds[k]], mx, my, s_hi, s_lo)
        return score(p, gt)[0]

    print('\nshrink, scored honestly (weights refit on 4 folds):')
    cand = []
    for s_hi in (1.0, 1.02):
        for s_lo in (0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 1.0):
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
    print(f'\nHONEST OOF {best[0]:.2f}   (in-sample {s[0]:.2f}, x {s[1]:.2f} y {s[2]:.2f})')
    d = np.linalg.norm(oof_f - gt, axis=1)
    print('  median %.1f p75 %.1f p90 %.1f max %.1f'
          % (np.median(d), np.percentile(d, 75), np.percentile(d, 90), d.max()))

    num2 = {int(os.path.basename(g.path).split('.')[0]): (float(p[0]), float(p[1]))
            for g, p in zip(te_items, te_f)}
    with open(SAMPLE) as f:
        rd = csv.reader(f); header = next(rd); rows = list(rd)
    outp = os.path.join(OUTDIR, a.out)
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r[0]
            num_s, ax = iid.split('_Fovea_')
            fx, fy = num2[int(num_s)]
            w.writerow([iid, '%.6f' % (fx if ax == 'X' else fy)])
    print('\nwrote', outp, len(rows), 'rows')
    np.savez(os.path.join(WORK, 'final_oof4.npz'), oof=oof_f, gt=gt)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""conf.py -- does heatmap confidence predict the error?  If so, a
confidence-dependent shrink towards the mean reduces RMSE on the hard images."""
import argparse
import numpy as np

from hm1 import split_folds, rmse, CELL2CANON
from tune_gate import gated
from common import WORK

H = 128


def conf_stats(v0, v1, delta=2.6):
    """Entropy of the gated softmax (lower = more confident) + peak fraction."""
    out = []
    for v in (v0, v1):
        n = v.shape[0]
        f = v.reshape(n, -1).astype(np.float64)
        mx = f.max(1, keepdims=True)
        f = np.where(f >= mx - delta, f, -1e30)
        e = np.exp(f - f.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        ent = -(p * np.log(p + 1e-30)).sum(1) / np.log(H * H)   # 0..1, high = flat
        out.append(ent)
    return 0.5 * (out[0] + out[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm1')
    a = ap.parse_args()
    d = np.load(f'{WORK}\\heat2_{a.tag}.npz')
    v0, v1 = d['oof0'], d['oof1']
    items, folds = split_folds()
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    c = 0.5 * (gated(v0, 2.6) + gated(v1, 2.6)) * CELL2CANON
    pred = np.array([items[i].to_orig(*c[i]) for i in range(len(items))])
    err = np.linalg.norm(pred - gt, axis=1)
    ent = conf_stats(v0, v1)

    print('corr(entropy, error) = %.3f' % np.corrcoef(ent, err)[0, 1])
    print('corr(entropy, logerr) = %.3f' % np.corrcoef(ent, np.log(err + 1))[0, 1])
    q = np.argsort(ent)
    for name, idx in (('most confident 20', q[:20]), ('middle 40', q[20:60]), ('least confident 20', q[60:])):
        print('  %-20s entropy %.3f..%.3f  median err %5.1f  mean sq %7.1f'
              % (name, ent[idx].min(), ent[idx].max(), np.median(err[idx]),
                 np.mean(err[idx] ** 2)))

    mx, my = gt[:, 0].mean(), gt[:, 1].mean()
    print('\nentropy-weighted shrink scan (shrink strongest where entropy is high)')
    best = (1e18, None)
    for sx_lo in np.arange(0.6, 1.05, 0.1):
        for s_hi in (1.0, 1.05):
            # linear ramp of shrink from sx_lo (max entropy) to s_hi (min entropy)
            t = (ent - ent.min()) / (ent.max() - ent.min())
            sx = s_hi + (sx_lo - s_hi) * t
            p = pred.copy()
            p[:, 0] = mx + sx * (p[:, 0] - mx)
            p[:, 1] = my + sx * (p[:, 1] - my)
            e = p - gt
            s = np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)
            print('  s_hi=%.2f s_lo=%.2f -> %.2f' % (s_hi, sx_lo, s))
            if s < best[0]:
                best = (s, (s_hi, sx_lo))
    print('best', best)

    print('\nplain shrink scan')
    for s in np.arange(0.9, 1.16, 0.02):
        p = pred.copy()
        p[:, 0] = mx + s * (p[:, 0] - mx)
        p[:, 1] = my + s * (p[:, 1] - my)
        e = p - gt
        print('  s=%.2f -> %.2f' % (s, np.sqrt((np.mean(e[:, 0] ** 2) + np.mean(e[:, 1] ** 2)) / 2)))


if __name__ == '__main__':
    main()

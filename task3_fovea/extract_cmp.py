# -*- coding: utf-8 -*-
"""extract_cmp.py -- compare coordinate-extraction rules on stored heatmaps.

global soft-argmax is pulled by secondary peaks (the observed "towards the disc"
bias).  Local / peak-gated extraction should be more robust.
"""
import os, argparse
import numpy as np

from common import WORK
from hm1 import split_folds, rmse, CELL2CANON, OUTSZ

H = OUTSZ


def _softmax(x, axis_last=True):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def ext_global(h):
    """h: (N,H,W) logits -> (N,2) coords in cells."""
    n = h.shape[0]
    p = _softmax(h.reshape(n, -1))
    idx = np.arange(H * H)
    xs = (idx % H + 0.5)
    ys = (idx // H + 0.5)
    return np.stack([(p * xs).sum(1), (p * ys).sum(1)], 1)


def ext_local(h, r=4):
    n = h.shape[0]
    flat = h.reshape(n, -1)
    am = flat.argmax(1)
    ax, ay = am % H, am // H
    out = np.zeros((n, 2))
    for i in range(n):
        x0, x1 = max(0, ax[i] - r), min(H, ax[i] + r + 1)
        y0, y1 = max(0, ay[i] - r), min(H, ay[i] + r + 1)
        win = h[i, y0:y1, x0:x1]
        p = _softmax(win.reshape(-1))
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        px = (p.reshape(y1 - y0, x1 - x0) * xs[None]).sum()
        py = (p.reshape(y1 - y0, x1 - x0) * ys[:, None]).sum()
        out[i] = (px, py)
    return out


def ext_argmax(h):
    n = h.shape[0]
    am = h.reshape(n, -1).argmax(1)
    return np.stack([am % H + 0.5, am // H + 0.5], 1)


def ext_gate(h, delta=4.0, r=None):
    """softmax over the whole map, but only cells within `delta` of the max."""
    n = h.shape[0]
    flat = h.reshape(n, -1)
    mx = flat.max(1, keepdims=True)
    g = np.where(flat >= mx - delta, flat, -1e9)
    p = _softmax(g)
    idx = np.arange(H * H)
    xs = (idx % H + 0.5)
    ys = (idx // H + 0.5)
    return np.stack([(p * xs).sum(1), (p * ys).sum(1)], 1)


def ext_parab(h):
    """argmax + quadratic sub-pixel refinement along each axis."""
    n = h.shape[0]
    am = h.reshape(n, -1).argmax(1)
    ax, ay = (am % H).astype(float), (am // H).astype(float)
    px, py = ax + 0.5, ay + 0.5
    for i in range(n):
        x, y = int(ax[i]), int(ay[i])
        if 0 < x < H - 1:
            a, b, c = h[i, y, x - 1], h[i, y, x], h[i, y, x + 1]
            d = a - 2 * b + c
            if d < 0:
                px[i] = x + 0.5 + np.clip(0.5 * (a - c) / d, -0.5, 0.5)
        if 0 < y < H - 1:
            a, b, c = h[i, y - 1, x], h[i, y, x], h[i, y + 1, x]
            d = a - 2 * b + c
            if d < 0:
                py[i] = y + 0.5 + np.clip(0.5 * (a - c) / d, -0.5, 0.5)
    return np.stack([px, py], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm1')
    a = ap.parse_args()
    d = np.load(os.path.join(WORK, f'heat_{a.tag}.npz'))
    oof = d['oof']
    items, folds = split_folds()

    cands = {'global': ext_global(oof), 'argmax': ext_argmax(oof),
             'parab': ext_parab(oof)}
    for r in (2, 3, 4, 6, 8):
        cands[f'local{r}'] = ext_local(oof, r)
    for dl in (1.5, 2, 3, 4, 6, 9):
        cands[f'gate{dl}'] = ext_gate(oof, dl)

    print('%-10s %8s %8s %8s' % ('rule', 'RMSE', 'x', 'y'))
    res = []
    for k, v in cands.items():
        s = rmse(v * CELL2CANON, items)
        res.append((s[0], k))
        print('%-10s %8.2f %8.2f %8.2f' % (k, s[0], s[1], s[2]))
    res.sort()
    print('\nbest:', res[0][1], '%.2f' % res[0][0])

    # per-image comparison for the best rule
    best = cands[res[0][1]]
    gl = cands['global']
    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    e1 = np.array([items[i].to_orig(*gl[i]) for i in range(len(items))]) - gt
    e2 = np.array([items[i].to_orig(*best[i]) for i in range(len(items))]) - gt
    d1 = np.linalg.norm(e1, axis=1)
    d2 = np.linalg.norm(e2, axis=1)
    print('\nimproved images (global -> best):')
    for i in np.argsort(-(d1 - d2))[:10]:
        print('  %-9s %7.1f -> %7.1f' % (os.path.basename(items[i].path), d1[i], d2[i]))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""make_candidates.py -- combine several submission CSVs by per-image median/mean.

Rationale for the median: our failure mode is "one pipeline is 50+ px off on a
hard image", and a single such image dominates the 20-image pooled RMSE.  A
per-image median over pipelines that are *architecturally* different
(resnet18 heatmap at 512, all-data snapshot family, resnet34) outvotes any one
pipeline's bad guess instead of averaging it in.
"""
import os, sys, csv
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
SAMPLE = r'C:\Users\zhaopangpang\Desktop\task3\detection\sample_submission.csv'


def load(p):
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            n, ax = r['ImageID'].split('_Fovea_')
            d.setdefault(int(n), {})[ax] = float(r['value'])
    return {k: np.array([v['X'], v['Y']]) for k, v in d.items()}


def write(avg, name):
    with open(SAMPLE) as f:
        rd = csv.reader(f); header = next(rd); rows = list(rd)
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r[0]
            n, ax = iid.split('_Fovea_')
            fx, fy = avg[int(n)]
            w.writerow([iid, '%.6f' % (fx if ax == 'X' else fy)])
    print('wrote', path)


def main():
    names = sys.argv[1:]
    sets = [load(os.path.join(OUT, n)) for n in names]
    keys = sorted(sets[0])
    stack = np.stack([[s[k] for k in keys] for s in sets])   # (M, N, 2)

    med = {k: np.median(stack[:, i], axis=0) for i, k in enumerate(keys)}
    mean = {k: stack[:, i].mean(0) for i, k in enumerate(keys)}

    print('per-image spread across the %d inputs (max distance between them):' % len(sets))
    rows = []
    for i, k in enumerate(keys):
        pts = stack[:, i]
        m = max(np.linalg.norm(pts[a] - pts[b]) for a in range(len(sets))
                for b in range(a + 1, len(sets)))
        rows.append((m, k))
    for m, k in sorted(rows, reverse=True)[:8]:
        print('  img %-4d %6.1f px' % (k, m))
    print('  median %.1f px' % np.median([m for m, _ in rows]))

    tag = '_'.join(n.replace('submission_', '').replace('.csv', '') for n in names)
    write(med, f'cand_median_{tag}.csv')
    write(mean, f'cand_mean_{tag}.csv')
    d = [np.linalg.norm(med[k] - mean[k]) for k in keys]
    print('median vs mean differ by: median %.2f px, max %.2f px'
          % (np.median(d), max(d)))


if __name__ == '__main__':
    main()

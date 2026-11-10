# -*- coding: utf-8 -*-
"""blend_csv.py -- average two or more submission CSVs.

Motivation: two independent pipelines both scored ~16.8 public, but they disagree
by up to 59 px on the three hard test images (81/86/96).  Averaging puts those
images at a middle value instead of betting the whole pooled RMSE on one
pipeline's guess.
"""
import os, sys, csv
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
SAMPLE = r'C:\Users\zhaopangpang\Desktop\task3\detection\sample_submission.csv'


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            n, ax = r['ImageID'].split('_Fovea_')
            d.setdefault(int(n), {})[ax] = float(r['value'])
    return {k: np.array([v['X'], v['Y']]) for k, v in d.items()}


def main():
    names = sys.argv[1:-1]
    outp = sys.argv[-1]
    sets = [load(n) for n in names]
    keys = sorted(sets[0])
    avg = {}
    for k in keys:
        avg[k] = np.mean([s[k] for s in sets], axis=0)

    spread = [np.max([np.linalg.norm(s[k] - avg[k]) for s in sets]) for k in keys]
    print('per-image max deviation from the average:')
    for k, sp in sorted(zip(keys, spread), key=lambda t: -t[1])[:6]:
        print('  img %-4d %.1f px   %s' % (k, sp, '  <-- pipelines disagree' if sp > 15 else ''))
    print('median %.1f px' % np.median(spread))

    with open(SAMPLE) as f:
        rd = csv.reader(f); header = next(rd); rows = list(rd)
    path = os.path.join(OUT, outp)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r[0]
            n, ax = iid.split('_Fovea_')
            fx, fy = avg[int(n)]
            w.writerow([iid, '%.6f' % (fx if ax == 'X' else fy)])
    print('wrote', path)


if __name__ == '__main__':
    main()

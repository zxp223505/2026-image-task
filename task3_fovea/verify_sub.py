# -*- coding: utf-8 -*-
"""verify_sub.py -- strict format + sanity check of a submission CSV."""
import csv, sys, os
import numpy as np

SAMPLE = r'C:\Users\zhaopangpang\Desktop\task3\detection\sample_submission.csv'


def check(path):
    ok = True
    with open(SAMPLE) as f:
        ref = list(csv.reader(f))
    with open(path) as f:
        got = list(csv.reader(f))

    print('file        :', path)
    print('header ref  :', ref[0])
    print('header got  :', got[0])
    if ref[0] != got[0]:
        print('  !! header mismatch'); ok = False
    if len(got) != len(ref):
        print(f'  !! row count {len(got)} != {len(ref)}'); ok = False
    if [r[0] for r in ref] != [r[0] for r in got]:
        print('  !! ImageID order/content mismatch')
        for a, b in zip(ref, got):
            if a[0] != b[0]:
                print('     first diff:', a[0], b[0]); break
        ok = False
    if len(set(r[0] for r in got[1:])) != len(got) - 1:
        print('  !! duplicate ImageIDs'); ok = False

    xs, ys = [], []
    for r in got[1:]:
        if len(r) != 2:
            print('  !! bad row', r); ok = False; continue
        try:
            v = float(r[1])
        except ValueError:
            print('  !! non-numeric value', r); ok = False; continue
        if not np.isfinite(v):
            print('  !! non-finite', r); ok = False
        if r[0].endswith('_Fovea_X'):
            xs.append(v)
        elif r[0].endswith('_Fovea_Y'):
            ys.append(v)
        else:
            print('  !! unexpected id', r[0]); ok = False
    xs, ys = np.array(xs), np.array(ys)
    print(f'X: n=%d  min {xs.min():.1f}  max {xs.max():.1f}  mean {xs.mean():.1f}' % len(xs))
    print(f'Y: n=%d  min {ys.min():.1f}  max {ys.max():.1f}  mean {ys.mean():.1f}' % len(ys))
    # sane ranges learnt from the training annotations
    if xs.min() < 200 or xs.max() > 2900:
        print('  !! X outside plausible fundus range'); ok = False
    if ys.min() < 200 or ys.max() > 1900:
        print('  !! Y outside plausible fundus range'); ok = False
    print('RESULT:', 'OK' if ok else 'PROBLEMS FOUND')
    return ok


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out',
        'submission.csv')
    check(p)

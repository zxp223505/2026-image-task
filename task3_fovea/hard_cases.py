# -*- coding: utf-8 -*-
"""hard_cases.py -- are the worst OOF images solvable at all?
Draws GT (green) vs each member's prediction at high magnification."""
import os
import numpy as np
import cv2
from hm1 import split_folds
from common import WORK

OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out'
MEMBERS = [('mem2_hm2.npz', 'oof_g', (255, 255, 0)),
           ('mem2_hm2b.npz', 'oof_g', (255, 0, 255)),
           ('mem2_hm1b.npz', 'oof_f', (0, 165, 255))]


def main(nums=(49, 72, 34, 16)):
    items, folds = split_folds()
    idx = {int(os.path.basename(g.path).split('.')[0]): i for i, g in enumerate(items)}
    data = []
    for fn, key, col in MEMBERS:
        d = np.load(os.path.join(WORK, fn))
        data.append((d[key], col))

    tiles = []
    for n in nums:
        i = idx[n]
        g = items[i]
        canon = np.load(os.path.join(WORK, 'canon', 'train_%04d.jpg.npy' % n))
        gx, gy = g.canon_gt()
        half = 560
        x0 = int(np.clip(gx - half, 0, 2000 - 2 * half))
        y0 = int(np.clip(gy - half, 0, 2000 - 2 * half))
        c = canon[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        c = cv2.resize(c, (340, 340))
        k = 340 / (2 * half)
        cv2.drawMarker(c, (int((gx - x0) * k), int((gy - y0) * k)),
                       (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        for arr, col in data:
            cx, cy = g.to_canon(arr[i, 0], arr[i, 1])   # arr is already original px
            cv2.drawMarker(c, (int((cx - x0) * k), int((cy - y0) * k)),
                           col, cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(c, '%04d green=GT' % n, (4, 16), 0, 0.45, (255, 255, 255), 1)
        tiles.append(c)
    cv2.imwrite(os.path.join(OUT, 'hard_cases.png'), np.hstack(tiles))
    print('saved hard_cases.png; green=GT, cyan=hm2, magenta=hm2b, orange=hm1b')


if __name__ == '__main__':
    main()

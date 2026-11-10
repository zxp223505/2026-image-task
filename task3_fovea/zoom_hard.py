# -*- coding: utf-8 -*-
"""zoom_hard.py -- high-magnification look at the two catastrophic OOF images,
showing the annotation (green) and the model prediction (magenta) side by side
so we can judge which one is really on the fovea.
"""
import os
import numpy as np
import cv2

from common import WORK
from hm1 import split_folds

OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out'


def panel(canon, gt_c, pr_c, half=260, size=420, title=''):
    tiles = []
    for cx, cy, col, lab in ((gt_c[0], gt_c[1], (0, 255, 0), 'GT'),
                             (pr_c[0], pr_c[1], (255, 0, 255), 'PRED')):
        x0 = int(np.clip(cx - half, 0, 2000 - 2 * half))
        y0 = int(np.clip(cy - half, 0, 2000 - 2 * half))
        c = canon[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        c = cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)
        k = size / (2 * half)
        cv2.drawMarker(c, (int((cx - x0) * k), int((cy - y0) * k)), col,
                       cv2.MARKER_CROSS, 46, 2)
        cv2.putText(c, lab, (6, 22), 0, 0.7, col, 2)
        tiles.append(c)
    # wide context with both marks
    wide = cv2.resize(canon, (size, size), interpolation=cv2.INTER_AREA)
    kk = size / 2000
    cv2.drawMarker(wide, (int(gt_c[0] * kk), int(gt_c[1] * kk)), (0, 255, 0),
                   cv2.MARKER_CROSS, 26, 2)
    cv2.drawMarker(wide, (int(pr_c[0] * kk), int(pr_c[1] * kk)), (255, 0, 255),
                   cv2.MARKER_TILTED_CROSS, 26, 2)
    cv2.putText(wide, title, (6, 22), 0, 0.7, (0, 255, 255), 2)
    tiles.append(wide)
    return np.hstack(tiles)


def main():
    items, folds = split_folds()
    idx = {int(os.path.basename(g.path).split('.')[0]): i for i, g in enumerate(items)}
    d = np.load(os.path.join(WORK, 'final_oof4.npz'))
    pred = d['oof']
    rows = []
    for n in (72, 49):
        i = idx[n]
        g = items[i]
        canon = np.load(os.path.join(WORK, 'canon', 'train_%04d.jpg.npy' % n))
        gt_c = g.canon_gt()
        pr_c = g.to_canon(pred[i, 0], pred[i, 1])
        rows.append(panel(canon, gt_c, pr_c, title='%04d' % n))
    cv2.imwrite(os.path.join(OUT, 'zoom_hard2.png'), np.vstack(rows))
    print('saved zoom_hard2.png  (left = annotation, middle = prediction, right = context)')


if __name__ == '__main__':
    main()

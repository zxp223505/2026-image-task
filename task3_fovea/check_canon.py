# -*- coding: utf-8 -*-
"""Verify canonical frame: disc + GT must land correctly; all 100 detections."""
import os
import numpy as np
import cv2
from common import train_items, test_items, WORK
from build_canon import path_of

OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out'


def outline(im, x, y, color, r=26, t=5):
    cv2.circle(im, (int(x), int(y)), r, color, t, cv2.LINE_AA)


if __name__ == '__main__':
    tr = train_items()
    te = test_items()
    # --- full check over train (GT vs detected disc) ---
    tiles = []
    for g in tr:
        c = np.load(path_of(g.path.split('detection\\')[-1].replace('\\', '/')))
        c = cv2.resize(c, (200, 200), interpolation=cv2.INTER_AREA).copy()
        dx, dy = g.canon_disc(); gx, gy = g.canon_gt()
        k = 200 / 2000
        outline(c, dx * k, dy * k, (255, 255, 0), 4, 1)
        outline(c, gx * k, gy * k, (0, 255, 0), 5, 1)
        tiles.append(c)
    rows = []
    for i in range(0, len(tiles), 10):
        rows.append(np.hstack(tiles[i:i + 10]))
    sheet = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT, 'canon_train.png'), sheet)
    print('train sheet', sheet.shape)

    tiles = []
    for g in te:
        c = np.load(path_of(g.path.split('detection\\')[-1].replace('\\', '/')))
        c = cv2.resize(c, (200, 200), interpolation=cv2.INTER_AREA).copy()
        dx, dy = g.canon_disc()
        k = 200 / 2000
        outline(c, dx * k, dy * k, (255, 255, 0), 4, 1)
        tiles.append(c)
    sheet = np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)])
    cv2.imwrite(os.path.join(OUT, 'canon_test.png'), sheet)
    print('test sheet', sheet.shape)

    # --- zoom on 6 canonical images at full res ---
    zooms = []
    for g in tr[:6]:
        c = np.load(path_of(g.path.split('detection\\')[-1].replace('\\', '/')))
        c = cv2.resize(c, (500, 500), interpolation=cv2.INTER_AREA).copy()
        dx, dy = g.canon_disc(); gx, gy = g.canon_gt()
        k = 500 / 2000
        outline(c, dx * k, dy * k, (255, 255, 0), 22, 3)
        outline(c, gx * k, gy * k, (0, 255, 0), 22, 3)
        zooms.append(c)
    cv2.imwrite(os.path.join(OUT, 'canon_zoom.png'), np.vstack([np.hstack(zooms[:3]),
                                                               np.hstack(zooms[3:6])]))
    print('zoom ok')

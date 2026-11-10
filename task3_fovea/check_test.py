# -*- coding: utf-8 -*-
"""check_test.py -- draw predicted foveae on the 20 test images for eyeballing."""
import os, csv
import numpy as np
import cv2
from common import ROOT, TEST, WORK

OUT = os.path.join(os.path.dirname(WORK), 'out')


def main(csv_path):
    pred = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            n, ax = r['ImageID'].split('_Fovea_')
            pred.setdefault(int(n), {})[ax] = float(r['value'])
    names = sorted(nm for nm in os.listdir(TEST) if nm.lower().endswith('.jpg'))
    tiles = []
    for nm in names:
        n = int(nm.split('.')[0])
        im = cv2.imread(os.path.join(TEST, nm))
        H, W = im.shape[:2]
        x, y = pred[n]['X'], pred[n]['Y']
        # crop around the prediction at native res so misplacement is obvious
        half = 420
        x0 = int(np.clip(x - half, 0, W - 2 * half))
        y0 = int(np.clip(y - half, 0, H - 2 * half))
        c = im[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        c = cv2.resize(c, (300, 300))
        k = 300 / (2 * half)
        cv2.drawMarker(c, (int((x - x0) * k), int((y - y0) * k)),
                       (0, 255, 0), cv2.MARKER_CROSS, 34, 2)
        cv2.putText(c, nm, (5, 18), 0, 0.55, (0, 255, 255), 1)
        tiles.append(c)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    sheet = np.vstack(rows)
    p = os.path.join(OUT, 'test_predictions.png')
    cv2.imwrite(p, sheet)
    print('saved', p, sheet.shape)


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'submission_v5.csv'))

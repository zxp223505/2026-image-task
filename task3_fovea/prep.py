# -*- coding: utf-8 -*-
"""prep.py -- cache per-image geometry for all train+test fundus images.

For each image we estimate:
  * the fundus field-of-view circle (center, radius) -> gives canonical scale
  * the optic-disc center (unsupervised: brightest compact blob inside the FOV)

Cached to geom.json so every training script sees identical numbers.
"""
import os, json, csv
import numpy as np
import cv2
from scipy import ndimage

ROOT = r'C:\Users\zhaopangpang\Desktop\task3\detection'
OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\work'
TRAIN, TEST = os.path.join(ROOT, 'train'), os.path.join(ROOT, 'test')


def load_gt():
    gt = {}
    with open(os.path.join(ROOT, 'fovea_localization_train_GT.csv')) as f:
        for r in csv.DictReader(f):
            gt[int(float(r['data']))] = (float(r['Fovea_X']), float(r['Fovea_Y']))
    return gt


def fov_circle(gray, thr=12):
    """gray in [0,1]; thr given in 0..255 units."""
    mask = gray > (thr / 255.0)
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    R = np.sqrt(mask.sum() / np.pi)
    return float(cx), float(cy), float(R)


def _blob_peak(lum, cx, cy, R, sigma_frac, r_frac=0.88):
    """Gaussian-smoothed argmax of luminance inside the FOV."""
    h, w = lum.shape
    sm = ndimage.gaussian_filter(lum, sigma=max(1.5, sigma_frac * R))
    yy, xx = np.mgrid[0:h, 0:w]
    inner = ((yy - cy) ** 2 + (xx - cx) ** 2) < (r_frac * R) ** 2
    sm = np.where(inner, sm, -1e9)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    return float(ix), float(iy)


def detect_disc(img):
    """Two-pass unsupervised optic-disc centre (original pixel coords)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cx, cy, R = fov_circle(gray)

    # BGR image; disc is bright & yellow-white
    B, G, Rc = (img[..., 0].astype(np.float32) / 255.0,
                img[..., 1].astype(np.float32) / 255.0,
                img[..., 2].astype(np.float32) / 255.0)
    lum = 0.15 * B + 0.40 * G + 0.45 * Rc

    # pass 1: global, coarse
    s1 = 0.20
    small = cv2.resize(lum, (max(16, int(W * s1)), max(16, int(H * s1))), interpolation=cv2.INTER_AREA)
    x1, y1 = _blob_peak(small, cx * s1, cy * s1, R * s1, 0.055)
    x1, y1 = x1 / s1, y1 / s1

    # pass 2: local refinement around pass-1 hit
    half = int(0.45 * R)
    x0, y0 = int(max(0, x1 - half)), int(max(0, y1 - half))
    x2_, y2_ = int(min(W, x1 + half)), int(min(H, y1 + half))
    s2 = 0.5
    crop = cv2.resize(lum[y0:y2_, x0:x2_],
                      (max(16, int((x2_ - x0) * s2)), max(16, int((y2_ - y0) * s2))),
                      interpolation=cv2.INTER_AREA)
    lx, ly = _blob_peak(crop, (x1 - x0) * s2, (y1 - y0) * s2, (x2_ - x0) * s2 * 0.5, 0.030, r_frac=1.5)
    dx, dy = x0 + lx / s2, y0 + ly / s2

    # sanity: keep inside FOV, else fall back to pass-1
    if (dx - cx) ** 2 + (dy - cy) ** 2 > (0.95 * R) ** 2:
        dx, dy = x1, y1
    return float(dx), float(dy), cx, cy, R


def build():
    meta = {}
    for split, folder in (('train', TRAIN), ('test', TEST)):
        for nm in sorted(os.listdir(folder)):
            if not nm.lower().endswith('.jpg'):
                continue
            img = cv2.imread(os.path.join(folder, nm))
            H, W = img.shape[:2]
            dx, dy, cx, cy, R = detect_disc(img)
            meta[f'{split}/{nm}'] = dict(w=W, h=H, disc_x=dx, disc_y=dy,
                                         fov_cx=cx, fov_cy=cy, fov_r=R)
    gt = load_gt()
    for n, (fx, fy) in gt.items():
        k = 'train/%04d.jpg' % n
        meta[k]['gt_x'] = fx
        meta[k]['gt_y'] = fy
    with open(os.path.join(OUT, 'geom.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('wrote geom.json with %d entries' % len(meta))
    return meta


if __name__ == '__main__':
    m = build()
    R = np.array([v['fov_r'] for v in m.values()])
    print('fov_r  min %.1f  max %.1f' % (R.min(), R.max()))
    tr = {k: v for k, v in m.items() if k.startswith('train/')}
    dx = np.array([v['disc_x'] for v in tr.values()])
    dy = np.array([v['disc_y'] for v in tr.values()])
    fx = np.array([v['gt_x'] for v in tr.values()])
    fy = np.array([v['gt_y'] for v in tr.values()])
    rr = np.array([v['fov_r'] for v in tr.values()])
    off_x = (fx - dx) / rr
    off_y = (fy - dy) / rr
    print('signed dx/R  mean %+.4f std %.4f' % (off_x.mean(), off_x.std()))
    print('|dx|/R       mean  %.4f std %.4f' % (np.abs(off_x).mean(), np.abs(off_x).std()))
    print('dy/R         mean %+.4f std %.4f' % (off_y.mean(), off_y.std()))
    print('max |dx|/R = %.3f  max |dy|/R = %.3f' % (np.abs(off_x).max(), np.abs(off_y).max()))
    same_side = ((fx - dx) > 0).sum()
    print('disc left of fovea: %d / %d' % (same_side, len(fx)))

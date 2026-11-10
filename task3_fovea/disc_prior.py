# -*- coding: utf-8 -*-
"""Test the anatomical prior: fovea position relative to the OPTIC DISC.

No disc labels exist, so we build a cheap unsupervised disc detector and check
whether (fovea - disc) is more stable than (fovea - fov_center).
"""
import os, csv
import numpy as np
from PIL import Image
from scipy import ndimage  # noqa

D = r'C:\Users\zhaopangpang\Desktop\task3\detection'


def load_gt():
    gt = {}
    with open(os.path.join(D, 'fovea_localization_train_GT.csv')) as f:
        for r in csv.DictReader(f):
            gt[int(float(r['data']))] = (float(r['Fovea_X']), float(r['Fovea_Y']))
    return gt


def detect_disc(path, scale=0.25, sigma_frac=0.06):
    """Bright compact blob = optic disc. Returns (x, y, W, H) in original px."""
    im = Image.open(path).convert('RGB')
    W, H = im.size
    w, h = int(W * scale), int(H * scale)
    a = np.asarray(im.resize((w, h)), dtype=np.float32) / 255.0
    # disc is bright & yellowish-white -> use luminance with strong red+green weight
    lum = 0.5 * a[..., 0] + 0.4 * a[..., 1] + 0.1 * a[..., 2]
    # restrict to fundus circle
    yy, xx = np.mgrid[0:h, 0:w]
    mask = lum > 0.05
    if mask.sum() < 10:
        return W / 2, H / 2, W, H
    cy0, cx0 = np.nonzero(mask)[0].mean(), np.nonzero(mask)[1].mean()
    R = np.sqrt(mask.sum() / np.pi)
    sm = ndimage.gaussian_filter(lum, sigma=max(2.0, sigma_frac * R))
    # keep only pixels well inside the circle (avoid rim / text)
    inner = ((yy - cy0) ** 2 + (xx - cx0) ** 2) < (0.88 * R) ** 2
    sm = np.where(inner, sm, -1)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    return ix / scale, iy / scale, W, H


if __name__ == '__main__':
    gt = load_gt()
    rows = []
    for n in sorted(gt):
        p = os.path.join(D, 'train', '%04d.jpg' % n)
        dx_, dy_, W, H = detect_disc(p)
        fx, fy = gt[n]
        rows.append((n, dx_, dy_, fx, fy, W, H))
    r = np.array([x[1:] for x in rows], float)
    dx_, dy_, fx, fy, W, H = r.T

    # per-geometry scale R
    R = np.where(W > 2500, 971.0, 930.0)
    print('--- raw pixel offsets ---')
    print('fovea-disc:  dX mean %+8.1f std %7.1f | dY mean %+8.1f std %6.1f'
          % ((fx - dx_).mean(), (fx - dx_).std(), (fy - dy_).mean(), (fy - dy_).std()))
    print('fovea-cent:  dX mean %+8.1f std %7.1f | dY mean %+8.1f std %6.1f'
          % ((fx - W / 2).mean(), (fx - W / 2).std(), (fy - H / 2).mean(), (fy - H / 2).std()))
    # disc side
    side = np.sign(fx - dx_)
    print('\ndisc on LEFT (fovea right of disc): %d   disc on RIGHT: %d'
          % ((side > 0).sum(), (side < 0).sum()))
    # normalised by R, per side, mirror the x offset so both eyes share a prior
    mx = np.abs(fx - dx_) / R
    my = (fy - dy_) / R
    print('\n|dX|/R  mean %.4f std %.4f   (=> %.0f px @R=971)' % (mx.mean(), mx.std(), mx.std() * 971))
    print(' dY /R  mean %+.4f std %.4f (=> %.0f px)' % (my.mean(), my.std(), my.std() * 971))
    print('\nfor comparison, fovea abs position /R: dX std %.4f (%.0f px), dY std %.4f (%.0f px)'
          % (((fx - 1500) / R).std(), ((fx - 1500) / R).std() * 971,
             ((fy - 991) / R).std(), ((fy - 991) / R).std() * 971))

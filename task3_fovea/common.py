# -*- coding: utf-8 -*-
"""common.py -- shared geometry / preprocessing for the fovea pipeline.

Canonical frame
---------------
Every image is mapped to a fixed 2*REF_R square whose inscribed circle is the
fundus field of view.  This removes the two different native sizes
(2992x2000 and 1956x1934) and their different fundus radii, so the network
always sees the same anatomy at the same scale.
"""
import os, json, csv
import numpy as np
import cv2

ROOT = r'C:\Users\zhaopangpang\Desktop\task3\detection'
WORK = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\work'
TRAIN, TEST = os.path.join(ROOT, 'train'), os.path.join(ROOT, 'test')

REF_R = 1000.0           # canonical fundus radius (px)
CANON = int(2 * REF_R)   # canonical image side
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_geom():
    with open(os.path.join(WORK, 'geom.json')) as f:
        return json.load(f)


def load_gt():
    gt = {}
    with open(os.path.join(ROOT, 'fovea_localization_train_GT.csv')) as f:
        for r in csv.DictReader(f):
            gt[int(float(r['data']))] = (float(r['Fovea_X']), float(r['Fovea_Y']))
    return gt


class Geo:
    """Canonical-frame mapping for one image."""

    __slots__ = ('path', 'w', 'h', 'disc_x', 'disc_y', 'fov_cx', 'fov_cy',
                 'fov_r', 's', 'gt_x', 'gt_y')

    def __init__(self, path, d):
        self.path = path
        self.w, self.h = d['w'], d['h']
        self.disc_x, self.disc_y = d['disc_x'], d['disc_y']
        self.fov_cx, self.fov_cy, self.fov_r = d['fov_cx'], d['fov_cy'], d['fov_r']
        self.s = REF_R / self.fov_r
        self.gt_x = d.get('gt_x')
        self.gt_y = d.get('gt_y')

    # original px  ->  canonical px
    def to_canon(self, x, y):
        return (x - self.fov_cx) * self.s + REF_R, (y - self.fov_cy) * self.s + REF_R

    # canonical px  ->  original px
    def to_orig(self, x, y):
        return (x - REF_R) / self.s + self.fov_cx, (y - REF_R) / self.s + self.fov_cy

    def canon_disc(self):
        return self.to_canon(self.disc_x, self.disc_y)

    def canon_gt(self):
        if self.gt_x is None:
            return None
        return self.to_canon(self.gt_x, self.gt_y)


def geom_index():
    meta = load_geom()
    out = {}
    for k, d in meta.items():
        split, nm = k.split('/')
        folder = TRAIN if split == 'train' else TEST
        out[k] = Geo(os.path.join(folder, nm), d)
    return out


def train_items():
    geos = geom_index()
    items = []
    for n in sorted(load_gt()):
        items.append(geos['train/%04d.jpg' % n])
    return items


def test_items():
    geos = geom_index()
    return [geos['test/%s' % nm] for nm in sorted(os.listdir(TEST))
            if nm.lower().endswith('.jpg')]


def load_canon(geo, img=None):
    """Return the 2000x2000 canonical uint8 BGR image (black outside the fundus)."""
    if img is None:
        img = cv2.imread(geo.path, cv2.IMREAD_COLOR)
    M = np.array([[geo.s, 0.0, -(geo.fov_cx - REF_R) * 0.0],
                  [0.0, geo.s, 0.0]], np.float64)
    # x' = (x - cx)*s + REF_R  ->  tx = REF_R - cx*s
    M[0, 2] = REF_R - geo.fov_cx * geo.s
    M[1, 2] = REF_R - geo.fov_cy * geo.s
    return cv2.warpAffine(img, M, (CANON, CANON), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), M


def dog_channel(img_bgr, size):
    """Difference-of-Gaussians at the macular scale, as an extra input channel.

    The fovea sits in a ~300 canonical px dark patch.  On the two images where
    our model fails badly (0072, 0049) the overall colour/illumination is so
    washed out that this patch is only barely darker than its surroundings, and
    a plain RGB network locks onto something else.  A band-pass at exactly that
    scale makes the blob explicit, independent of global brightness or colour
    cast.  sigma values are in units of `size` (the network input side).
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s1 = max(1.0, 0.010 * size)      # ~ macula core
    s2 = max(2.0, 0.045 * size)      # ~ local background
    a = cv2.GaussianBlur(g, (0, 0), s1)
    b = cv2.GaussianBlur(g, (0, 0), s2)
    d = a - b                        # negative where locally dark
    sc = np.abs(d).max()
    if sc > 1e-6:
        d = d / sc
    return ((d * 0.5 + 0.5) * 255.0).astype(np.float32)


def to_input(img_bgr, size, dog=False):
    """uint8 BGR (any square crop) -> normalised CHW float32 tensor (numpy)."""
    im = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA
                    if img_bgr.shape[0] > size else cv2.INTER_LINEAR)
    a = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    a = (a - MEAN) / STD
    out = np.ascontiguousarray(a.transpose(2, 0, 1))
    if dog:
        d = dog_channel(im, size)[None] / 255.0
        d = (d - 0.5) / 0.25
        out = np.concatenate([out, d.astype(np.float32)], 0)
    return out


def affine_about(center, angle_deg, scale, tx, ty):
    """3x3 matrix rotating/scaling about `center` then translating by (tx,ty)."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a) * scale, np.sin(a) * scale
    cx, cy = center
    M = np.array([[ca, sa, (1 - ca) * cx - sa * cy + tx],
                  [-sa, ca, sa * cx + (1 - ca) * cy + ty],
                  [0.0, 0.0, 1.0]], np.float64)
    return M


def apply_pt(M, x, y):
    v = M @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def gauss_map(hw, cx, cy, sigma):
    """Gaussian heatmap of shape (hw,hw) peaked at (cx,cy) in cell units."""
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)

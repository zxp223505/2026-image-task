# -*- coding: utf-8 -*-
"""build_canon.py -- materialise every image in the canonical frame as .npy.

2000x2000 uint8, black outside the fundus circle.  Read later with mmap so the
OS page cache does the caching for us (no 1.2 GB of Python objects).
"""
import os, sys
import numpy as np
import cv2
from common import geom_index, load_canon, CANON, WORK

CACHE = os.path.join(WORK, 'canon')


def build():
    os.makedirs(CACHE, exist_ok=True)
    geos = geom_index()
    n = 0
    for k, g in sorted(geos.items()):
        out = os.path.join(CACHE, k.replace('/', '_') + '.npy')
        if os.path.exists(out):
            continue
        img, _ = load_canon(g)
        assert img.shape == (CANON, CANON, 3), img.shape
        np.save(out, img)
        n += 1
        if n % 20 == 0:
            print('  ... %d' % n, flush=True)
    print('built %d, total %d' % (n, len(os.listdir(CACHE))))


def path_of(key):
    return os.path.join(CACHE, key.replace('/', '_') + '.npy')


if __name__ == '__main__':
    build()

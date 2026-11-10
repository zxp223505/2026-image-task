# -*- coding: utf-8 -*-
"""build_canon1k.py -- half-resolution canonical cache for fast stage-1 training.

Stage 1 feeds a 512x512 network input, so a 1000x1000 source loses nothing but
cuts the per-sample memory traffic 4x.
"""
import os
import numpy as np
import cv2
from common import WORK
from build_canon import path_of

SRC = 1000
CACHE = os.path.join(WORK, 'canon1k')


def build():
    os.makedirs(CACHE, exist_ok=True)
    src = os.path.join(WORK, 'canon')
    n = 0
    for fn in sorted(os.listdir(src)):
        out = os.path.join(CACHE, fn)
        if os.path.exists(out):
            continue
        a = np.load(os.path.join(src, fn))
        np.save(out, cv2.resize(a, (SRC, SRC), interpolation=cv2.INTER_AREA))
        n += 1
    print('built %d / %d' % (n, len(os.listdir(CACHE))))


def path_of_1k(key):
    return os.path.join(CACHE, key.replace('/', '_') + '.npy')


if __name__ == '__main__':
    build()

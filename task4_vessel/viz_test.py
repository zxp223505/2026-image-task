"""Qualitative check: test images + predicted vessel overlay montage."""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, load_test_item

pred_dir = os.path.join(WORK, "pred", "image")
rows = []
for i in [1, 3, 7, 12]:
    img, fov = load_test_item(i)
    mask = np.array(Image.open(os.path.join(pred_dir, f"{i}.png")).convert("L")) > 127
    ov = img.copy()
    ov[mask] = (0.35 * img[mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    # zoom on optic disc region (right side usually)
    h, w = img.shape[:2]
    z = []
    for (y0, x0) in [(140, 300), (260, 330)]:
        z.append(cv2.resize(ov[y0:y0 + 140, x0:x0 + 140], (280, 280), interpolation=cv2.INTER_NEAREST))
    row = np.hstack([img, ov, np.dstack([mask.astype(np.uint8) * 255] * 3), np.hstack(z)])
    rows.append(row)
out = np.vstack(rows)
out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
p = os.path.join(WORK, "pred", "montage_test.png")
cv2.imwrite(p, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("wrote", p, out.shape)

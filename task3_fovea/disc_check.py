# -*- coding: utf-8 -*-
"""Visual check: are the unsupervised 'disc' detections on the optic disc?"""
import os, csv
import numpy as np
from PIL import Image, ImageDraw
from disc_prior import detect_disc, load_gt, D

OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out'

if __name__ == '__main__':
    gt = load_gt()
    nums = list(range(1, 25))
    cols, box = 6, (320, 320)
    rows = (len(nums) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * box[0], rows * (box[1] + 18)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, n in enumerate(nums):
        im = Image.open(os.path.join(D, 'train', '%04d.jpg' % n)).convert('RGB')
        dx, dy, W, H = detect_disc(os.path.join(D, 'train', '%04d.jpg' % n))
        fx, fy = gt[n]
        dd = ImageDraw.Draw(im)
        r = W // 90
        dd.ellipse([dx - r, dy - r, dx + r, dy + r], outline=(0, 255, 255), width=6)   # cyan = disc
        dd.ellipse([fx - r, fy - r, fx + r, fy + r], outline=(0, 255, 0), width=6)     # green = fovea GT
        im.thumbnail(box)
        c, rr = i % cols, i // cols
        sheet.paste(im, (c * box[0] + (box[0] - im.size[0]) // 2, rr * (box[1] + 18)))
        d.text((c * box[0] + 5, rr * (box[1] + 18) + box[1] + 2), '%04d' % n, fill=(255, 255, 0))
    sheet.save(os.path.join(OUT, 'disc_check.png'))
    print('saved', sheet.size)

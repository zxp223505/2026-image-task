# -*- coding: utf-8 -*-
"""Visual inspection: montage of train images with GT fovea marked + stats."""
import os, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = r'C:\Users\zhaopangpang\Desktop\task3\detection'
TRAIN = os.path.join(D, 'train')
OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out'


def load_gt():
    gt = {}
    with open(os.path.join(D, 'fovea_localization_train_GT.csv')) as f:
        for r in csv.DictReader(f):
            gt[int(float(r['data']))] = (float(r['Fovea_X']), float(r['Fovea_Y']))
    return gt


def mark(num, gt, box=(300, 300), radius=6, scale=1.0):
    p = os.path.join(TRAIN, '%04d.jpg' % num)
    im = Image.open(p).convert('RGB')
    W, H = im.size
    x, y = gt[num]
    d = ImageDraw.Draw(im)
    r = radius
    d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=max(3, W // 300))
    d.line([x - 3 * r, y, x + 3 * r, y], fill=(0, 255, 0), width=max(2, W // 500))
    d.line([x, y - 3 * r, x, y + 3 * r], fill=(0, 255, 0), width=max(2, W // 500))
    im.thumbnail((box[0], box[1]))
    return im


def montage(nums, path, cols=5, box=(400, 400)):
    gt = load_gt()
    rows = (len(nums) + cols - 1) // cols
    tile_w, tile_h = box
    sheet = Image.new('RGB', (cols * tile_w, rows * (tile_h + 18)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, n in enumerate(nums):
        im = mark(n, gt, box=box)
        c, r = i % cols, i // cols
        sheet.paste(im, (c * tile_w + (tile_w - im.size[0]) // 2, r * (tile_h + 18)))
        d.text((c * tile_w + 5, r * (tile_h + 18) + tile_h + 2), '%04d' % n, fill=(255, 255, 0))
    sheet.save(path)
    print('saved', path, sheet.size)


if __name__ == '__main__':
    gt = load_gt()
    # full-image montage, all 80
    montage(list(range(1, 81)), os.path.join(OUT, 'montage_all.png'), cols=8, box=(240, 240))
    # zoom crops around fovea at native res
    nums = [1, 4, 9, 24, 39, 49, 59, 77]
    tiles = []
    for n in nums:
        im = Image.open(os.path.join(TRAIN, '%04d.jpg' % n)).convert('RGB')
        x, y = gt[n]
        c = im.crop((int(x) - 200, int(y) - 200, int(x) + 200, int(y) + 200))
        d = ImageDraw.Draw(c)
        d.line([196, 200, 204, 200], fill=(0, 255, 0), width=2)
        d.line([200, 196, 200, 204], fill=(0, 255, 0), width=2)
        tiles.append((n, c))
    sheet = Image.new('RGB', (4 * 400, 2 * 418), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, (n, c) in enumerate(tiles):
        cc, rr = i % 4, i // 4
        sheet.paste(c, (cc * 400, rr * 418))
        d.text((cc * 400 + 5, rr * 418 + 402), '%04d' % n, fill=(255, 255, 0))
    sheet.save(os.path.join(OUT, 'zoom_fovea.png'))
    print('saved zoom', sheet.size)

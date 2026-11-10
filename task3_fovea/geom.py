# -*- coding: utf-8 -*-
"""FOV circle geometry analysis + test montage."""
import os, csv
import numpy as np
from PIL import Image, ImageDraw

D = r'C:\Users\zhaopangpang\Desktop\task3\detection'
OUT = r'C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\task3\out'


def fov_geom(path, thr=12):
    """Estimate fundus field-of-view circle from non-black pixels."""
    a = np.asarray(Image.open(path).convert('L').resize((400, 400)) if False else
                   Image.open(path).convert('L'))
    H, W = a.shape
    mask = a > thr
    # drop rows/cols that are mostly text overlays later; use area-based radius
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    area = mask.sum()
    R = np.sqrt(area / np.pi)
    return cx, cy, R, W, H, mask.mean()


def load_gt():
    gt = {}
    with open(os.path.join(D, 'fovea_localization_train_GT.csv')) as f:
        for r in csv.DictReader(f):
            gt[int(float(r['data']))] = (float(r['Fovea_X']), float(r['Fovea_Y']))
    return gt


if __name__ == '__main__':
    gt = load_gt()
    print('%-6s %-11s %8s %8s %8s %8s %8s %8s' % ('num', 'size', 'cx', 'cy', 'R', 'fx', 'fy', 'fovfrac'))
    rows = []
    for n in sorted(gt):
        p = os.path.join(D, 'train', '%04d.jpg' % n)
        cx, cy, R, W, H, frac = fov_geom(p)
        fx, fy = gt[n]
        rows.append((n, W, H, cx, cy, R, fx, fy, frac))
        print('%04d %4dx%4d %8.1f %8.1f %8.1f %8.1f %8.1f %8.3f' % (n, W, H, cx, cy, R, fx, fy, frac))
    r = np.array([[x[3], x[4], x[5], x[6], x[7], x[1], x[2]] for x in rows], float)
    cx, cy, R, fx, fy, W, H = r.T
    print()
    print('--- fovea relative to FOV circle ---')
    print('dx=(fx-cx)/R  mean %.4f std %.4f' % (((fx - cx) / R).mean(), ((fx - cx) / R).std()))
    print('dy=(fy-cy)/R  mean %.4f std %.4f' % (((fy - cy) / R).mean(), ((fy - cy) / R).std()))
    print('--- raw normalized ---')
    print('fx/W std %.4f   fy/H std %.4f' % ((fx / W).std(), (fy / H).std()))
    print('px std X %.1f  px std Y %.1f' % (fx.std(), fy.std()))

    # test montage
    names = sorted(os.listdir(os.path.join(D, 'test')))
    cols, box = 5, (380, 380)
    rows_n = (len(names) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * box[0], rows_n * (box[1] + 18)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, nm in enumerate(names):
        im = Image.open(os.path.join(D, 'test', nm)).convert('RGB')
        im.thumbnail(box)
        c, rr = i % cols, i // cols
        sheet.paste(im, (c * box[0] + (box[0] - im.size[0]) // 2, rr * (box[1] + 18)))
        d.text((c * box[0] + 5, rr * (box[1] + 18) + box[1] + 2), nm, fill=(255, 255, 0))
    sheet.save(os.path.join(OUT, 'montage_test.png'))
    print('\nsaved test montage', sheet.size)

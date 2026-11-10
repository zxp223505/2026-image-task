# -*- coding: utf-8 -*-
"""extract2.py -- dump BOTH TTA views' heatmaps so we can tune the extraction
rule against the exact inference pipeline.

view 0 : heatmap of the image
view 1 : heatmap of the mirrored image, mirrored back into the original frame
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
from hm1 import FoveaDS, FoveaNet, split_folds, N_FOLDS, CKPT
from common import train_items, test_items, WORK


def load_model(tag, k):
    ck = torch.load(os.path.join(CKPT, f'{tag}_fold{k}.pt'), map_location='cuda',
                    weights_only=False)
    m = FoveaNet(ck.get('arch', 'resnet18')).cuda()
    m.load_state_dict(ck['state'])
    m.eval()
    return m


@torch.no_grad()
def two_views(model, geos, split, bs=8):
    ds = FoveaDS(geos, split, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    v0, v1 = [], []
    for x, _, _ in dl:
        h0, _ = model(x.cuda())
        h1, _ = model(torch.flip(x, dims=[3]).cuda())
        v0.append(h0[:, 0].cpu().numpy())
        v1.append(torch.flip(h1[:, 0], dims=[2]).cpu().numpy())
    return np.concatenate(v0).astype(np.float32), np.concatenate(v1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm1')
    ap.add_argument('--mask_disc', type=float, default=0.0)
    a = ap.parse_args()
    hm1.MASK_DISC = a.mask_disc

    items, folds = split_folds()
    te = test_items()
    H = 128
    oof0 = np.zeros((len(items), H, H), np.float32)
    oof1 = np.zeros((len(items), H, H), np.float32)
    for k in range(N_FOLDS):
        m = load_model(a.tag, k)
        va = [items[i] for i in folds[k]]
        a0, a1 = two_views(m, va, 'train')
        oof0[folds[k]] = a0
        oof1[folds[k]] = a1
        print(f'  fold{k} done', flush=True)
        del m
        torch.cuda.empty_cache()

    te0 = np.zeros((len(te), H, H), np.float32)
    te1 = np.zeros((len(te), H, H), np.float32)
    for k in range(N_FOLDS):
        m = load_model(a.tag, k)
        a0, a1 = two_views(m, te, 'test')
        te0 += a0 / N_FOLDS
        te1 += a1 / N_FOLDS
        del m
        torch.cuda.empty_cache()

    out = os.path.join(WORK, f'heat2_{a.tag}.npz')
    np.savez(out, oof0=oof0, oof1=oof1, te0=te0, te1=te1)
    print('wrote', out)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""extract.py -- dump the raw fovea heatmaps for every train image (leak-free,
fold model only) and the test images (5-fold average), so different coordinate
extraction rules can be compared offline.

The global soft-argmax expectation is sensitive to a *secondary* peak elsewhere
in the map (observed failure mode: predictions pulled towards the optic disc).
Storing the maps lets us test local / peak-robust extraction for free.
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
def heatmaps(model, geos, split, tta=True, bs=8):
    ds = FoveaDS(geos, split, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    outs = []
    for x, _, _ in dl:
        h, _ = model(x.cuda())
        h = h[:, 0]
        if tta:
            h2, _ = model(torch.flip(x, dims=[3]).cuda())
            h = 0.5 * (h + torch.flip(h2[:, 0], dims=[2]))
        outs.append(h.cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hm1')
    ap.add_argument('--mask_disc', type=float, default=0.0)
    args = ap.parse_args()
    hm1.MASK_DISC = args.mask_disc

    items, folds = split_folds()
    te = test_items()
    H = 128
    oof = np.zeros((len(items), H, H), np.float32)
    for k in range(N_FOLDS):
        m = load_model(args.tag, k)
        va = [items[i] for i in folds[k]]
        hmaps = heatmaps(m, va, 'train')
        oof[folds[k]] = hmaps
        print(f'  {args.tag} fold{k} heatmaps {hmaps.shape}', flush=True)
        del m
        torch.cuda.empty_cache()

    te_hm = np.zeros((len(te), H, H), np.float32)
    for k in range(N_FOLDS):
        m = load_model(args.tag, k)
        te_hm += heatmaps(m, te, 'test') / N_FOLDS
        del m
        torch.cuda.empty_cache()

    out = os.path.join(WORK, f'heat_{args.tag}.npz')
    np.savez(out, oof=oof, te=te_hm)
    print('wrote', out, oof.shape, te_hm.shape)


if __name__ == '__main__':
    main()

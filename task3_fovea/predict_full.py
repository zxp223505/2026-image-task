# -*- coding: utf-8 -*-
"""predict_full.py -- average every fixed-epoch snapshot of one or more
all-data models and emit train/test coordinates in the canonical frame
(for centring stage 2) and in original pixels (for the submission).
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import hm1
from hm1 import FoveaDS, FoveaNet, CELL2CANON, GATE
from hm2 import PatchDS, CELL2CANON2, PATCH
from common import train_items, test_items, WORK

CKPT = os.path.join(WORK, 'ckpt')
KEYS = ('tr_g', 'tr_f', 'tr_c', 'te_g', 'te_f', 'te_c')


def snaps(tag):
    out = []
    while os.path.exists(os.path.join(CKPT, f'{tag}_s{len(out)}.pt')):
        out.append(len(out))
    return out


def load(tag, s):
    ck = torch.load(os.path.join(CKPT, f'{tag}_s{s}.pt'), map_location='cuda',
                    weights_only=False)
    st = ck['state']
    hm1.DOG = (st['stem.0.weight'].shape[1] == 4)
    # resnet18 has 2 blocks in layer4, resnet34 has 3 -> detect from the weights
    arch = 'resnet34' if any(k.startswith('l4.2.') for k in st) else 'resnet18'
    m = FoveaNet(arch).cuda()
    m.load_state_dict(st)
    m.eval()
    return m


def entropy_of(h):
    f = h.flatten(2)
    mx = f.max(2, keepdim=True).values
    f = torch.where(f >= mx - GATE, f, torch.full_like(f, -1e30))
    p = torch.softmax(f, 2)
    ent = -(p * torch.log(p + 1e-30)).sum(2)[:, 0]
    w = h.shape[-1]
    return (ent / np.log(w * w)).cpu().numpy()


@torch.no_grad()
def views(model, ds):
    """gated coords, flat coords, entropy (all in the dataset's frame)."""
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    G, F, C = [], [], []
    for x, _, _ in dl:
        g = np.zeros((x.shape[0], 2)); f = np.zeros((x.shape[0], 2))
        c = np.zeros(x.shape[0])
        for mirror in (False, True):
            xx = torch.flip(x, dims=[3]) if mirror else x
            h, _ = model(xx.cuda())
            if mirror:
                h = torch.flip(h, dims=[3])
            g += hm1.soft_argmax(h).cpu().numpy() / 2
            f += hm1.soft_argmax(h, gate=0.0).cpu().numpy() / 2
            c += entropy_of(h) / 2
        G.append(g); F.append(f); C.append(c)
    return np.concatenate(G), np.concatenate(F), np.concatenate(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--tags', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--centers', default=None,
                    help='name of a full_<centers>.npz produced at stage 1')
    ap.add_argument('--dog', action='store_true')
    a = ap.parse_args()

    hm1.DOG = a.dog
    items, te = train_items(), test_items()
    tr_c = te_c = None
    if a.stage == 2:
        d = np.load(os.path.join(WORK, f'full_{a.centers}.npz'))
        tr_c, te_c = d['train_canon'], d['test_canon']

    acc = {k: None for k in KEYS}
    nmod = 0
    for tag in a.tags.split(','):
        ss = snaps(tag)
        assert ss, f'no snapshots for {tag}'
        print(f'{tag}: {len(ss)} snapshots', flush=True)
        for s in ss:
            m = load(tag, s)
            if a.stage == 1:
                g, f, c = (np.asarray(v) * CELL2CANON for v in
                           views(m, FoveaDS(items, 'train', train=False)))
                g2, f2, c2 = (np.asarray(v) * CELL2CANON for v in
                              views(m, FoveaDS(te, 'test', train=False)))
            else:
                def run(geos, cen, split):
                    gg, ff, cc = views(m, PatchDS(geos, cen, split, train=False))
                    base = np.clip(np.asarray(cen, float), PATCH / 2,
                                   hm1.CANON - PATCH / 2) - PATCH / 2
                    return base + gg * CELL2CANON2, base + ff * CELL2CANON2, cc
                g, f, c = run(items, tr_c, 'train')
                g2, f2, c2 = run(te, te_c, 'test')
            cur = dict(tr_g=g, tr_f=f, tr_c=np.asarray(c),
                       te_g=g2, te_f=f2, te_c=np.asarray(c2))
            if acc['tr_g'] is None:
                acc = {k: np.array(v, dtype=np.float64) for k, v in cur.items()}
            else:
                for k in KEYS:
                    acc[k] += cur[k]
            nmod += 1
            del m
            torch.cuda.empty_cache()
    for k in KEYS:
        acc[k] /= nmod

    gt = np.array([[g.gt_x, g.gt_y] for g in items])
    tr_orig = np.array([items[i].to_orig(*acc['tr_g'][i]) for i in range(len(items))])
    e = np.linalg.norm(tr_orig - gt, axis=1)
    print(f'[sanity] fit on the 80 training images: median {np.median(e):.1f} px, '
          f'p90 {np.percentile(e, 90):.1f} px (small = pipeline healthy)')

    te_orig = np.array([te[i].to_orig(*acc['te_g'][i]) for i in range(len(te))])
    te_flat = np.array([te[i].to_orig(*acc['te_f'][i]) for i in range(len(te))])
    out = os.path.join(WORK, f'full_{a.name}.npz')
    np.savez(out, train_canon=acc['tr_g'], test_canon=acc['te_g'],
             train_orig=tr_orig, test_orig=te_orig, test_flat=te_flat,
             train_conf=acc['tr_c'], test_conf=acc['te_c'], n_models=nmod)
    print(f'averaged {nmod} snapshots -> {out}')


if __name__ == '__main__':
    main()

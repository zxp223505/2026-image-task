# -*- coding: utf-8 -*-
"""hm2.py -- stage 2: high-resolution refinement around a coarse fovea estimate.

Stage 1 (full 2000px canonical frame -> 512) reaches ~30px.  Stage 2 crops a
600px window *at canonical (near native) resolution* centred on the stage-1
prediction, feeds it at 384 and regresses the residual.  Because only 80
training images exist, stage-2 fold k is trained with the leak-free stage-1
out-of-fold predictions as crop centres (each produced by the stage-1 model
that never saw that image).
"""
import os, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import hm1

from common import (train_items, test_items, CANON, WORK, affine_about, apply_pt,
                    gauss_map, to_input)
from build_canon import path_of
from hm1 import (FoveaNet, soft_argmax, Prefetch, split_folds, N_FOLDS, SEED,
                 key_of, rmse)

CKPT = os.path.join(WORK, 'ckpt')
PATCH = 700
SIZE2 = 448
STRIDE = 4
OUTSZ2 = SIZE2 // STRIDE
SIGMA = 2.0
CELL2CANON2 = STRIDE * PATCH / SIZE2          # canonical px per heatmap cell
JITTER = 60.0                                  # training centre jitter (px)


class PatchDS(Dataset):
    def __init__(self, geos, centers, split, train, size=SIZE2, patch=PATCH,
                 jitter=JITTER, aug=1.0, fixed_rng=None):
        self.geos = geos
        self.centers = centers
        self.split = split
        self.train = train
        self.size = size
        self.patch = patch
        self.jitter = jitter
        self.aug = aug
        self.outsz = size // STRIDE
        self.rng = fixed_rng if fixed_rng is not None else np.random.default_rng(0)

    def __len__(self):
        return len(self.geos)

    def _region(self, canon, pcx, pcy, half):
        """Copy only the needed window out of the (memmapped) canonical image."""
        side = 2 * half
        x0 = int(round(pcx)) - half
        y0 = int(round(pcy)) - half
        out = np.zeros((side, side, 3), np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(canon.shape[1], x0 + side), min(canon.shape[0], y0 + side)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = canon[sy0:sy1, sx0:sx1]
        return out, x0, y0

    def __getitem__(self, i):
        g = self.geos[i]
        canon = np.load(path_of(key_of(g, self.split)), mmap_mode='r')
        gt = g.canon_gt()
        if self.train:
            tx, ty = gt
            # mixture: mostly stage-1-sized errors, sometimes a big failure
            sig = self.jitter if self.rng.random() < 0.75 else 2.6 * self.jitter
            jx = self.rng.normal(0, sig)
            jy = self.rng.normal(0, sig)
            pcx = tx + jx
            pcy = ty + jy
        else:
            pcx, pcy = self.centers[i]
            tx, ty = gt if gt is not None else (pcx, pcy)

        pcx = float(np.clip(pcx, self.patch / 2, CANON - self.patch / 2))
        pcy = float(np.clip(pcy, self.patch / 2, CANON - self.patch / 2))
        left = pcx - self.patch / 2
        top = pcy - self.patch / 2

        # window big enough for the patch + augmentation margins
        half = int(np.ceil(self.patch * 0.75)) + 8
        reg, rx0, ry0 = self._region(canon, pcx, pcy, half)

        M0 = np.array([[1., 0., -left + rx0], [0., 1., -top + ry0], [0., 0., 1.]])
        Mc = np.array([[1., 0., -left], [0., 1., -top], [0., 0., 1.]])
        if self.train:
            a = self.aug
            flip = self.rng.random() < 0.5
            ang = self.rng.uniform(-10, 10) * a
            sc = 1.0 + self.rng.uniform(-0.08, 0.08) * a
            trx = self.rng.uniform(-25, 25) * a
            try_ = self.rng.uniform(-25, 25) * a
        else:
            flip, ang, sc, trx, try_ = False, 0.0, 1.0, 0.0, 0.0

        Fl = np.array([[-1., 0., self.patch], [0., 1., 0.], [0., 0., 1.]]) if flip else np.eye(3)
        A = affine_about((self.patch / 2, self.patch / 2), ang, sc, trx, try_)
        k = self.size / self.patch
        D = np.array([[k * hm1.ZOOM, 0., 0.], [0., k * hm1.ZOOM, 0.], [0., 0., 1.]])
        base = D @ A @ Fl
        M = base @ M0                           # reg-array px -> input px
        Mt = base @ Mc                          # canonical px -> input px

        img = cv2.warpAffine(reg, M[:2], (self.size, self.size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        px, py = apply_pt(Mt, tx, ty)
        if self.train:
            from hm1 import colour_jitter, STRONG_AUG
            img = colour_jitter(img, self.rng, self.aug, strong=STRONG_AUG)
        if hm1.NORM:
            img = hm1.illum_norm(img, k=hm1.NORM_K2)

        cx, cy = px / STRIDE, py / STRIDE
        hm = gauss_map((self.outsz, self.outsz), cx, cy, SIGMA)
        hm /= max(hm.sum(), 1e-6)
        return (torch.from_numpy(to_input(img, self.size, hm1.DOG)),
                torch.from_numpy(hm)[None].float(),
                torch.from_numpy(np.array([cx, cy], np.float32)))


@torch.no_grad()
def infer2(model, geos, centers, split, tta=True, bs=8):
    was_training = model.training
    model.eval()
    ds = PatchDS(geos, centers, split, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    n = len(geos)
    out = np.zeros((n, 2))
    cur = 0
    for x, _, _ in dl:
        b = x.shape[0]
        acc = np.zeros((b, 2))
        views = [False, True] if tta else [False]
        for fl in views:
            xx = torch.flip(x, dims=[3]) if fl else x
            h, _ = model(xx.cuda())
            ph = soft_argmax(h).cpu().numpy()
            if fl:
                ph[:, 0] = OUTSZ2 - ph[:, 0]
            acc += ph
        off = acc / len(views) * CELL2CANON2          # offset inside patch, px
        cen = np.asarray(centers[cur:cur + b], float)
        cen = np.clip(cen, PATCH / 2, CANON - PATCH / 2)
        out[cur:cur + b, 0] = cen[:, 0] - PATCH / 2 + off[:, 0]
        out[cur:cur + b, 1] = cen[:, 1] - PATCH / 2 + off[:, 1]
        cur += b
    if was_training:
        model.train()
    return out


def train_fold(k, tr, tr_centers, va, va_centers, epochs, arch='resnet18',
               lr=1e-3, lr_bb=2e-4, bs=8, aug=1.0, seed=SEED, tag='hm2',
               eval_every=5, verbose=True, snap_from=20, snap_every=10):
    torch.manual_seed(seed + k)
    ds = PatchDS(tr, tr_centers, 'train', train=True, aug=aug,
                 fixed_rng=np.random.default_rng(seed * 100 + k + 7))
    dl = Prefetch(DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0), 4)

    model = FoveaNet(arch).cuda()
    head = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith('l4')]
    bb = [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith('l4')]
    opt = torch.optim.AdamW([{'params': head, 'lr': lr}, {'params': bb, 'lr': lr_bb}],
                            weight_decay=1e-4)
    steps = epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, lr_bb], total_steps=steps, pct_start=0.3,
        div_factor=20.0, final_div_factor=500.0)

    best = (1e9, None, -1)
    ns = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, hm_t, co_t in dl:
            x, hm_t, co_t = x.cuda(), hm_t.cuda(), co_t.cuda()
            h, r = model(x)
            l_hm = -(hm_t.flatten(2) * F.log_softmax(h.flatten(2), 2)).sum(1).mean()
            loss = l_hm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if ep % eval_every == 0 or ep == epochs:
            p = infer2(model, va, va_centers, 'train', tta=False)
            s = rmse(p, va)[0]
            if s < best[0]:
                best = (s, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}, ep)
            if verbose:
                print(f'  f2 fold{k} ep{ep:3d} loss {tot / nb:6.2f} rmse {s:6.2f} '
                      f'| best {best[0]:6.2f}@{best[2]}', flush=True)
            # fixed-epoch snapshots: selecting "the best epoch" on 16 val images
            # is optimistically biased, so the final prediction averages these
            # instead of trusting the argmin over epochs.
            if ep >= snap_from and (ep - snap_from) % snap_every == 0:
                snap = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
                torch.save({'state': snap, 'ep': ep, 'arch': arch},
                           os.path.join(CKPT, f'{tag}_fold{k}_s{ns}.pt'))
                ns += 1
    path = os.path.join(CKPT, f'{tag}_fold{k}.pt')
    torch.save({'state': best[1], 'ep': best[2], 'score': best[0], 'arch': arch,
                'n_snap': ns}, path)
    return best[0], path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=90)
    ap.add_argument('--tag', default='hm2')
    ap.add_argument('--oof', default='hm1')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--arch', default='resnet18')
    ap.add_argument('--aug', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--snap_from', type=int, default=20)
    ap.add_argument('--snap_every', type=int, default=10)
    ap.add_argument('--dog', action='store_true')
    args = ap.parse_args()
    hm1.DOG = args.dog

    # crop centres come from the stage-1 member.  Prefer the no-selection
    # snapshot-averaged file (mem3_*) when it exists.
    p3 = os.path.join(WORK, f'mem3_{args.oof}.npz')
    if os.path.exists(p3):
        oof_centers = np.load(p3)['oof_canon']
        print('centres <-', p3)
    else:
        oof_centers = np.load(os.path.join(WORK, f'oof_{args.oof}.npz'))['oof_hm']
        print('centres <- oof_%s.npz' % args.oof)
    items, folds = split_folds()
    want = [int(v) for v in args.folds.split(',')]
    scores = []
    for k in range(N_FOLDS):
        if k not in want:
            continue
        va_i, tr_i = folds[k], [i for f in range(N_FOLDS) if f != k for i in folds[f]]
        va = [items[i] for i in va_i]
        tr = [items[i] for i in tr_i]
        print(f'== f2 fold {k}: train {len(tr)} / val {len(va)} ==', flush=True)
        s, p = train_fold(k, tr, oof_centers[tr_i], va, oof_centers[va_i],
                          args.epochs, arch=args.arch, aug=args.aug, tag=args.tag,
                          seed=args.seed, snap_from=args.snap_from,
                          snap_every=args.snap_every)
        scores.append(s)
        print(f'  f2 fold{k} BEST {s:.2f} -> {p}', flush=True)
    if scores:
        print(f'\nSTAGE2 CV {[round(s, 2) for s in scores]} mean {np.mean(scores):.2f}')


if __name__ == '__main__':
    main()

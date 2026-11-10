# -*- coding: utf-8 -*-
"""train_full.py -- final models trained on ALL 80 labelled images.

Why this is legitimate
----------------------
Earlier every model held out 1/5 of the data for validation, and "best epoch"
was picked on that held-out fold -- which turned out to be badly optimistically
biased (selected 27.62 vs 33-40 for any fixed epoch on hm1b stage 1).

With the fixed-epoch-snapshot protocol no validation signal is used at all, so
there is no reason to sacrifice 20% of the data.  Every model here sees all 80
images and the prediction is the average of many fixed-epoch snapshots.
The 5-fold OOF number (stage 1: 32.4, stage 2: 27.6, blended: 25.3) therefore
*understates* what these models can do, since they get 25% more data and more
ensemble members.
"""
import os, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hm1
import hm2
from hm1 import (FoveaDS, FoveaNet, Prefetch, split_folds, CELL2CANON,
                 soft_argmax, colour_jitter, STRONG_AUG)
from hm2 import PatchDS, CELL2CANON2, PATCH
from common import train_items, test_items, WORK

CKPT = os.path.join(WORK, 'ckpt')


def loss_hm(h, hm_t):
    return -(hm_t.flatten(2) * F.log_softmax(h.flatten(2), 2)).sum(1).mean()


def train_stage1(tr, epochs, seed, tag, snap_from, snap_every, aug, lr, lr_bb, bs,
                 arch='resnet18'):
    torch.manual_seed(seed)
    ds = FoveaDS(tr, 'train', train=True, aug=aug)
    ds.rng = np.random.default_rng(seed)
    dl = Prefetch(DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0), 4)

    model = FoveaNet(arch).cuda()
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('l4')]
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('l4')]
    opt = torch.optim.AdamW([{'params': head, 'lr': lr}, {'params': bb, 'lr': lr_bb}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, lr_bb], total_steps=epochs * max(1, len(dl)),
        pct_start=0.3, div_factor=20.0, final_div_factor=500.0)

    ns = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, hm_t, _ in dl:
            x, hm_t = x.cuda(), hm_t.cuda()
            h, _ = model(x)
            loss = loss_hm(h, hm_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        if ep >= snap_from and (ep - snap_from) % snap_every == 0:
            st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state': st, 'ep': ep}, os.path.join(CKPT, f'{tag}_s{ns}.pt'))
            ns += 1
            print(f'  {tag} ep{ep:3d} loss {tot / nb:6.3f}  (snapshot {ns})', flush=True)
        elif ep % 20 == 0:
            print(f'  {tag} ep{ep:3d} loss {tot / nb:6.3f}', flush=True)
    print(f'{tag}: saved {ns} snapshots', flush=True)
    return ns


def train_stage2(tr, centers, epochs, seed, tag, snap_from, snap_every, aug, lr, lr_bb, bs,
                 arch='resnet18'):
    torch.manual_seed(seed)
    ds = PatchDS(tr, centers, 'train', train=True, aug=aug,
                 fixed_rng=np.random.default_rng(seed + 7))
    dl = Prefetch(DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0), 4)

    model = FoveaNet(arch).cuda()
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('l4')]
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('l4')]
    opt = torch.optim.AdamW([{'params': head, 'lr': lr}, {'params': bb, 'lr': lr_bb}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, lr_bb], total_steps=epochs * max(1, len(dl)),
        pct_start=0.3, div_factor=20.0, final_div_factor=500.0)

    ns = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, hm_t, _ in dl:
            x, hm_t = x.cuda(), hm_t.cuda()
            h, _ = model(x)
            loss = loss_hm(h, hm_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        if ep >= snap_from and (ep - snap_from) % snap_every == 0:
            st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state': st, 'ep': ep}, os.path.join(CKPT, f'{tag}_s{ns}.pt'))
            ns += 1
            print(f'  {tag} ep{ep:3d} loss {tot / nb:6.3f}  (snapshot {ns})', flush=True)
        elif ep % 20 == 0:
            print(f'  {tag} ep{ep:3d} loss {tot / nb:6.3f}', flush=True)
    print(f'{tag}: saved {ns} snapshots', flush=True)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--snap_from', type=int, default=30)
    ap.add_argument('--snap_every', type=int, default=10)
    ap.add_argument('--aug', type=float, default=1.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_bb', type=float, default=2e-4)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--centers', default=None,
                    help='stage-1 tag whose all-data predictions centre stage 2')
    ap.add_argument('--dog', action='store_true', help='extra DoG input channel')
    ap.add_argument('--arch', default='resnet18')
    a = ap.parse_args()

    hm1.DOG = a.dog
    print('DOG input channel:', hm1.DOG, ' arch:', a.arch)
    items = train_items()
    if a.stage == 1:
        train_stage1(items, a.epochs, a.seed, a.tag, a.snap_from, a.snap_every,
                     a.aug, a.lr, a.lr_bb, a.bs, arch=a.arch)
    else:
        # PatchDS ignores the supplied centres while training (it jitters around
        # the ground truth), so the centre array only needs to have the right
        # shape.  Prefer the stage-1 predictions when available.
        p = os.path.join(WORK, f'full_{a.centers}.npz') if a.centers else None
        if p and os.path.exists(p):
            c = np.load(p)['train_canon']
        else:
            c = np.stack([np.asarray(g.canon_gt(), float) for g in items])
        train_stage2(items, c, a.epochs, a.seed, a.tag, a.snap_from, a.snap_every,
                     a.aug, a.lr, a.lr_bb, a.bs, arch=a.arch)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""hm1.py -- stage 1: full-canonical-frame heatmap + coordinate regression.

Input : canonical 2000x2000 fundus frame resized to SIZE (512)
Output: stride-4 heatmap (128x128)  +  direct (x,y) head
Loss  : CE(heatmap vs normalised Gaussian) + w_rg * L1(coords in cell units)
Metric: RMSE in ORIGINAL image pixels, leak-free 5-fold CV (seed 42 split,
        identical to the previous baseline so numbers are comparable).
"""
import os, argparse, threading, queue
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from common import (train_items, test_items, CANON, WORK, affine_about, apply_pt,
                    gauss_map, to_input)
from build_canon import path_of
from build_canon1k import path_of_1k, SRC as SRC1K

CKPT = os.path.join(WORK, 'ckpt')
os.makedirs(CKPT, exist_ok=True)

SIZE = 512
STRIDE = 4
OUTSZ = SIZE // STRIDE
SIGMA = 2.0
N_FOLDS = 5
SEED = 42
CELL2CANON = STRIDE * CANON / SIZE     # 15.625 canonical px per heatmap cell
MASK_DISC = 0.0                        # radius (canonical px) of disc occlusion, 0=off
STRONG_AUG = False                     # wide photometric augment (colour-cast robustness)
NORM = False                           # retinex illumination flattening at input
# k is the downsample grid of the illumination estimate.  Keep the *physical*
# blur (~70 canonical px) constant: stage 1 sees 512 px per 2000 -> k=28,
# stage 2 sees 448 px per 700  -> k=10.
NORM_K1, NORM_K2 = 28, 10
DOG = False                            # extra DoG input channel at the macular scale
ZOOM = 1.0                             # test-time scale (multi-scale TTA hook)
GATE = 2.6                            # peak-gate: drop cells >max-GATE before the
                                       # soft-argmax expectation.  Without it a weak
                                       # secondary peak (optic disc) drags the mean;
                                       # tuned on OOF (35.4 -> 27.6 on stage 1).


def key_of(g, split):
    return '%s/%s' % (split, os.path.basename(g.path))


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
class FoveaDS(Dataset):
    def __init__(self, geos, split, train, aug=1.0, size=SIZE, src=SRC1K,
                 mask_disc=0.0):
        self.geos = geos
        self.split = split
        self.train = train
        self.aug = aug
        self.size = size
        self.src = src
        self.mask_disc = mask_disc if mask_disc else MASK_DISC
        self.q = CANON / src                     # canonical px per source-array px
        self.outsz = size // STRIDE
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.geos)

    def __getitem__(self, i):
        g = self.geos[i]
        arr = path_of_1k(key_of(g, self.split)) if self.src == SRC1K \
            else path_of(key_of(g, self.split))
        canon = np.ascontiguousarray(np.load(arr, mmap_mode='r'))
        if self.mask_disc > 0:
            dx, dy = g.canon_disc()
            cv2.circle(canon, (int(dx * self.q), int(dy * self.q)),
                       int(self.mask_disc * self.q), (0, 0, 0), -1)
        gt = g.canon_gt()
        tx, ty = gt if gt is not None else (CANON / 2, CANON / 2)

        if self.train:
            a = self.aug
            flip = self.rng.random() < 0.5
            ang = self.rng.uniform(-8, 8) * a
            sc = 1.0 + self.rng.uniform(-0.05, 0.05) * a
            trx = self.rng.uniform(-60, 60) * a
            try_ = self.rng.uniform(-60, 60) * a
        else:
            flip, ang, sc, trx, try_ = False, 0.0, 1.0, 0.0, 0.0

        Fl = np.array([[-1., 0., CANON], [0., 1., 0.], [0., 0., 1.]]) if flip else np.eye(3)
        A = affine_about((CANON / 2, CANON / 2), ang, sc, trx, try_)
        k = self.size / CANON
        D = np.array([[k * ZOOM, 0., 0.], [0., k * ZOOM, 0.], [0., 0., 1.]])
        M = D @ A @ Fl                            # canonical px -> input px
        Q = np.diag([self.q, self.q, 1.0])         # source-array px -> canonical px
        Msrc = M @ Q

        img = cv2.warpAffine(canon, Msrc[:2], (self.size, self.size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        px, py = apply_pt(M, tx, ty)
        if self.train:
            img = colour_jitter(img, self.rng, self.aug, strong=STRONG_AUG)
        if NORM:
            img = illum_norm(img, k=NORM_K1)

        cx, cy = px / STRIDE, py / STRIDE
        hm = gauss_map((self.outsz, self.outsz), cx, cy, SIGMA)
        hm /= max(hm.sum(), 1e-6)
        coord = np.array([cx, cy], np.float32)
        return (torch.from_numpy(to_input(img, self.size, DOG)),
                torch.from_numpy(hm)[None].float(),
                torch.from_numpy(coord))


def illum_norm(img, k=28, level=120.0):
    """Retinex-style illumination flattening.

    Divides by a heavily downsampled copy of itself, which removes the slow
    colour/illumination gradient that makes some fundus photos nearly unreadable
    while keeping the ~300 px dark macular blob intact.
    """
    h = img.shape[0]
    small = cv2.resize(img, (k, k), interpolation=cv2.INTER_AREA)
    low = cv2.resize(small, (h, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    f = img.astype(np.float32) / (low + 6.0) * level
    out = np.clip(f, 0, 255).astype(np.uint8)
    out[img.max(2) <= 8] = 0
    return out


def colour_jitter(img, rng, a=1.0, strong=False):
    if a <= 0:
        return img
    f = img.astype(np.float32)
    if strong:
        # deliberately wide: several training images are near-monochrome deep red
        # or washed-out yellow, and a narrow range leaves them as hard outliers.
        f *= rng.uniform(1 - 0.35 * a, 1 + 0.35 * a)
        m = f.mean()
        f = (f - m) * rng.uniform(1 - 0.35 * a, 1 + 0.35 * a) + m
        if rng.random() < 0.75 * a:
            f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, rng.uniform(0.55, 1.8))
        if rng.random() < 0.6 * a:
            f = f * rng.uniform(0.65, 1.55, size=(1, 1, 3)).astype(np.float32)
        if rng.random() < 0.5 * a:
            h = f.shape[0]
            g = np.linspace(-1, 1, h, dtype=np.float32)
            ang = rng.uniform(0, 2 * np.pi)
            ramp = (np.cos(ang) * g[:, None] + np.sin(ang) * g[None, :])
            f = f * (1.0 + 0.30 * a * ramp)[..., None]
        return np.clip(f, 0, 255).astype(np.uint8)
    f *= rng.uniform(1 - 0.22 * a, 1 + 0.22 * a)
    m = f.mean()
    f = (f - m) * rng.uniform(1 - 0.22 * a, 1 + 0.22 * a) + m
    if rng.random() < 0.5 * a:
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, rng.uniform(0.75, 1.35))
    if rng.random() < 0.4 * a:
        f = f * rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  model
# --------------------------------------------------------------------------- #
class FoveaNet(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True, freeze_early=True, in_ch=None):
        super().__init__()
        if in_ch is None:
            in_ch = 4 if DOG else 3
        w = 'IMAGENET1K_V1' if pretrained else None
        b = getattr(models, arch)(weights=w)
        if in_ch != 3:
            # Widen conv1 for the extra DoG channel.  The new channel is seeded
            # with the RGB mean so it contributes immediately even before it is
            # fine-tuned -- initialising it to zero while the stem is frozen
            # (freeze_early) would make the extra channel a silent no-op.
            old = b.conv1
            new = nn.Conv2d(in_ch, old.out_channels, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :3] = old.weight
                new.weight[:, 3:] = old.weight.mean(1, keepdim=True)
            b.conv1 = new
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)   # /4
        self.l1 = b.layer1                                            # /4,  64
        self.l2 = b.layer2                                            # /8, 128
        self.l3 = b.layer3                                            # /16,256
        self.l4 = b.layer4                                            # /32,512
        c1, c2, c3, c4 = 64, 128, 256, 512
        self.up4 = nn.Sequential(nn.Conv2d(c4, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.Conv2d(256 + c3, 128, 3, padding=1),
                                  nn.BatchNorm2d(128), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + c2, 96, 3, padding=1),
                                  nn.BatchNorm2d(96), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Conv2d(96 + c1, 64, 3, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU(True))
        self.hm = nn.Conv2d(64, 1, 1)
        self.reg = nn.Sequential(nn.Linear(c4, 128), nn.ReLU(True), nn.Linear(128, 2))
        nn.init.zeros_(self.hm.weight)
        nn.init.constant_(self.hm.bias, -2.0)
        nn.init.zeros_(self.reg[-1].weight)
        nn.init.constant_(self.reg[-1].bias, OUTSZ / 2)
        if freeze_early:
            for m in (self.stem, self.l1, self.l2, self.l3):
                for p in m.parameters():
                    p.requires_grad = False
            if in_ch != 3:
                # conv1 carries the extra DoG channel, so it must stay trainable
                self.stem[0].weight.requires_grad_(True)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        u = self.up4(f4)
        u = F.interpolate(u, size=f3.shape[-2:], mode='nearest')
        u = self.dec3(torch.cat([u, f3], 1))
        u = F.interpolate(u, size=f2.shape[-2:], mode='nearest')
        u = self.dec2(torch.cat([u, f2], 1))
        u = F.interpolate(u, size=f1.shape[-2:], mode='nearest')
        u = self.dec1(torch.cat([u, f1], 1))
        heat = self.hm(u)                                   # (B,1,128,128)
        pooled = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        return heat, self.reg(pooled)


def soft_argmax(heat, gate=None):
    """heat (B,1,H,W) logits -> (B,2) expected (x,y) in cell coords.

    gate: if given, cells more than `gate` below the per-image max are dropped
    before the expectation.  Suppresses a weak secondary peak far away (the
    optic disc) which otherwise drags the global expectation off the fovea.
    """
    b, _, h, w = heat.shape
    flat = heat.flatten(2)
    if gate is None:
        gate = GATE
    if gate:
        mx = flat.max(2, keepdim=True).values
        flat = torch.where(flat >= mx - gate, flat, torch.full_like(flat, -1e30))
    p = F.softmax(flat, dim=2)                                  # (B,1,h*w)
    xs = (torch.arange(w, device=heat.device, dtype=heat.dtype) + 0.5)
    ys = (torch.arange(h, device=heat.device, dtype=heat.dtype) + 0.5)
    fx = xs.view(1, 1, 1, w).expand(b, 1, h, w).reshape(b, 1, h * w)
    fy = ys.view(1, 1, h, 1).expand(b, 1, h, w).reshape(b, 1, h * w)
    px = (p * fx).sum(2)
    py = (p * fy).sum(2)
    return torch.cat([px, py], 1)


# --------------------------------------------------------------------------- #
#  inference
# --------------------------------------------------------------------------- #
class Prefetch:
    """Background-thread prefetcher (DataLoader workers need named pipes, which
    the sandbox forbids, so we overlap CPU work with GPU work ourselves).
    A fresh producer thread is started on every __iter__ so it can be re-used
    for each epoch."""

    def __init__(self, loader, n=4):
        self.loader = loader
        self.n = n

    def __iter__(self):
        q = queue.Queue(maxsize=self.n)

        def run():
            try:
                for b in self.loader:
                    q.put(b)
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        while True:
            b = q.get()
            if b is None:
                return
            yield b

    def __len__(self):
        return len(self.loader)


@torch.no_grad()
def infer(model, geos, split, size=SIZE, tta=True, bs=8, src=SRC1K):
    """(N,2) canonical px predictions for heatmap head and regression head."""
    was_training = model.training
    model.eval()                     # BN must use running stats, never batch stats
    ds = FoveaDS(geos, split, train=False, size=size, src=src)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    n = len(geos)
    hm = np.zeros((n, 2), np.float64)
    rg = np.zeros((n, 2), np.float64)
    cur = 0
    for x, _, _ in dl:
        b = x.shape[0]
        views = [False, True] if tta else [False]
        acc_h = np.zeros((b, 2))
        acc_r = np.zeros((b, 2))
        for fl in views:
            xx = torch.flip(x, dims=[3]) if fl else x
            h, r = model(xx.cuda())
            ph = soft_argmax(h).cpu().numpy()
            pr = r.cpu().numpy()
            if fl:                                     # undo mirror about map centre
                ph[:, 0] = OUTSZ - ph[:, 0]
                pr[:, 0] = OUTSZ - pr[:, 0]
            acc_h += ph
            acc_r += pr
        hm[cur:cur + b] = acc_h / len(views)
        rg[cur:cur + b] = acc_r / len(views)
        cur += b
    if was_training:
        model.train()
    return hm * CELL2CANON, rg * CELL2CANON


def rmse(pred_canon, geos):
    e = []
    for (cx, cy), g in zip(pred_canon, geos):
        ox, oy = g.to_orig(cx, cy)
        e.append((ox - g.gt_x, oy - g.gt_y))
    e = np.array(e)
    rx = float(np.sqrt(np.mean(e[:, 0] ** 2)))
    ry = float(np.sqrt(np.mean(e[:, 1] ** 2)))
    return float(np.sqrt((rx ** 2 + ry ** 2) / 2)), rx, ry


# --------------------------------------------------------------------------- #
#  train
# --------------------------------------------------------------------------- #
def wseed(worker_id):
    pass


def train_fold(k, tr, va, epochs, arch, lr=1e-3, lr_bb=2e-4, bs=8, aug=1.0,
               w_rg=1.0, seed=SEED, tag='hm1', verbose=True, eval_every=5,
               snap_from=40, snap_every=10):
    torch.manual_seed(seed + k)
    ds = FoveaDS(tr, 'train', train=True, aug=aug)
    ds.rng = np.random.default_rng(seed * 100 + k)
    raw = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    dl = Prefetch(raw, n=4)

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
            logp = F.log_softmax(h.flatten(2), dim=2)
            l_hm = -(hm_t.flatten(2) * logp).sum(1).mean()
            l_rg = F.l1_loss(r, co_t)
            loss = l_hm + w_rg * l_rg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if ep % eval_every == 0 or ep == epochs:
            hmp, rgp = infer(model, va, 'train', tta=False)
            s_hm = rmse(hmp, va)[0]
            s_rg = rmse(rgp, va)[0]
            s = s_hm                      # select on the heatmap head (primary)
            if s < best[0]:
                best = (s, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}, ep)
            if verbose:
                print(f'  fold{k} ep{ep:3d} loss {tot / nb:6.2f}  hm {s_hm:6.2f}  '
                      f'rg {s_rg:6.2f}  | best {best[0]:6.2f}@{best[2]}', flush=True)
            if ep >= snap_from and (ep - snap_from) % snap_every == 0:
                snap = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
                torch.save({'state': snap, 'ep': ep, 'arch': arch},
                           os.path.join(CKPT, f'{tag}_fold{k}_s{ns}.pt'))
                ns += 1
    path = os.path.join(CKPT, f'{tag}_fold{k}.pt')
    torch.save({'state': best[1], 'ep': best[2], 'score': best[0], 'arch': arch,
                'n_snap': ns}, path)
    return best[0], path


def split_folds():
    items = train_items()
    idx = np.random.RandomState(SEED).permutation(len(items))
    folds = np.array_split(idx, N_FOLDS)
    return items, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=90)
    ap.add_argument('--arch', default='resnet18')
    ap.add_argument('--aug', type=float, default=1.0)
    ap.add_argument('--tag', default='hm1')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--w_rg', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_bb', type=float, default=2e-4)
    ap.add_argument('--mask_disc', type=float, default=0.0)
    ap.add_argument('--snap_from', type=int, default=40)
    ap.add_argument('--snap_every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--strong_aug', action='store_true')
    ap.add_argument('--norm', action='store_true')
    ap.add_argument('--dog', action='store_true')
    args = ap.parse_args()

    global MASK_DISC, STRONG_AUG, NORM, DOG
    MASK_DISC = args.mask_disc
    STRONG_AUG = args.strong_aug
    NORM = args.norm
    DOG = args.dog

    items, folds = split_folds()
    want = [int(v) for v in args.folds.split(',')]
    scores = []
    t0 = __import__('time').time()
    for k in range(N_FOLDS):
        if k not in want:
            continue
        va = [items[i] for i in folds[k]]
        tr = [items[i] for f in range(N_FOLDS) if f != k for i in folds[f]]
        print(f'== fold {k}: train {len(tr)} / val {len(va)} ==', flush=True)
        s, p = train_fold(k, tr, va, args.epochs, args.arch, aug=args.aug,
                          tag=args.tag, bs=args.bs, w_rg=args.w_rg, lr=args.lr,
                          lr_bb=args.lr_bb,
                          snap_from=args.snap_from, snap_every=args.snap_every,
                          seed=args.seed)
        scores.append(s)
        print(f'  fold{k} BEST {s:.2f}  ({__import__("time").time() - t0:.0f}s)', flush=True)
    if scores:
        print(f'\nCV folds {[round(s, 2) for s in scores]}  mean {np.mean(scores):.2f}')


if __name__ == '__main__':
    main()

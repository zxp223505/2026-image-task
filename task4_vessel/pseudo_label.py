"""Optional self-training (pseudo labels) on the unlabelled test images.

Pipeline
--------
1. `make_submission.py` (or `infer.py`) must have written `pred/<id>_prob.npy`.
2. For every trained fold this script builds confident pseudo labels on the 20 test
   images (prob > 0.85 / < 0.15 are used, 0.15..0.85 is ignored via a weight map).
3. Each fold model is fine-tuned for a few epochs on 50% ground-truth crops +
   50% pseudo-label crops with a very small LR, keeping an EMA copy.
4. Fine-tuned weights are saved to `runs/<run>_pl/` and are picked up automatically
   by `make_submission.py` (use `--exclude "*_pl"` there to ignore them).

This is transductive self-training on unlabelled images only; no test annotations
or external labels are used.
"""
import argparse
import copy
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IMAGENET_MEAN, IMAGENET_STD, WORK, ResNetUNet, build_input, clahe_channel,
                    dice_loss, dice_np, load_test_item, load_train_item, photometric, predict_full,
                    set_seed, test_ids, train_ids)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None, help="run dirs / globs (default: auto discover)")
    p.add_argument("--pred", default=os.path.join(WORK, "pred"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--spe", type=int, default=128, help="samples per epoch (half train / half pseudo)")
    p.add_argument("--bs", type=int, default=4, help="must be even")
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--pl_weight", type=float, default=0.5)
    p.add_argument("--lo", type=float, default=0.30, help="confidence below this is ignored")
    p.add_argument("--hi", type=float, default=0.70, help="confidence above this has full weight")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--suffix", default="_pl")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gt_thr", type=int, default=0,
                   help="GT binarisation for the supervised half: 0 = official thick GT (default)")
    return p.parse_args()


# --------------------------------------------------------------------------- data
class PLDataset(Dataset):
    """Half supervised crops, half pseudo-labelled test crops."""

    def __init__(self, tr_ids, pl_items, crop=384, samples_per_epoch=128, seed=0):
        self.tr = [load_train_item(i) for i in tr_ids]
        self.pl = pl_items
        self.crop = crop
        self.n = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.tr_vpts = [self._vpts(it[1]) for it in self.tr]
        self.pl_vpts = [self._vpts(it["y"]) for it in self.pl]

    @staticmethod
    def _vpts(gt):
        ys, xs = np.nonzero(gt > 0.5)
        if len(ys) == 0:
            return np.zeros((1, 2), np.int64)
        rs = np.random.RandomState(0)
        idx = rs.choice(len(ys), min(len(ys), 20000), replace=False)
        return np.stack([ys[idx], xs[idx]], 1)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState((self.seed * 1000003 + self.epoch * 7919 + idx) % (2 ** 31))
        if idx % 2 == 0:
            k = rng.randint(len(self.tr))
            img, gt, _ = self.tr[k]
            gt = gt.astype(np.float32)
            w = np.ones_like(gt, np.float32)
            vpts = self.tr_vpts[k]
        else:
            k = rng.randint(len(self.pl))
            it = self.pl[k]
            img, gt, w = it["img"], it["y"], it["w"]
            vpts = self.pl_vpts[k]

        C, H, W = self.crop, img.shape[0], img.shape[1]
        if rng.random() < 0.65:
            p = vpts[rng.randint(len(vpts))]
            cy, cx = int(p[0]), int(p[1])
            cy += int(rng.randint(-40, 41))
            cx += int(rng.randint(-40, 41))
        else:
            cy, cx = rng.randint(0, H), rng.randint(0, W)
        y0 = int(np.clip(cy - C // 2, 0, H - C))
        x0 = int(np.clip(cx - C // 2, 0, W - C))
        img = img[y0:y0 + C, x0:x0 + C]
        gt = gt[y0:y0 + C, x0:x0 + C]
        w = w[y0:y0 + C, x0:x0 + C]

        if rng.random() < 0.5:
            img, gt, w = img[:, ::-1], gt[:, ::-1], w[:, ::-1]
        if rng.random() < 0.5:
            img, gt, w = img[::-1], gt[::-1], w[::-1]

        ang = rng.uniform(-180, 180)
        scale = rng.uniform(0.85, 1.2)
        M = cv2.getRotationMatrix2D((C / 2 - 0.5, C / 2 - 0.5), ang, scale)
        M[0, 2] += rng.uniform(-0.06, 0.06) * C
        M[1, 2] += rng.uniform(-0.06, 0.06) * C
        img = cv2.warpAffine(img, M, (C, C), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        gt = cv2.warpAffine(gt, M, (C, C), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w = cv2.warpAffine(w, M, (C, C), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        img = photometric(np.ascontiguousarray(img), rng)
        gt = np.ascontiguousarray(gt).astype(np.float32)
        w = np.clip(np.ascontiguousarray(w), 0, 1).astype(np.float32)

        cl = clahe_channel(img[:, :, 1], rng.uniform(1.0, 4.0))
        rgb = img.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        cl = (cl.astype(np.float32) / 255.0 - 0.5) / 0.25
        x = np.concatenate([rgb.transpose(2, 0, 1), cl[None]], 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(gt[None]), torch.from_numpy(w[None])


def load_model_from(run_dir, device="cuda"):
    path = os.path.join(run_dir, "best_ema.pt")
    if not os.path.exists(path):
        path = os.path.join(run_dir, "best.pt")
    ck = torch.load(path, map_location=device)
    cfg = ck.get("model_cfg") or {}
    model = ResNetUNet(arch=cfg.get("arch", ck.get("arch", "resnet34")),
                       in_ch=cfg.get("in_ch", 4), dec=tuple(cfg.get("dec", (256, 128, 64, 32))),
                       se=cfg.get("se", False), aux=cfg.get("aux", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def build_pl_items(a):
    items = []
    for i in test_ids():
        p = os.path.join(a.pred, f"{i}_prob.npy")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run `python src/make_submission.py` first")
        prob = np.load(p).astype(np.float32)
        img, fov = load_test_item(i)
        y = (prob > 0.5).astype(np.float32)
        conf = np.abs(prob - 0.5) * 2.0
        w = np.clip((conf - a.lo) / max(a.hi - a.lo, 1e-6), 0.0, 1.0).astype(np.float32)
        w *= (fov > 0)
        items.append({"id": i, "img": img, "y": y, "w": w})
    used = np.mean([it["w"].mean() for it in items])
    print(f"pseudo labels: {len(items)} images, mean confident-pixel weight {used:.3f}", flush=True)
    return items


# --------------------------------------------------------------------------- train
@torch.no_grad()
def eval_fold(model, val_ids, tta=1):
    ds = []
    for i in val_ids:
        img, gt, fov = load_train_item(i)
        p = predict_full(model, build_input(img)[None].cuda(), tta=tta)
        ds.append(dice_np((p > 0.5) & (fov > 0), gt))
    return float(np.mean(ds))


def finetune(run_dir, tr_ids, val_ids, pl_items, a, tag):
    out_dir = os.path.join(WORK, "runs", tag)
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, ck = load_model_from(run_dir)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_t = list(ema.state_dict().values())
    mod_t = list(model.state_dict().values())

    ds = PLDataset(tr_ids, pl_items, crop=a.crop, samples_per_epoch=a.spe, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, a.spe // a.bs) * a.epochs
    bce = torch.nn.BCEWithLogitsLoss()
    step = 0
    best = -1.0
    ckpt_meta = {"arch": cfg.get("arch", "resnet34"), "model_cfg": cfg}

    # keep the un-tuned model as the initial "best" so pseudo-label training can only improve it
    if val_ids:
        best = eval_fold(ema, val_ids, tta=1)
        torch.save({"model": ema.state_dict(), "epoch": -1, "dice": best, **ckpt_meta},
                   os.path.join(out_dir, "best_ema.pt"))
        print(f"  [{tag}] base val dice {best:.4f} (kept unless fine-tuning beats it)", flush=True)

    log_f = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["epoch", "loss_train", "loss_pseudo", "val_dice", "best"])
    t0 = time.time()

    for ep in range(a.epochs):
        model.train()
        ds.set_epoch(ep)
        lts, lps = [], []
        for x, y, w in dl:
            x, y, w = x.cuda(non_blocking=True), y.cuda(non_blocking=True), w.cuda(non_blocking=True)
            ratio = 1e-6 + 0.5 * (1 - 1e-6) * (1 + np.cos(np.pi * step / steps))
            for g in opt.param_groups:
                g["lr"] = a.lr * ratio
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                ot, op = out[0::2], out[1::2]
                yt, yp, wp = y[0::2], y[1::2], w[0::2]
                lt = bce(ot, yt) + dice_loss(ot, yt)
                lp = (F.binary_cross_entropy_with_logits(op, yp, reduction="none") * wp).sum() / (wp.sum() + 1e-6)
                lp = lp + dice_loss(op, yp)
                loss = lt + a.pl_weight * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            with torch.no_grad():
                decay = min(0.99, (1 + step) / (10 + step))
                for pe, pm in zip(ema_t, mod_t):
                    if pe.dtype.is_floating_point:
                        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                    else:
                        pe.copy_(pm)
            lts.append(float(lt.detach()))
            lps.append(float(lp.detach()))

        vd = float("nan")
        if val_ids and ((ep + 1) % a.val_every == 0 or ep == a.epochs - 1):
            vd = eval_fold(ema, val_ids, tta=1)
            if vd > best:
                best = vd
                torch.save({"model": ema.state_dict(), "epoch": ep, "dice": vd, "arch": cfg.get("arch", "resnet34"),
                            "model_cfg": cfg}, os.path.join(out_dir, "best_ema.pt"))
        writer.writerow([ep + 1, f"{np.mean(lts):.4f}", f"{np.mean(lps):.4f}", f"{vd:.4f}", f"{best:.4f}"])
        log_f.flush()
        print(f"  [{tag}] ep {ep+1}/{a.epochs} train {np.mean(lts):.4f} pseudo {np.mean(lps):.4f} "
              f"val {vd:.4f} best {best:.4f} {time.time()-t0:.0f}s", flush=True)

    if not val_ids:
        torch.save({"model": ema.state_dict(), "epoch": a.epochs - 1, "dice": None,
                    "arch": cfg.get("arch", "resnet34"), "model_cfg": cfg},
                   os.path.join(out_dir, "best_ema.pt"))
    elif best < 0:
        torch.save({"model": ema.state_dict(), "epoch": a.epochs - 1, "dice": None,
                    "arch": cfg.get("arch", "resnet34"), "model_cfg": cfg},
                   os.path.join(out_dir, "best_ema.pt"))
    log_f.close()
    return best


def main():
    a = get_args()
    set_seed(a.seed)
    import common
    common.GT_THR = a.gt_thr
    torch.backends.cudnn.benchmark = True
    if a.bs % 2:
        raise SystemExit("--bs must be even (half supervised / half pseudo)")

    runs = []
    for r in (a.runs or [os.path.join(WORK, "runs", "*")]):
        runs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])
    runs = [r for r in runs if os.path.isdir(r) and not os.path.basename(r).endswith(a.suffix)]
    if not runs:
        raise SystemExit("no run dirs found")

    ids = train_ids()
    pl_items = build_pl_items(a)
    print("fine-tuning runs:", [os.path.basename(r) for r in runs], flush=True)

    for d in runs:
        tag = os.path.basename(d) + a.suffix
        ck_path = os.path.join(d, "best_ema.pt")
        if not os.path.exists(ck_path):
            ck_path = os.path.join(d, "best.pt")
        ck = torch.load(ck_path, map_location="cpu")
        fold = (ck.get("args") or {}).get("fold", -1)
        val_ids = ids[fold::5] if fold >= 0 else []
        tr_ids = [i for i in ids if i not in val_ids]
        print(f"\n=== {tag}: fold {fold}, train {len(tr_ids)}, val {len(val_ids)} ===", flush=True)
        dice = finetune(d, tr_ids, val_ids, pl_items, a, tag)
        print(f"=== {tag} done, val dice {dice:.4f} ===", flush=True)

    print("\nAll done. Re-run `python src/make_submission.py` to ensemble the *_pl models.")


if __name__ == "__main__":
    main()

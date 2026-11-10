"""Train one CV fold of the vessel segmentation model.

Improvements over the baseline:
  * SE decoder blocks + full-resolution head (optional, checkpoint-compatible)
  * deep supervision at stride 4 / 16
  * clDice topology loss for thin vessels (soft-skeleton, Shit et al. 2021)
  * differential LR for the pretrained encoder
  * boundary-weighted BCE (focus on vessel edges)
  * faster EMA via foreach ops, resume support (`last.pt`)
"""
import argparse
import copy
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import (WORK, VesselDataset, ResNetUNet, build_input, cldice_loss, dice_loss, dice_np,
                    edge_weighted_bce, load_extra_item, load_train_item, predict_full, set_seed, train_ids)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--arch", type=str, default="resnet34")
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--enc_lr_scale", type=float, default=0.2, help="encoder LR / decoder LR")
    p.add_argument("--spe", type=int, default=192, help="samples per epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_tta", type=int, default=1, help="flip TTA used for validation during training")
    p.add_argument("--tta", type=int, default=4)
    p.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--se", type=int, default=1, help="squeeze-excitation decoder blocks")
    p.add_argument("--aux", type=int, default=1, help="deep supervision")
    p.add_argument("--aux_w", type=float, default=0.3, help="total weight of the two aux losses")
    p.add_argument("--cldice", type=float, default=0.15, help="weight of clDice loss (0 disables)")
    p.add_argument("--cldice_iters", type=int, default=6)
    p.add_argument("--cldice_warmup", type=int, default=10,
                   help="ramp clDice weight over the first N epochs (0 = constant)")
    p.add_argument("--edge_w", type=float, default=1.0,
                   help="extra BCE weight on the 1px vessel boundary (0 disables)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init_from", type=str, default="",
                   help="initialise weights from another checkpoint, e.g. to fine-tune an "
                        "existing model on the thick (>0) GT")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--gt_thr", type=int, default=0,
                   help="binarise resized mask with >gt_thr. Default 0 = official thick-GT (>0) "
                        "convention (matches the 512x512 evaluation GT); use 127 for the thin masks")
    p.add_argument("--extra_dir", type=str, default="",
                   help="external dataset folder with images/ and masks/ (e.g. CHASE); mixed into training")
    p.add_argument("--extra_only", action="store_true",
                   help="train on the external dataset only (for pretraining before DRIVE fine-tuning)")
    p.add_argument("--drive_rep", type=int, default=2, help="repeat DRIVE items when --extra_dir is used")
    p.add_argument("--extra_rep", type=int, default=1, help="repeat external items when --extra_dir is used")
    return p.parse_args()


def make_val(items, ids):
    xs, gts, fovs = [], [], []
    for i in ids:
        img, gt, fov = items[i]
        xs.append(build_input(img))
        gts.append(gt)
        fovs.append(fov)
    return torch.stack(xs), gts, fovs


@torch.no_grad()
def evaluate(model, xs, gts, fovs, tta=1, thrs=(0.5,)):
    model.eval()
    probs = []
    for x in xs:
        p = predict_full(model, x[None].cuda(), tta=tta)
        probs.append(p)
    out = {}
    for t in thrs:
        ds = []
        for p, g, f in zip(probs, gts, fovs):
            pr = (p > t) & (f > 0)
            ds.append(dice_np(pr, g))
        out[t] = (float(np.mean(ds)), ds)
    return out, probs


def main():
    a = get_args()
    set_seed(a.seed)
    common.GT_THR = a.gt_thr   # affects load_train_item for both train crops and validation
    torch.backends.cudnn.benchmark = True

    ids = train_ids()
    if a.fold >= 0:
        val_ids = ids[a.fold::5]
        tr_ids = [i for i in ids if i not in val_ids]
    else:
        val_ids, tr_ids = [], ids

    tag = a.out or f"{a.arch}_f{a.fold}_c{a.crop}_s{a.seed}"
    run_dir = os.path.join(WORK, "runs", tag)
    os.makedirs(os.path.join(run_dir, "oof"), exist_ok=True)
    print("run:", tag, "train:", tr_ids, "val:", val_ids, flush=True)

    items = [load_train_item(i) for i in tr_ids]
    if a.extra_dir:
        ex_imgs = sorted(glob.glob(os.path.join(a.extra_dir, "images", "*")))
        ex_masks = sorted(glob.glob(os.path.join(a.extra_dir, "masks", "*")))
        if not ex_imgs or len(ex_imgs) != len(ex_masks):
            raise SystemExit(f"bad --extra_dir {a.extra_dir}: {len(ex_imgs)} images / {len(ex_masks)} masks")
        extra = [load_extra_item(im, mk) for im, mk in zip(ex_imgs, ex_masks)]
        print(f"extra data: {len(extra)} images from {a.extra_dir} "
              f"(drive x{a.drive_rep}, extra x{a.extra_rep}, extra_only={a.extra_only})", flush=True)
        items = extra * a.extra_rep if a.extra_only else items * a.drive_rep + extra * a.extra_rep

    ds = VesselDataset(tr_ids, crop=a.crop, samples_per_epoch=a.spe, seed=a.seed, items=items)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)

    model_cfg = dict(arch=a.arch, in_ch=4, dec=(256, 128, 64, 32), se=bool(a.se), aux=bool(a.aux))
    init_ck = None
    if a.init_from:
        init_ck = torch.load(a.init_from, map_location="cuda")
        cfg = init_ck.get("model_cfg") or {}
        # follow the architecture of the checkpoint (v1 checkpoints have no model_cfg)
        model_cfg = dict(arch=cfg.get("arch", a.arch), in_ch=cfg.get("in_ch", 4),
                         dec=tuple(cfg.get("dec", (256, 128, 64, 32))),
                         se=cfg.get("se", False), aux=cfg.get("aux", False))
        print("init from:", a.init_from, model_cfg, flush=True)
    model = ResNetUNet(**model_cfg).cuda()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    if init_ck is not None:
        model.load_state_dict(init_ck["model"])
        ema.load_state_dict(init_ck["model"])
    # state_dict() returns references; load_state_dict/resume updates them in place
    ema_tensors = list(ema.state_dict().values())
    model_tensors = list(model.state_dict().values())

    # differential LR: pretrained encoder moves slower than the randomly initialised decoder
    enc_prefix = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
    enc, dec = [], []
    for n, param in model.named_parameters():
        (enc if n.startswith(enc_prefix) else dec).append(param)
    groups = [{"params": enc, "lr": a.lr * a.enc_lr_scale},
              {"params": dec, "lr": a.lr}]
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)

    steps_per_epoch = max(1, a.spe // a.bs)
    total = a.epochs * steps_per_epoch
    warm = 3 * steps_per_epoch

    def lr_ratio(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 1e-6 + 0.5 * (1 - 1e-6) * (1 + np.cos(np.pi * t))

    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16"))
    bce = torch.nn.BCEWithLogitsLoss()
    use_aux = bool(a.aux) and model.aux_enabled

    items = {i: load_train_item(i) for i in val_ids}
    if val_ids:
        xs, gts, fovs = make_val(items, val_ids)

    log_path = os.path.join(run_dir, "log.csv")
    new_log = not (a.resume and os.path.exists(log_path))
    log_f = open(log_path, "a" if not new_log else "w", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["epoch", "step", "lr", "train_loss", "train_dice", "val_dice", "val_ema",
                         "best", "skipped", "sec"])

    best = -1.0
    best_ema = -1.0
    step = 0
    skipped = 0
    start_ep = 0
    t0 = time.time()

    last_path = os.path.join(run_dir, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cuda")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if a.amp == "fp16" and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_ep = int(ck.get("epoch", -1)) + 1
        step = start_ep * steps_per_epoch
        best = float(ck.get("best", -1.0))
        best_ema = float(ck.get("best_ema", -1.0))
        print(f"resume from epoch {start_ep} (best {best:.4f} / ema {best_ema:.4f})", flush=True)

    def save_last(ep):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "arch": a.arch,
                    "model_cfg": model_cfg, "args": vars(a), "best": best, "best_ema": best_ema},
                   last_path)

    for ep in range(start_ep, a.epochs):
        model.train()
        ds.set_epoch(ep)
        losses, dices = [], []
        for x, y in dl:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            ratio = lr_ratio(step)
            for g, blr in zip(opt.param_groups, base_lrs):
                g["lr"] = blr * ratio
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.amp != "none")):
                if use_aux:
                    out, aux1, aux2 = model(x, return_aux=True)
                else:
                    out = model(x)
                if a.edge_w > 0:
                    loss = edge_weighted_bce(out, y, a.edge_w) + dice_loss(out, y)
                else:
                    loss = bce(out, y) + dice_loss(out, y)
                if use_aux:
                    w1, w2 = a.aux_w / 3.0, a.aux_w * 2.0 / 3.0
                    loss = loss + w1 * (bce(aux1, y) + dice_loss(aux1, y))
                    loss = loss + w2 * (bce(aux2, y) + dice_loss(aux2, y))
            if a.cldice > 0:
                # topology term in fp32; weight ramps in over the first epochs
                ramp = 1.0 if a.cldice_warmup <= 0 else min(1.0, (ep + 1) / a.cldice_warmup)
                loss = loss + a.cldice * ramp * cldice_loss(out.float(), y.float(), a.cldice_iters)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ok = torch.isfinite(loss) and torch.isfinite(gn)
            if ok:
                opt.step()
            else:
                skipped += 1
                opt.zero_grad(set_to_none=True)
            step += 1
            with torch.no_grad():
                p = torch.sigmoid(out.float())
                d = (2 * (p * y).sum() / (p.sum() + y.sum() + 1e-6)).item()
                decay = min(0.999, (1 + step) / (10 + step))
                if ok:
                    for pe, pm in zip(ema_tensors, model_tensors):
                        if pe.dtype.is_floating_point:
                            pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                        else:
                            pe.copy_(pm)
            losses.append(loss.item())
            dices.append(d)

        if val_ids and ((ep + 1) % a.val_every == 0 or ep == a.epochs - 1):
            res, _ = evaluate(model, xs, gts, fovs, tta=a.val_tta, thrs=(0.5,))
            res_e, _ = evaluate(ema, xs, gts, fovs, tta=a.val_tta, thrs=(0.5,))
            vd, vde = res[0.5][0], res_e[0.5][0]
            if vd > best:
                best = vd
                torch.save({"model": model.state_dict(), "epoch": ep, "dice": vd, "arch": a.arch,
                            "model_cfg": model_cfg, "args": vars(a)},
                           os.path.join(run_dir, "best.pt"))
            if vde > best_ema:
                best_ema = vde
                torch.save({"model": ema.state_dict(), "epoch": ep, "dice": vde, "arch": a.arch,
                            "model_cfg": model_cfg, "args": vars(a)},
                           os.path.join(run_dir, "best_ema.pt"))
        else:
            vd = vde = float("nan")

        writer.writerow([ep + 1, step, f"{base_lrs[-1] * lr_ratio(step):.2e}", f"{np.mean(losses):.4f}",
                         f"{np.mean(dices):.4f}", f"{vd:.4f}", f"{vde:.4f}", f"{max(best, best_ema):.4f}",
                         skipped, f"{time.time() - t0:.0f}"])
        log_f.flush()
        if (ep + 1) % max(1, a.save_every) == 0 or ep == a.epochs - 1:
            save_last(ep)
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"ep {ep+1:3d} loss {np.mean(losses):.4f} dice {np.mean(dices):.4f} val {vd:.4f} ema {vde:.4f} "
                  f"best {max(best, best_ema):.4f} skip {skipped} {time.time()-t0:.0f}s", flush=True)

    # final OOF predictions with the better of raw/EMA weights, flip TTA
    if val_ids:
        which = "best_ema.pt" if best_ema >= best else "best.pt"
        ck = torch.load(os.path.join(run_dir, which), map_location="cuda")
        model.load_state_dict(ck["model"])
        res, probs = evaluate(model, xs, gts, fovs, tta=a.tta, thrs=(0.5,))
        print(f"FINAL {which} dice@0.5 {res[0.5][0]:.4f} per-image {[round(float(d),3) for d in res[0.5][1]]}", flush=True)
        for i, p, g, f in zip(val_ids, probs, gts, fovs):
            np.save(os.path.join(run_dir, "oof", f"{i}.npy"), p.astype(np.float16))
            np.save(os.path.join(run_dir, "oof", f"{i}_gt.npy"), g)
            np.save(os.path.join(run_dir, "oof", f"{i}_fov.npy"), f)
        with open(os.path.join(run_dir, "result.json"), "w") as fh:
            json.dump({"tag": tag, "fold": a.fold, "which": which, "dice": res[0.5][0],
                       "per_image": [float(d) for d in res[0.5][1]], "args": vars(a)}, fh, indent=2)
    else:
        torch.save({"model": ema.state_dict(), "epoch": a.epochs, "dice": None, "arch": a.arch,
                    "model_cfg": model_cfg, "args": vars(a)}, os.path.join(run_dir, "best_ema.pt"))
    save_last(a.epochs - 1)
    log_f.close()


if __name__ == "__main__":
    main()

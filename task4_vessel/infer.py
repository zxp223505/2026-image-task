"""Ensemble inference on the test set -> 512x512 PNG masks -> submission.csv."""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_submission import validate_rows
from common import WORK, ResNetUNet, build_input, load_test_item, predict_full, set_seed, test_ids


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    cfg = ck.get("model_cfg") or {}
    arch = cfg.get("arch", ck.get("arch", "resnet34"))
    m = ResNetUNet(arch=arch, in_ch=cfg.get("in_ch", 4), dec=tuple(cfg.get("dec", (256, 128, 64, 32))),
                   se=cfg.get("se", False), aux=cfg.get("aux", False)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def remove_small(mask, min_size):
    if min_size <= 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = mask.copy()
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_size:
            out[lab == k] = False
    return out


def hysteresis(prob, fov, hi, lo):
    """Keep weak (>lo) components that contain at least one strong (>hi) pixel."""
    strong = (prob > hi) & (fov > 0)
    weak = (prob > lo) & (fov > 0)
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def postprocess(prob, fov, thr, min_size=0, close=0, hyst_lo=0.0):
    if hyst_lo > 0:
        m = hysteresis(prob, fov, thr, hyst_lo)
    else:
        m = prob > thr
        m &= fov > 0
    if close > 0:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close, close), np.uint8)) > 0
        m &= fov > 0
    m = remove_small(m, min_size)
    return m


def rle_encode(mask):
    dots = np.where(mask.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join(str(r) for r in run_lengths)


def _tile_starts(length, tile, stride):
    if tile >= length:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_img(model, img, args, device):
    """Probability map for one image: multi-scale + optional tiled inference.

    The model is trained on 384px crops; tiled inference at the same scale can match
    that context better than one 512px forward pass.
    """
    def predict_at(im):
        return predict_full(model, build_input(im)[None].to(device), tta=args.tta, rots=args.rots)

    def multiscale(im):
        acc = np.zeros(im.shape[:2], np.float32)
        for sc in args.scales:
            if sc == 1.0:
                im2 = im
            else:
                im2 = cv2.resize(im, None, fx=sc, fy=sc,
                                 interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            p = predict_at(im2)
            if sc != 1.0:
                p = cv2.resize(p, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc += p
        return acc / max(len(args.scales), 1)

    full = multiscale(img)
    tile = int(getattr(args, "tile", 0) or 0)
    if tile and tile < min(img.shape[:2]):
        h, w = img.shape[:2]
        acc = np.zeros((h, w), np.float32)
        wacc = np.zeros((h, w), np.float32)
        for y in _tile_starts(h, tile, args.tile_stride):
            for x in _tile_starts(w, tile, args.tile_stride):
                crop = img[y:y + tile, x:x + tile]
                p = predict_at(crop)
                acc[y:y + tile, x:x + tile] += p
                wacc[y:y + tile, x:x + tile] += 1
        tiled = acc / np.maximum(wacc, 1e-6)
        blend = float(getattr(args, "tile_blend", 0.5))
        return blend * tiled + (1 - blend) * full
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (glob ok)")
    ap.add_argument("--out", default=os.path.join(WORK, "pred"))
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--rots", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_size", type=int, default=0)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--hyst_lo", type=float, default=0.0,
                    help="if >0: keep components with prob>thr that are connected to prob>hyst_lo")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--tile", type=int, default=0, help="tile size for tiled inference (0 = off)")
    ap.add_argument("--tile_stride", type=int, default=192)
    ap.add_argument("--tile_blend", type=float, default=0.5, help="weight of the tiled map")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(0)
    torch.backends.cudnn.benchmark = True

    run_dirs = []
    for r in args.runs:
        run_dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])
    paths = []
    for d in run_dirs:
        ck = os.path.join(d, "best_ema.pt")
        if not os.path.exists(ck):
            ck = os.path.join(d, "best.pt")
        paths.append(ck)
    print("models:", paths, flush=True)

    ids = test_ids()
    out_img = os.path.join(args.out, "image")
    os.makedirs(out_img, exist_ok=True)
    imgs = {i: load_test_item(i) for i in ids}

    # one model at a time (keeps VRAM low even for 10+ model ensembles)
    prob = {i: np.zeros(imgs[i][0].shape[:2], np.float32) for i in ids}
    for k, path in enumerate(paths):
        model = load_model(path, args.device)
        for i in ids:
            prob[i] += predict_img(model, imgs[i][0], args, args.device)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"model {k + 1}/{len(paths)} done", flush=True)
    n_terms = max(len(paths), 1)

    rows = []
    for i in ids:
        img, fov = imgs[i]
        p = prob[i] / n_terms
        if p.shape[0] != 512:
            # trained/inferred at 640 (or another size) but the submission must be 512
            p = cv2.resize(p, (512, 512), interpolation=cv2.INTER_LINEAR)
            fov = load_test_item(i, 512)[1]
        mask = postprocess(p, fov, args.thr, args.min_size, args.close, args.hyst_lo)
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_img, f"{i}.png"))
        np.save(os.path.join(args.out, f"{i}_prob.npy"), p.astype(np.float16))
        rows.append({"Id": i, "Predicted": rle_encode(mask.astype(np.uint8))})
        print(f"test {i}: vessel frac {mask.mean():.4f}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "submission.csv")
    df.to_csv(csv_path, index=False)
    validate_rows(rows, expected_ids=ids, verbose=False)
    print("RLE format check: OK")
    print("wrote", csv_path, df.shape, flush=True)


if __name__ == "__main__":
    main()

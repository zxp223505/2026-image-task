"""Compare inference variants (raw/EMA, multi-scale, flip+rot TTA) on OOF images."""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (WORK, ResNetUNet, build_input, dice_np, load_train_item, predict_full,
                    set_seed, train_ids)
from infer import load_model, postprocess


def load_ckpt(path):
    return load_model(path)


@torch.no_grad()
def probs_at(model, img, scales, tta, rots):
    acc = np.zeros(img.shape[:2], np.float32)
    n = 0
    for sc in scales:
        if sc == 1.0:
            im2 = img
        else:
            im2 = cv2.resize(img, None, fx=sc, fy=sc,
                             interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        p = predict_full(model, build_input(im2)[None].cuda(), tta=tta, rots=rots)
        if sc != 1.0:
            p = cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        acc += p
        n += 1
    return acc / n


VARIANT_SPECS = {
    "ema": [("ema", [1.0], 4, 1)],
    "both": [("ema", [1.0], 4, 1), ("raw", [1.0], 4, 1)],
    "ms_both": [("ema", [0.9, 1.0, 1.1], 4, 1), ("raw", [0.9, 1.0, 1.1], 4, 1)],
    "rot_both": [("ema", [1.0], 4, 4), ("raw", [1.0], 4, 4)],
    "full": [("ema", [0.9, 1.0, 1.1], 4, 4), ("raw", [0.9, 1.0, 1.1], 4, 4)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_SPECS))
    ap.add_argument("--out", default=os.path.join(WORK, "oof_variants.json"))
    ap.add_argument("--probs_out", default="")
    ap.add_argument("--gt_thr", type=int, default=0,
                    help="GT binarisation: 0 = official thick GT (default), 127 = thin masks")
    args = ap.parse_args()
    set_seed(0)
    import common
    common.GT_THR = args.gt_thr
    torch.backends.cudnn.benchmark = True

    run_dirs = []
    for r in args.runs:
        run_dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])
    ids = train_ids()
    store = {v: {} for v in args.variants}
    gts, fovs = {}, {}

    for d in run_dirs:
        ck_path = os.path.join(d, "best_ema.pt")
        if not os.path.exists(ck_path):
            ck_path = os.path.join(d, "best.pt")
        ck = torch.load(ck_path, map_location="cpu")
        ck_args = ck.get("args") or {}
        if "fold" in ck_args:
            fold = ck_args["fold"]
        else:
            import re
            fold = int(re.search(r"_f(\d+)", os.path.basename(d)).group(1))
        val_ids = ids[fold::5]
        models = {"ema": load_ckpt(os.path.join(d, "best_ema.pt"))}
        raw_path = os.path.join(d, "best.pt")
        has_raw = os.path.exists(raw_path)
        if has_raw:
            models["raw"] = load_ckpt(raw_path)
        for i in val_ids:
            img, gt, fov = load_train_item(i)
            gts[i], fovs[i] = gt, fov
            for v in args.variants:
                ps, cnt = np.zeros(img.shape[:2], np.float32), 0
                for name, scales, tta, rots in VARIANT_SPECS[v]:
                    if name == "raw" and not has_raw:
                        continue
                    ps += probs_at(models[name], img, scales, tta, rots)
                    cnt += 1
                store[v][i] = ps / max(cnt, 1)
            print("done", d, i, flush=True)
        del models
        torch.cuda.empty_cache()

    report = {}
    for v in args.variants:
        best = (-1, None, None)
        rows = {}
        for thr in np.arange(0.20, 0.72, 0.02):
            ds = {i: float(dice_np(postprocess(store[v][i], fovs[i], thr), gts[i])) for i in store[v]}
            m = float(np.mean(list(ds.values())))
            rows[round(float(thr), 2)] = m
            if m > best[0]:
                best = (m, float(thr), ds)
        print(f"{v:8s} best dice {best[0]:.4f} @thr {best[1]:.2f}")
        print("   ", {k: round(x, 4) for k, x in rows.items()})
        print("    per-image", {k: round(x, 4) for k, x in best[2].items()})
        report[v] = {"best_dice": best[0], "best_thr": best[1], "per_image": best[2], "sweep": rows}

    with open(args.out, "w") as fh:
        json.dump({"runs": run_dirs, "report": report}, fh, indent=2)
    print("wrote", args.out)

    if args.probs_out:
        os.makedirs(args.probs_out, exist_ok=True)
        np.savez_compressed(os.path.join(args.probs_out, "variants.npz"),
                            **{f"{v}_{i}": store[v][i].astype(np.float16) for v in store for i in store[v]})
        np.savez_compressed(os.path.join(args.probs_out, "gt.npz"),
                            **{str(i): gts[i] for i in gts})
        print("saved probs", args.probs_out)


if __name__ == "__main__":
    main()

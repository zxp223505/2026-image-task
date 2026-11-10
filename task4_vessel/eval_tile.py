"""Compare full-image vs tiled inference on the OOF validation images.

Tiled inference uses the same 384px scale the model was trained on, which can match
the training context better than a single 512px forward pass.
"""
import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import dice_np, load_train_item, set_seed
from infer import load_model, postprocess, predict_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--tile_stride", type=int, default=192)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--gt_thr", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    set_seed(0)
    common.GT_THR = args.gt_thr
    torch.backends.cudnn.benchmark = True

    run_dirs = []
    for r in args.runs:
        run_dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])

    full_pr, tile_pr, gts, fovs = {}, {}, {}, {}
    for d in run_dirs:
        ck_path = os.path.join(d, "best_ema.pt")
        if not os.path.exists(ck_path):
            ck_path = os.path.join(d, "best.pt")
        ck = torch.load(ck_path, map_location="cpu")
        fold = int((ck.get("args") or {}).get("fold", 0))
        ids = common.train_ids()[fold::5]
        model = load_model(ck_path, "cuda")
        for i in ids:
            img, gt, fov = load_train_item(i)
            ns_full = SimpleNamespace(tta=args.tta, rots=1, scales=[1.0], tile=0, tile_stride=args.tile_stride, tile_blend=0.0)
            ns_tile = SimpleNamespace(tta=args.tta, rots=1, scales=[1.0], tile=args.tile,
                                      tile_stride=args.tile_stride, tile_blend=1.0)
            full_pr[i] = predict_img(model, img, ns_full, "cuda")
            tile_pr[i] = predict_img(model, img, ns_tile, "cuda")
            gts[i], fovs[i] = gt, fov
            print("done", i, flush=True)
        del model
        torch.cuda.empty_cache()

    ids = sorted(full_pr)
    print("images:", len(ids))
    report = {}
    for blend in [0.0, 0.3, 0.5, 0.7, 1.0]:
        best = (-1, 0.0)
        for thr in np.arange(0.30, 0.62, 0.02):
            ds = [dice_np(postprocess(blend * tile_pr[i] + (1 - blend) * full_pr[i], fovs[i], thr), gts[i])
                  for i in ids]
            m = float(np.mean(ds))
            if m > best[0]:
                best = (m, float(thr))
        print(f"tile blend {blend:.1f}: dice {best[0]:.4f} @thr {best[1]:.2f}")
        report[blend] = best

    if args.out:
        np.savez_compressed(args.out, **{f"full_{i}": full_pr[i].astype(np.float16) for i in ids},
                            **{f"tile_{i}": tile_pr[i].astype(np.float16) for i in ids})
        print("saved", args.out)


if __name__ == "__main__":
    main()

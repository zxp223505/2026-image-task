"""Evaluate OOF predictions from trained folds; tune threshold / post-processing."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import WORK, dice_np, load_train_item, train_ids
from infer import postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--gt_thr", type=int, default=None,
                    help="recompute GT from labels with >gt_thr; 0 = official thick-GT (>0) convention. "
                         "Default: use the GT saved during training.")
    args = ap.parse_args()
    if args.gt_thr is not None:
        common.GT_THR = args.gt_thr

    run_dirs = []
    for r in args.runs:
        run_dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])

    probs, gts, fovs = {}, {}, {}
    per_run = {}
    for d in run_dirs:
        n = 0
        for p in glob.glob(os.path.join(d, "oof", "*.npy")):
            name = os.path.basename(p)[:-4]
            if name.endswith("_gt") or name.endswith("_fov"):
                continue
            i = int(name)
            pr = np.load(p).astype(np.float32)
            probs.setdefault(i, []).append(pr)
            n += 1
        per_run[os.path.basename(d)] = n
    ids = sorted(probs)
    for i in ids:
        if args.gt_thr is not None:
            _, gt, fov = load_train_item(i)
            gts[i], fovs[i] = gt, fov
            continue
        for d in run_dirs:
            gp = os.path.join(d, "oof", f"{i}_gt.npy")
            if os.path.exists(gp):
                gts[i] = np.load(gp)
                img, _, fov = load_train_item(i)
                fovs[i] = fov
                break
    print("images with OOF:", ids, "counts:", per_run)

    ens = {i: np.mean(np.stack(v), 0) for i, v in probs.items()}

    best = (-1, None)
    # wide range: thin-GT models peak near 0.5, thick-GT/threshold-shifted models near 0.05
    for thr in np.arange(0.02, 0.72, 0.02):
        ds = [dice_np(postprocess(ens[i], fovs[i], thr), gts[i]) for i in ids]
        m = float(np.mean(ds))
        if m > best[0]:
            best = (m, float(thr))
        print(f"thr {thr:.2f} dice {m:.4f}")
    print("BEST", best)

    thr = best[1]
    print("\nper-image @best thr:")
    for i in ids:
        print(i, f"{dice_np(postprocess(ens[i], fovs[i], thr), gts[i]):.4f}")

    print("\nsingle-model dice @0.52:")
    for i in ids:
        print(i, [round(float(dice_np(postprocess(p, fovs[i], 0.52), gts[i])), 4) for p in probs[i]])

    print("\npost-proc sweep:")
    for ms in [0, 10, 20, 30, 50]:
        for cl in [0, 3]:
            ds = [dice_np(postprocess(ens[i], fovs[i], thr, ms, cl), gts[i]) for i in ids]
            print(f"min_size {ms:3d} close {cl} dice {np.mean(ds):.4f}")

    print("\nhysteresis sweep (thr = best above):")
    for lo in [0.10, 0.20, 0.30, 0.40]:
        ds = [dice_np(postprocess(ens[i], fovs[i], thr, 0, 0, lo), gts[i]) for i in ids]
        print(f"hyst_lo {lo:.2f} dice {np.mean(ds):.4f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"runs": run_dirs, "ids": ids, "best_thr": thr, "best_dice": best[0],
                       "per_image": {i: float(dice_np(postprocess(ens[i], fovs[i], thr), gts[i])) for i in ids}},
                      fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

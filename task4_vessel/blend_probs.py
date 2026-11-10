"""Blend probability maps produced by several infer.py runs and write a submission.

Example:
    python src/blend_probs.py --dirs pred_tile pred_640 --weights 0.5 0.5 \
        --thr 0.42 --min_size 30 --out pred_final
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import WORK, load_test_item, test_ids
from check_submission import validate_rows
from infer import postprocess, rle_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="pred_* dirs containing <id>_prob.npy")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min_size", type=int, default=0)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--hyst_lo", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(WORK, "pred_blend"))
    args = ap.parse_args()

    weights = args.weights or [1.0 / len(args.dirs)] * len(args.dirs)
    if len(weights) != len(args.dirs):
        raise SystemExit("--weights must match --dirs")
    s = float(sum(weights))
    weights = [w / s for w in weights]

    ids = test_ids()
    out_img = os.path.join(args.out, "image")
    os.makedirs(out_img, exist_ok=True)
    rows = []
    for i in ids:
        p = np.zeros((512, 512), np.float32)
        for d, w in zip(args.dirs, weights):
            pm = np.load(os.path.join(d, f"{i}_prob.npy")).astype(np.float32)
            if pm.shape != (512, 512):
                raise SystemExit(f"{d}/{i}_prob.npy has shape {pm.shape}, expected (512, 512)")
            p += w * pm
        fov = load_test_item(i, 512)[1]
        mask = postprocess(p, fov, args.thr, args.min_size, args.close, args.hyst_lo)
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_img, f"{i}.png"))
        np.save(os.path.join(args.out, f"{i}_prob.npy"), p.astype(np.float16))
        rows.append({"Id": i, "Predicted": rle_encode(mask.astype(np.uint8))})
        print(f"test {i}: vessel frac {mask.mean():.4f}", flush=True)

    csv_path = os.path.join(args.out, "submission.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    validate_rows(rows, expected_ids=ids, verbose=False)
    print("RLE format check: OK")
    print("wrote", csv_path, flush=True)


if __name__ == "__main__":
    main()

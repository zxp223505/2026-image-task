"""End-to-end pipeline: train folds -> OOF eval -> ensemble inference -> submission.csv.

Usage:
    python src/run_all.py                        # full pipeline (5 folds, v2 model)
    python src/run_all.py --seeds 2              # 2 random seeds per fold (10 models)
    python src/run_all.py --full                 # + one model trained on all 20 images
    python src/run_all.py --skip_train           # reuse existing runs/*/best*.pt
    python src/run_all.py --folds 0 1 --cldice 0 # baseline-style training without clDice
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# VESSEL_WORK lets a code-only snapshot under versions/ reuse the main runs/ folder
WORK = os.environ.get("VESSEL_WORK", os.path.dirname(HERE))
PY = sys.executable


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=WORK)


def tag_of(prefix, fold, seed_idx):
    return f"{prefix}_f{fold}" if seed_idx == 0 else f"{prefix}_f{fold}_s{seed_idx}"


def discover(prefix, folds, seeds, seed_start=0):
    """Find trained run dirs; seed-0 tags accept older suffixed dirs like r34_f0_b."""
    runs = []
    for s in range(seed_start, seeds):
        for f in folds:
            tag = tag_of(prefix, f, s)
            cand = [os.path.join(WORK, f"runs/{tag}")]
            if s == 0:
                cand += sorted(glob.glob(os.path.join(WORK, f"runs/{tag}_*")))
            found = None
            for c in cand:
                if os.path.exists(os.path.join(c, "best.pt")) or os.path.exists(os.path.join(c, "best_ema.pt")):
                    found = os.path.relpath(c, WORK).replace("\\", "/")
                    break
            if found:
                runs.append(found)
            else:
                print(f"skip missing run: {tag}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--spe", type=int, default=192)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=1, help="independent random seeds per fold")
    ap.add_argument("--seed_start", type=int, default=0,
                    help="first seed index to train (use 1 with --seeds 2 to train only the 2nd seed)")
    ap.add_argument("--arch", type=str, default="resnet34")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--se", type=int, default=1)
    ap.add_argument("--aux", type=int, default=1)
    ap.add_argument("--aux_w", type=float, default=0.3)
    ap.add_argument("--cldice", type=float, default=0.15)
    ap.add_argument("--cldice_warmup", type=int, default=10)
    ap.add_argument("--edge_w", type=float, default=1.0)
    ap.add_argument("--enc_lr_scale", type=float, default=0.2)
    ap.add_argument("--gt_thr", type=int, default=0,
                    help="mask binarisation after resize. Default 0 = official thick-GT (>0) "
                         "convention; use 127 for the old thin-mask experiments")
    ap.add_argument("--full", action="store_true",
                    help="also train one model on all 20 images (no validation) and add it to the ensemble")
    ap.add_argument("--full_epochs", type=int, default=80)
    ap.add_argument("--thr", type=float, default=None,
                    help="decision threshold (default: tuned from OOF if available, else 0.5)")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--rots", type=int, default=1)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--hyst_lo", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="continue from runs/*/last.pt if present")
    args = ap.parse_args()

    prefix = args.arch.replace("resnet", "r")
    runs = []
    if not args.skip_train:
        for s in range(args.seed_start, args.seeds):
            for k, f in enumerate(args.folds):
                tag = tag_of(prefix, f, s)
                cmd = [PY, os.path.join(HERE, "train.py"), "--fold", str(f),
                       "--epochs", str(args.epochs), "--spe", str(args.spe), "--arch", args.arch,
                       "--crop", str(args.crop), "--bs", "4", "--lr", str(args.lr),
                       "--enc_lr_scale", str(args.enc_lr_scale),
                       "--seed", str(args.seed_base + 1000 * s + 17 * k),
                       "--amp", "bf16", "--se", str(args.se), "--aux", str(args.aux),
                       "--aux_w", str(args.aux_w), "--cldice", str(args.cldice),
                       "--cldice_warmup", str(args.cldice_warmup), "--edge_w", str(args.edge_w),
                       "--gt_thr", str(args.gt_thr),
                       "--out", tag, "--workers", "0"]
                if args.resume:
                    cmd.append("--resume")
                run(cmd)
                runs.append(f"runs/{tag}")
    else:
        runs = discover(prefix, args.folds, args.seeds, args.seed_start)
        if not runs:
            raise SystemExit("no trained run found, run training first")

    # optional: one model on all 20 images (more data per model) for the ensemble
    full_tag = f"runs/{prefix}_full"
    if args.full:
        if not args.skip_train:
            run([PY, os.path.join(HERE, "train.py"), "--fold", "-1",
                 "--epochs", str(args.full_epochs), "--spe", str(args.spe), "--arch", args.arch,
                 "--crop", str(args.crop), "--bs", "4", "--lr", str(args.lr),
                 "--enc_lr_scale", str(args.enc_lr_scale), "--seed", str(args.seed_base + 5001),
                 "--amp", "bf16", "--se", str(args.se), "--aux", str(args.aux),
                 "--aux_w", str(args.aux_w), "--cldice", str(args.cldice),
                 "--cldice_warmup", str(args.cldice_warmup), "--edge_w", str(args.edge_w),
                 "--gt_thr", str(args.gt_thr),
                 "--out", f"{prefix}_full", "--workers", "0"])
        if os.path.exists(os.path.join(WORK, full_tag, "best_ema.pt")):
            runs.append(full_tag)
        else:
            print("skip full-data model:", full_tag, "not found")

    # OOF evaluation + threshold tuning (the full-data model has no OOF and is ignored there)
    run([PY, os.path.join(HERE, "oof_eval.py"), "--runs"] + runs +
        ["--out", os.path.join(WORK, "runs", "oof_summary.json")])

    infer_cmd = [PY, os.path.join(HERE, "infer.py"), "--runs"] + runs + [
        "--tta", str(args.tta), "--rots", str(args.rots), "--hyst_lo", str(args.hyst_lo),
        "--scales"] + [str(s) for s in args.scales]
    thr = args.thr
    if thr is None:
        summary = os.path.join(WORK, "runs", "oof_summary.json")
        if os.path.exists(summary):
            try:
                thr = float(json.load(open(summary))["best_thr"])
                print(f"using OOF tuned threshold {thr:.2f}")
            except Exception:
                thr = None
    if thr is not None:
        infer_cmd += ["--thr", str(thr)]
    run(infer_cmd)
    print("\nDone. Submission:", os.path.join(WORK, "pred", "submission.csv"))


if __name__ == "__main__":
    main()

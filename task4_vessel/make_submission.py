"""One-click entry: build pred/submission.csv from every trained checkpoint found.

PyCharm: right click -> Run. No arguments required.

It ensembles all `runs/*/best_ema.pt` (or `best.pt`) checkpoints, applies flip TTA,
and writes 512x512 PNG masks to `pred/image/` plus the RLE file `pred/submission.csv`.

Threshold: if `runs/oof_summary.json` exists (produced by oof_eval.py) its tuned
threshold is used, otherwise 0.5.
"""
import argparse
import fnmatch
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _has_runs(work):
    for d in glob.glob(os.path.join(work, "runs", "*")):
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "best_ema.pt"))
                                 or os.path.exists(os.path.join(d, "best.pt"))):
            return True
    return False


def _resolve_work():
    """Main folder of the repo. A code-only snapshot under versions/<name>/ has no
    runs/, so fall back to the repo root (two levels up) when it has checkpoints."""
    env = os.environ.get("VESSEL_WORK")
    if env:
        return env
    work = os.path.dirname(HERE)
    if not _has_runs(work):
        parent = os.path.dirname(os.path.dirname(work))
        if _has_runs(parent):
            print(f"[info] no runs/ next to this copy, using main folder: {parent}")
            return parent
    return work


WORK = _resolve_work()


def find_runs(exclude):
    runs = []
    for d in sorted(glob.glob(os.path.join(WORK, "runs", "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if any(fnmatch.fnmatch(name, pat) for pat in exclude):
            continue
        if os.path.exists(os.path.join(d, "best_ema.pt")) or os.path.exists(os.path.join(d, "best.pt")):
            runs.append(os.path.relpath(d, WORK).replace("\\", "/"))
    return runs


def tuned_threshold():
    p = os.path.join(WORK, "runs", "oof_summary.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            v = json.load(fh).get("best_thr")
            return float(v) if v is not None else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run dirs (default: auto discover)")
    ap.add_argument("--exclude", nargs="*", default=[], help="glob patterns of run names to skip, e.g. '*_pl'")
    ap.add_argument("--thr", type=float, default=None, help="default: tuned OOF threshold, else 0.5")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--rots", type=int, default=1)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--hyst_lo", type=float, default=0.0)
    ap.add_argument("--min_size", type=int, default=0)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(WORK, "pred"))
    args = ap.parse_args()

    runs = args.runs or find_runs(args.exclude)
    if not runs:
        raise SystemExit(
            f"no checkpoint found under {os.path.join(WORK, 'runs')}\n"
            "If you are running a code-only snapshot from versions/, either run the main "
            "retina/src/make_submission.py, or point VESSEL_WORK at the main folder:\n"
            '  $env:VESSEL_WORK = "C:\\\\...\\\\retina"')

    # never mix thin-GT and thick-GT models with one threshold: prefer the thick ones
    thick_runs = [r for r in runs if "_thick" in os.path.basename(r)]
    if thick_runs and len(thick_runs) != len(runs):
        print("[info] thick-GT checkpoints found; ignoring the old thin-GT runs")
        runs = thick_runs
    thick = bool(thick_runs)

    thr = args.thr
    if thr is None:
        thr = tuned_threshold()
        if thr is not None and thick:
            # the saved summary may come from thin-GT tuning; only trust a thick-GT one
            try:
                with open(os.path.join(WORK, "runs", "oof_summary.json")) as fh:
                    s_runs = json.load(fh).get("runs", [])
                if not (s_runs and all("_thick" in os.path.basename(r) for r in s_runs)):
                    thr = None
            except Exception:
                thr = None
        if thr is not None:
            print(f"threshold from runs/oof_summary.json: {thr:.2f}")
        else:
            thr = 0.45 if thick else 0.06
            print("no matching oof_summary.json, threshold =", thr)

    print("checkpoints:")
    for r in runs:
        print("  ", r)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cuda"
    print("device:", device, " thr:", thr, " tta:", args.tta, " rots:", args.rots, flush=True)

    cmd = [sys.executable, os.path.join(HERE, "infer.py"), "--runs"] + runs + [
        "--out", args.out, "--thr", str(thr), "--tta", str(args.tta), "--rots", str(args.rots),
        "--hyst_lo", str(args.hyst_lo), "--min_size", str(args.min_size), "--close", str(args.close),
        "--device", device, "--scales",
    ] + [str(s) for s in args.scales]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=WORK)
    print("\nDone ->", os.path.join(args.out, "submission.csv"))


if __name__ == "__main__":
    main()

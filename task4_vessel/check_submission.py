"""Validate a submission CSV against the competition RLE rules.

Checks
  * columns are exactly Id,Predicted
  * every expected Id is present, no duplicates, no missing
  * each Predicted is a space separated list of (start, length) pairs
  * starts are 1-based, strictly increasing, positive lengths, no overlap
  * total decoded pixels <= 512*512 per image

Usage:
    python src/check_submission.py pred/submission.csv
    python src/check_submission.py pred/submission.csv --ids 1 2 3
"""
import argparse
import os
import sys

import pandas as pd

SIZE = 512
MAX_PIX = SIZE * SIZE


def parse_rle(s, max_pix=MAX_PIX):
    """Return number of decoded pixels; raise ValueError on malformed RLE."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return 0
    parts = str(s).split()
    if len(parts) % 2 != 0:
        raise ValueError(f"odd number of tokens ({len(parts)})")
    vals = [int(p) for p in parts]
    total = 0
    prev_end = 0
    for k in range(0, len(vals), 2):
        start, length = vals[k], vals[k + 1]
        if start < 1:
            raise ValueError(f"start {start} < 1")
        if length < 1:
            raise ValueError(f"length {length} < 1")
        if start <= prev_end:
            raise ValueError(f"start {start} overlaps previous end {prev_end}")
        end = start + length - 1
        if end > max_pix:
            raise ValueError(f"run ends at {end} > {max_pix}")
        total += length
        prev_end = end
    return total


def validate_rows(rows, expected_ids=None, size=SIZE, verbose=True):
    """rows: list of dicts with Id / Predicted. Returns list of pixel counts."""
    max_pix = size * size
    df = pd.DataFrame(rows)
    assert list(df.columns) == ["Id", "Predicted"], f"bad columns: {list(df.columns)}"

    ids = [int(i) for i in df["Id"]]
    if expected_ids is not None:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        assert not missing, f"missing ids: {missing}"
        assert not extra, f"unexpected ids: {extra}"
    assert len(ids) == len(set(ids)), "duplicate Id rows"

    counts = []
    for i, pred in zip(ids, df["Predicted"]):
        try:
            n = parse_rle(pred, max_pix)
        except ValueError as e:
            raise ValueError(f"Id {i}: {e}") from None
        counts.append(n)
        if verbose:
            print(f"  Id {i:>3}: {n:>7} vessel px  frac {n / max_pix:.4f}")

    if verbose:
        mean_frac = sum(counts) / len(counts) / max_pix
        print(f"OK: {len(counts)} rows, mean vessel fraction {mean_frac:.4f}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join("pred", "submission.csv"))
    ap.add_argument("--ids", nargs="*", type=int, default=None, help="expected ids (default 1..20)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"file not found: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.ids is None and "Id" in df.columns:
        args.ids = list(range(1, 21))
    validate_rows(df.to_dict("records"), args.ids, verbose=not args.quiet)
    print("submission format OK:", args.csv)


if __name__ == "__main__":
    main()

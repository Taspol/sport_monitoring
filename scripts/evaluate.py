#!/usr/bin/env python3
"""Evaluate the injury pipeline against a small hand-labelled golden-clip set.

This is the *integration* gate: it scores the two things the whole risk grade
depends on -- landing-event detection and LESS agreement -- by reading the CSVs
``--analyze`` already wrote (``output/<clip>_jumps.csv``). It does NOT re-run the
video pipeline, so it finishes in well under a second.

Two metrics per clip (and aggregated):

  1. Landing detection -- precision / recall / F1 with a frame tolerance.
     A detected jump matches a labelled landing if its initial-contact frame is
     within ``--tol`` frames; matching is greedy nearest-first. Also reports the
     mean |IC frame error| over matched pairs. This is the over/under-count check.

  2. LESS agreement -- MAE, signed bias, Pearson r and ICC(2,1) between the
     predicted and labelled LESS-subset score over matched landings.

Backbone comparison: pass ``--variants yolo_rtm,yolo_mediapipe,rfdetr_rtm,rfdetr_mediapipe``
to print one row per backbone combination (each read from output/<variant>/, with
``yolo_rtm`` falling back to the output root). The table adds a FPS column read from
the ``<clip>_timing.json`` the CLI writes per pass, so detector x pose x speed are
compared on the same golden clips.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --tol 4 --labels benchmark/labels.csv
    uv run python scripts/evaluate.py --variants yolo_rtm,yolo_mediapipe,rfdetr_rtm,rfdetr_mediapipe
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- IO
def _parse_ints(field: str) -> list[int]:
    field = (field or "").strip()
    if not field:
        return []
    return [int(round(float(x))) for x in field.split(";") if x.strip()]


def read_labels(path: Path) -> list[dict]:
    """Read benchmark/labels.csv, skipping '#' comment lines and blank rows."""
    rows: list[dict] = []
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        clip = (r.get("clip") or "").strip()
        if not clip:
            continue
        frames = _parse_ints(r.get("true_landing_frames", ""))
        less = _parse_ints(r.get("less_per_landing", ""))
        rows.append({"clip": clip, "frames": frames, "less": less})
    return rows


def read_predicted(jumps_csv: Path) -> list[tuple[int, int]]:
    """Return [(ic_frame, less_score), ...] from a <clip>_jumps.csv."""
    if not jumps_csv.exists():
        return []
    out: list[tuple[int, int]] = []
    with open(jumps_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append((int(float(r["ic_frame"])), int(float(r["less_score"]))))
    return out


def read_fps(out_dir: Path, clips: list[str]) -> float | None:
    """Mean FPS across the per-clip ``<clip>_timing.json`` files the CLI wrote."""
    vals: list[float] = []
    for clip in clips:
        tj = out_dir / f"{clip}_timing.json"
        if tj.exists():
            try:
                vals.append(float(json.loads(tj.read_text())["fps"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
    return sum(vals) / len(vals) if vals else None


def variant_dir(output_root: Path, variant: str) -> Path:
    """Map a variant id to its output dir (yolo_rtm / '.' live in the root)."""
    return output_root if variant in ("yolo_rtm", ".", "") else output_root / variant


# ---------------------------------------------------------------------- metrics
def match_landings(
    pred_ics: list[int], true_frames: list[int], tol: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy nearest matching of predicted IC frames to labelled landings.

    Returns ``(matches, unmatched_pred_idx, unmatched_true_idx)`` where each match
    is ``(pred_index, true_index)``.
    """
    candidates = []
    for pi, ic in enumerate(pred_ics):
        for ti, tf in enumerate(true_frames):
            d = abs(ic - tf)
            if d <= tol:
                candidates.append((d, pi, ti))
    candidates.sort()
    used_p, used_t, matches = set(), set(), []
    for _, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append((pi, ti))
    unmatched_p = [i for i in range(len(pred_ics)) if i not in used_p]
    unmatched_t = [i for i in range(len(true_frames)) if i not in used_t]
    return matches, unmatched_p, unmatched_t


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def icc_2_1(xs: list[float], ys: list[float]) -> float | None:
    """ICC(2,1): two-way random, single rater, absolute agreement (k=2 raters)."""
    n = len(xs)
    if n < 2:
        return None
    k = 2
    rows = list(zip(xs, ys))
    grand = sum(xs + ys) / (n * k)
    row_means = [sum(r) / k for r in rows]
    col_means = [sum(xs) / n, sum(ys) / n]
    ss_rows = k * sum((rm - grand) ** 2 for rm in row_means)
    ss_cols = n * sum((cm - grand) ** 2 for cm in col_means)
    ss_total = sum((v - grand) ** 2 for r in rows for v in r)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return None
    return (msr - mse) / denom


def _fmt(v: float | None, nd: int = 2) -> str:
    return "  n/a" if v is None else f"{v:.{nd}f}"


# -------------------------------------------------------------------- evaluation
def evaluate_dir(
    labels: list[dict], out_dir: Path, tol: int, verbose: bool
) -> dict | None:
    """Score every labelled clip's CSVs in ``out_dir``; return aggregate metrics.

    When ``verbose`` is set, also print the per-clip table. Returns ``None`` only if
    not a single clip had outputs in this directory.
    """
    tot_tp = tot_fp = tot_fn = 0
    all_pred_less: list[float] = []
    all_true_less: list[float] = []
    all_ic_err: list[int] = []
    seen = 0

    if verbose:
        print(f"  {'clip':22s} {'P':>4} {'R':>4} {'F1':>5}  {'IC err':>6}  "
              f"{'LESS MAE':>8}  detail")
        print("  " + "-" * 78)

    for lab in labels:
        clip = lab["clip"]
        jumps = out_dir / f"{clip}_jumps.csv"
        if not jumps.exists():
            if verbose:
                print(f"  {clip:22s}  -- no {jumps.name} (run --analyze first)")
            continue
        seen += 1
        pred = read_predicted(jumps)
        pred_ics = [ic for ic, _ in pred]
        pred_less = [ls for _, ls in pred]
        matches, un_p, un_t = match_landings(pred_ics, lab["frames"], tol)

        tp, fp, fn = len(matches), len(un_p), len(un_t)
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn
        p, r, f1 = prf(tp, fp, fn)

        ic_errs = [abs(pred_ics[pi] - lab["frames"][ti]) for pi, ti in matches]
        all_ic_err += ic_errs
        ic_err_mean = sum(ic_errs) / len(ic_errs) if ic_errs else None

        mae_pairs = [
            (pred_less[pi], lab["less"][ti])
            for pi, ti in matches
            if ti < len(lab["less"])
        ]
        if mae_pairs:
            all_pred_less += [a for a, _ in mae_pairs]
            all_true_less += [b for _, b in mae_pairs]
            mae = sum(abs(a - b) for a, b in mae_pairs) / len(mae_pairs)
        else:
            mae = None

        if verbose:
            detail = f"TP{tp} FP{fp} FN{fn}"
            if fp:
                detail += f" | extra@{[pred_ics[i] for i in un_p]}"
            if fn:
                detail += f" | missed@{[lab['frames'][i] for i in un_t]}"
            print(f"  {clip:22s} {p:4.2f} {r:4.2f} {f1:5.2f}  "
                  f"{_fmt(ic_err_mean, 1):>6}  {_fmt(mae):>8}  {detail}")

    if seen == 0:
        return None

    P, R, F1 = prf(tot_tp, tot_fp, tot_fn)
    agg = {
        "P": P, "R": R, "F1": F1, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "ic_err": sum(all_ic_err) / len(all_ic_err) if all_ic_err else None,
        "mae": (sum(abs(a - b) for a, b in zip(all_pred_less, all_true_less))
                / len(all_pred_less)) if all_pred_less else None,
        "bias": (sum(a - b for a, b in zip(all_pred_less, all_true_less))
                 / len(all_pred_less)) if all_pred_less else None,
        "r": pearson(all_pred_less, all_true_less),
        "icc": icc_2_1(all_pred_less, all_true_less),
        "n": len(all_pred_less),
        "fps": read_fps(out_dir, [lab["clip"] for lab in labels]),
    }
    return agg


# ------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "benchmark" / "labels.csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--tol", type=int, default=5,
                    help="IC frame tolerance for a detection to count as a match.")
    ap.add_argument("--variants", type=str, default=None,
                    help="Comma list of backbone variants to compare, e.g. "
                    "yolo_rtm,yolo_mediapipe,rfdetr_rtm,rfdetr_mediapipe. Each is read "
                    "from output/<variant>/ (yolo_rtm from the output root).")
    args = ap.parse_args(argv)

    labels = read_labels(args.labels)
    if not labels:
        print(f"No labelled clips in {args.labels}. Add rows and re-run.")
        return 1

    if args.variants:
        return _compare(labels, args)

    print(f"Evaluating {len(labels)} clip(s)  |  IC tolerance = +/-{args.tol} frames\n")
    agg = evaluate_dir(labels, args.output_dir, args.tol, verbose=True)
    print("  " + "-" * 78)
    if agg is None:
        print("\n  No clip outputs found. Run --analyze first.\n")
        return 1
    print("\n  === AGGREGATE ===")
    print(f"  landing detection : P={agg['P']:.2f}  R={agg['R']:.2f}  F1={agg['F1']:.2f}"
          f"   (TP={agg['tp']} FP={agg['fp']} FN={agg['fn']})")
    print(f"  IC frame error    : {_fmt(agg['ic_err'], 1)} frames mean |error|")
    print(f"  LESS agreement    : MAE={_fmt(agg['mae'])}  bias={_fmt(agg['bias'])}  "
          f"r={_fmt(agg['r'])}  ICC(2,1)={_fmt(agg['icc'])}  (n={agg['n']})")
    print()
    return 0


def _compare(labels: list[dict], args: argparse.Namespace) -> int:
    """Print one aggregate row per backbone variant for side-by-side comparison."""
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"Backbone comparison over {len(labels)} clip(s)  |  "
          f"IC tolerance = +/-{args.tol} frames\n")
    head = (f"  {'variant':20s} {'F1':>5} {'P':>5} {'R':>5}  {'IC err':>6}  "
            f"{'MAE':>5} {'bias':>5} {'r':>5} {'ICC':>5}  {'FPS':>6}  {'n':>3}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    any_row = False
    for v in variants:
        agg = evaluate_dir(labels, variant_dir(args.output_dir, v), args.tol,
                           verbose=False)
        if agg is None:
            print(f"  {v:20s}  -- no outputs in {variant_dir(args.output_dir, v)}")
            continue
        any_row = True
        print(f"  {v:20s} {agg['F1']:5.2f} {agg['P']:5.2f} {agg['R']:5.2f}  "
              f"{_fmt(agg['ic_err'], 1):>6}  {_fmt(agg['mae']):>5} "
              f"{_fmt(agg['bias']):>5} {_fmt(agg['r']):>5} {_fmt(agg['icc']):>5}  "
              f"{_fmt(agg['fps'], 1):>6}  {agg['n']:>3}")
    print()
    if not any_row:
        print("  No variant had outputs. Run --analyze with --detector/--pose first.\n")
        return 1
    print("  F1/P/R = landing detection;  IC err = mean |frame error| on matches;\n"
          "  MAE/bias/r/ICC = LESS agreement vs labels;  FPS = analyse throughput.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

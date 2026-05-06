# =========================================================================
# Hyperparameter tuning for Brain, Drain, LenMa, SHISO, Spell
# on the Welkin and Welkin_v2 datasets.
#
# To avoid overfitting hyperparameters to the test set the script
# operates on a configurable prefix of each log file (TUNE_ROWS).
# The parser sees only those rows; evaluation is also done only on
# those rows against the matching slice of the ground-truth CSV.
# The remaining rows are NOT touched here – they remain as the
# unseen test set when you later run benchmarking.py.
#
# With TUNE_ROWS = None the full log file is used (same behaviour
# as benchmarking.py – useful if you just want to do a parameter
# sweep without caring about a held-out test split).
#
# Usage:
#   python3 hyperparameter_tuning.py
# =========================================================================
from __future__ import annotations

import importlib
import inspect
import itertools
import os
import shutil
import sys
import tempfile
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from logparser.logparser.utils import evaluator  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARSERS = ["Brain", "Drain", "LenMa", "Logram", "SHISO", "Spell"]
DATASETS = ["Welkin_v2"]

INPUT_DIR_ROOT = "logparser/data/loghub_2k/"
OUTPUT_CSV = "hyperparameter_results.csv"

# How many rows of each log file to use for hyperparameter tuning.
# Set to None to use every row (no held-out test split).
#
# Recommendation: set to half the total (e.g. 1000 for a 2000-row dataset).
# The first TUNE_ROWS lines are used; the remainder stay unseen until you
# run benchmarking.py on the full file.
TUNE_ROWS = 500   # ← change to None to disable splitting

# ---------------------------------------------------------------------------
# Parameter grids – key tunable params for each parser
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "Brain": {
        # threshold: min distinct tokens at a position to become a wildcard
        "threshold": [2, 3, 4, 5, 6, 7, 8],
    },
    "Drain": {
        # st   : similarity threshold for merging into an existing cluster
        # depth: depth of the parse-tree
        "st":    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "depth": [3, 4, 5, 6],
    },
    "LenMa": {
        # threshold: cosine-similarity threshold for token-length vectors
        "threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "SHISO": {
        # mergeThreshold: merge two clusters when similarity >= this value
        "mergeThreshold": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    },
    "Logram": {
        # doubleThreshold: min bigram frequency to be treated as a constant
        # triThreshold   : min trigram frequency to be treated as a constant
        "doubleThreshold": [10, 15, 20, 30, 40, 50, 75, 90, 100],
        "triThreshold":    [2, 5, 10, 15, 20, 25],
    },
    "Spell": {
        # tau: minimum LCS ratio for a log line to match a template
        "tau": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
}


# ---------------------------------------------------------------------------
# Log-slice helpers
# ---------------------------------------------------------------------------

def slice_log_file(src_log: str, n_rows: int, dst_log: str) -> int:
    """
    Write the first n_rows *content* lines of src_log to dst_log.
    Returns the number of lines actually written.
    Lines that cannot be parsed by the log format are kept as-is
    (the parser itself will skip them).
    """
    with open(src_log, "r", encoding="utf-8", errors="replace") as fin:
        lines = fin.readlines()
    subset = lines[:n_rows]
    with open(dst_log, "w", encoding="utf-8") as fout:
        fout.writelines(subset)
    return len(subset)


def slice_groundtruth(gt_csv: str, n_rows: int, dst_csv: str) -> None:
    """
    Write the first n_rows data rows of the ground-truth structured CSV
    to dst_csv (header + n_rows rows).
    """
    df = pd.read_csv(gt_csv)
    df.head(n_rows).to_csv(dst_csv, index=False)


# ---------------------------------------------------------------------------
# Parser / settings loader
# ---------------------------------------------------------------------------

def load_parser_and_settings(parser_name: str):
    """Return (LogParser class, benchmark_settings dict)."""
    mod = importlib.import_module(f"logparser.logparser.{parser_name}")
    LogParser = mod.LogParser
    settings_mod = importlib.import_module(
        f"logparser.logparser.{parser_name}.settings"
    )
    settings = settings_mod.benchmark_settings
    return LogParser, settings


def build_param_combinations(parser_name: str):
    """Return list of dicts, each one hyperparameter combination."""
    grid = PARAM_GRIDS[parser_name]
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, vals)))
    return combos


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_one(LogParser, setting: dict, dataset: str,
            extra_params: dict, outdir: str,
            tune_rows: int | None = None) -> dict | None:
    """
    Instantiate the parser with setting + extra_params, parse the log
    (optionally only the first tune_rows lines), evaluate against the
    matching ground-truth slice, and return {'accuracy', 'f1'} or None.
    """
    src_indir  = os.path.join(INPUT_DIR_ROOT, os.path.dirname(setting["log_file"]))
    log_fname  = os.path.basename(setting["log_file"])
    src_log    = os.path.join(src_indir, log_fname)

    # Ground-truth CSV (full)
    gt_candidates = [
        os.path.join(src_indir, log_fname + "_structured.csv"),
        os.path.join(src_indir, os.path.splitext(log_fname)[0] + "_structured.csv"),
    ]
    gt_full = next((p for p in gt_candidates if os.path.exists(p)), None)
    if gt_full is None:
        print(f"FAILED (ground-truth not found in {src_indir})")
        return None

    # If tune_rows requested, create sliced copies in outdir
    if tune_rows is not None and tune_rows > 0:
        sliced_log = os.path.join(outdir, log_fname)
        actual_rows = slice_log_file(src_log, tune_rows, sliced_log)
        sliced_gt = os.path.join(outdir, log_fname + "_gt_slice.csv")
        # The ground-truth has one row per parsed log line, but the CSV may
        # have fewer rows than the log if some lines were unparseable.
        # Use actual_rows as an upper bound.
        slice_groundtruth(gt_full, actual_rows, sliced_gt)
        run_indir = outdir
        gt_to_use = sliced_gt
    else:
        run_indir = src_indir
        gt_to_use = gt_full

    # Build constructor params
    all_params = {
        "log_format":     setting["log_format"],
        "indir":          run_indir,
        "outdir":         outdir,
        "rex":            setting.get("regex", []),
        # Brain
        "logname":        dataset,
        "threshold":      setting.get("theshold", setting.get("threshold", 2)),
        "delimeter":      setting.get("delimiter", []),
        # Drain
        "st":             setting.get("st", 0.5),
        "depth":          setting.get("depth", 4),
        # SHISO
        "mergeThreshold": setting.get("mergeThreshold", 0.5),
        # Logram
        "doubleThreshold": setting.get("doubleThreshold", 15),
        "triThreshold":    setting.get("triThreshold", 10),
        # Spell
        "tau":            setting.get("tau", 0.5),
    }
    all_params.update(extra_params)

    sig = inspect.signature(LogParser.__init__)
    accepted = {
        k: v for k, v in all_params.items()
        if k in sig.parameters and k != "self"
    }

    # Parse
    try:
        parser = LogParser(**accepted)
        parser.parse(log_fname)
    except Exception as exc:
        print(f"FAILED (parse: {exc})")
        return None

    structured_out = os.path.join(outdir, log_fname + "_structured.csv")
    if not os.path.exists(structured_out):
        print("FAILED (no structured output)")
        return None

    # Align row counts: the ground-truth slice must match the parsed output
    try:
        df_parsed = pd.read_csv(structured_out)
        df_gt     = pd.read_csv(gt_to_use)
        n = min(len(df_parsed), len(df_gt))
        if n == 0:
            print("FAILED (0 matched rows)")
            return None
        # Write aligned slices to temp files for evaluator
        aligned_parsed = os.path.join(outdir, "_aligned_parsed.csv")
        aligned_gt     = os.path.join(outdir, "_aligned_gt.csv")
        df_parsed.head(n).to_csv(aligned_parsed, index=False)
        df_gt.head(n).to_csv(aligned_gt, index=False)
    except Exception as exc:
        print(f"FAILED (align: {exc})")
        return None

    # Evaluate
    try:
        f1, accuracy, precision, recall = evaluator.evaluate(
            groundtruth=aligned_gt,
            parsedresult=aligned_parsed,
        )
        return {"accuracy": round(float(accuracy), 4), "f1": round(float(f1), 4)}
    except Exception as exc:
        print(f"FAILED (eval: {exc})")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []

    if TUNE_ROWS is not None:
        print(f"\n[INFO] Hyperparameter tuning uses the first {TUNE_ROWS} rows of each log.")
        print(f"[INFO] The remaining rows are held out as the unseen test set.")
    else:
        print("\n[INFO] TUNE_ROWS=None – using the full log file (no held-out split).")

    for parser_name in PARSERS:
        print(f"\n{'='*60}")
        print(f"  Tuning: {parser_name}")
        print(f"{'='*60}")

        try:
            LogParser, all_settings = load_parser_and_settings(parser_name)
        except Exception as exc:
            print(f"  [ERROR] Could not load {parser_name}: {exc}")
            continue

        combos = build_param_combinations(parser_name)
        total  = len(combos) * len(DATASETS)
        print(f"  {len(combos)} param combo(s) × {len(DATASETS)} dataset(s) = {total} runs")

        for dataset in DATASETS:
            if dataset not in all_settings:
                print(f"\n  [SKIP] '{dataset}' not in {parser_name} settings")
                continue

            setting = all_settings[dataset]
            print(f"\n  --- {dataset} ---")

            for idx, combo in enumerate(combos, 1):
                combo_str = ", ".join(f"{k}={v}" for k, v in combo.items())
                print(f"    [{idx:3d}/{len(combos)}] {combo_str:35s}", end="  ", flush=True)

                tmpdir = tempfile.mkdtemp(prefix="hptune_")
                try:
                    metrics = run_one(
                        LogParser, setting, dataset, combo, tmpdir,
                        tune_rows=TUNE_ROWS,
                    )
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

                if metrics is None:
                    continue

                print(f"accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

                row = {
                    "parser":    parser_name,
                    "dataset":   dataset,
                    "tune_rows": TUNE_ROWS if TUNE_ROWS is not None else "all",
                    **combo,
                    "accuracy":  metrics["accuracy"],
                    "f1":        metrics["f1"],
                }
                rows.append(row)

    if not rows:
        print("\nNo results collected.")
        return

    df = pd.DataFrame(rows)

    # Mark best combo per (parser, dataset) by highest F1
    df["best"] = False
    for (pname, dname), grp in df.groupby(["parser", "dataset"]):
        if grp.empty:
            continue
        best_idx = grp["f1"].idxmax()
        df.loc[best_idx, "best"] = True

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"Results written to: {OUTPUT_CSV}  ({len(df)} rows)")

    # Summary
    param_cols = [c for c in df.columns
                  if c not in ("parser", "dataset", "tune_rows", "accuracy", "f1", "best")]
    print(f"\nBest setting per (parser, dataset) — ranked by F1"
          f"  [tuned on {TUNE_ROWS if TUNE_ROWS else 'all'} rows]:")
    print(f"{'Parser':<10} {'Dataset':<12} {'F1':>6}  {'Accuracy':>8}  Best params")
    print("-" * 72)
    best_df = df[df["best"]].sort_values(["parser", "dataset"])
    for _, row in best_df.iterrows():
        params = {k: row[k] for k in param_cols if pd.notna(row.get(k))}
        print(f"  {row['parser']:<8} {row['dataset']:<12} "
              f"{row['f1']:6.4f}  {row['accuracy']:8.4f}  {params}")


if __name__ == "__main__":
    main()
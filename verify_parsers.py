#!/usr/bin/env python3
"""
verify_parsers.py

Reads README files for each parser (Brain, Drain, LenMa, Logram, SHISO, Spell)
and extracts the published accuracy and F1 scores per dataset.

Reads actual benchmark results from Benchmark_results/_Parser_summary_stats/.

Produces:
  parser_verification.csv          – comparison table
  parser_verification_f1.png       – heatmap of diff_f1
  parser_verification_accuracy.png – heatmap of diff_accuracy
"""

import os
import re
import csv
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ONLINE_PARSERS = ["Brain", "Drain", "LenMa", "Logram", "SHISO", "Spell"]

README_TEMPLATE = os.path.join(
    BASE_DIR, "logparser", "logparser", "{parser}", "README.md"
)

SUMMARY_STATS_DIR = os.path.join(
    BASE_DIR, "Benchmark_results", "_Parser_summary_stats"
)

OUTPUT_CSV = os.path.join(BASE_DIR, "parser_verification.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_cell(s: str) -> str:
    """Strip markdown bold/italic formatting and whitespace from a table cell."""
    s = s.strip()
    s = re.sub(r"\*+", "", s)   # remove **bold** markers
    s = re.sub(r"_+", "", s)    # remove _italic_ markers
    return s.strip()


def _is_separator_row(cells):
    """Return True if this table row is the --- separator row."""
    return all(re.fullmatch(r"[-: ]+", c) for c in cells if c)


def _col_index(headers, *candidates):
    """
    Return the index of the first header that matches any candidate substring
    (case-insensitive).  Returns None if no match.
    """
    for i, h in enumerate(headers):
        hl = h.lower()
        for cand in candidates:
            if cand.lower() in hl:
                return i
    return None


def _diff(actual: str, readme: str) -> str:
    """
    Return the numeric difference (actual - readme) as a rounded float string,
    or '' if either value is missing or non-numeric.
    Positive  -> actual is higher than README claim.
    Negative  -> actual is lower  than README claim.
    """
    try:
        return f"{float(actual) - float(readme):.6f}"
    except (ValueError, TypeError):
        return ""


def _find_column(columns: list, *candidates):
    """
    Return the first column name (from `columns`) whose lowercase representation
    contains any of the candidate substrings.  Returns None if not found.
    """
    for col in columns:
        cl = col.lower().strip()
        for cand in candidates:
            if cand.lower() in cl:
                return col
    return None


# ---------------------------------------------------------------------------
# README parsing
# ---------------------------------------------------------------------------

def parse_readme(readme_path: str) -> dict:
    """
    Parse a logparser README.md file and extract the benchmark table that
    contains both F1 and Accuracy columns.

    Returns a dict:  { dataset_name: {'f1': str, 'accuracy': str} }
    """
    if not os.path.exists(readme_path):
        print(f"  [WARN] README not found: {readme_path}", file=sys.stderr)
        return {}

    with open(readme_path, encoding="utf-8") as fh:
        lines = fh.readlines()

    results = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        if "|" not in raw:
            i += 1
            continue

        # Potential table header row
        cells = [_strip_cell(c) for c in raw.strip().strip("|").split("|")]

        headers_lower = [c.lower() for c in cells]

        has_f1 = any("f1" in h for h in headers_lower)
        has_acc = any("accuracy" in h or "acc" in h for h in headers_lower)

        if not (has_f1 and has_acc):
            i += 1
            continue

        # We found a candidate header row -- locate column indices
        dataset_idx = _col_index(cells, "dataset", "log_type", "log", "data")
        if dataset_idx is None:
            dataset_idx = 0  # assume first column is dataset

        f1_idx = _col_index(cells, "f1")
        accuracy_idx = _col_index(cells, "accuracy", "acc")

        if f1_idx is None or accuracy_idx is None:
            i += 1
            continue

        # Skip separator row
        i += 1
        if i < len(lines) and "---" in lines[i]:
            i += 1

        # Read data rows until table ends
        while i < len(lines) and "|" in lines[i]:
            row_raw = lines[i].strip()
            row_cells = [_strip_cell(c) for c in row_raw.strip("|").split("|")]

            if _is_separator_row(row_cells):
                i += 1
                continue

            max_idx = max(dataset_idx, f1_idx, accuracy_idx)
            if len(row_cells) > max_idx:
                dataset = row_cells[dataset_idx]
                f1 = row_cells[f1_idx]
                accuracy = row_cells[accuracy_idx]

                if dataset and dataset not in results:
                    results[dataset] = {"f1": f1, "accuracy": accuracy}

            i += 1
        # Inner loop already advanced i past the table; outer loop continues.

    return results


# ---------------------------------------------------------------------------
# Actual results reader
# ---------------------------------------------------------------------------

def read_actual_results(parser: str) -> dict:
    """
    Read the _Parser_summary_stats CSV for the given parser.

    Returns a dict:  { dataset_name: {'f1': str, 'accuracy': str} }
    """
    csv_path = os.path.join(
        SUMMARY_STATS_DIR, f"{parser}_benchmark_result.csv"
    )
    if not os.path.exists(csv_path):
        print(f"  [WARN] Summary CSV not found: {csv_path}", file=sys.stderr)
        return {}

    results = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        stripped_cols = [c.strip() for c in columns]

        dataset_col = _find_column(stripped_cols, "dataset", "log_type", "log")
        if dataset_col is None:
            dataset_col = stripped_cols[0] if stripped_cols else None

        f1_col = _find_column(stripped_cols, "f1")
        accuracy_col = _find_column(stripped_cols, "accuracy", "groupacc", "ga")

        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items()}

            dataset = row.get(dataset_col, "").strip() if dataset_col else ""
            f1 = row.get(f1_col, "") if f1_col else ""
            accuracy = row.get(accuracy_col, "") if accuracy_col else ""

            if dataset:
                results[dataset] = {"f1": f1, "accuracy": accuracy}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_grid(rows, diff_col, parsers, datasets):
    """Build a (n_parsers x n_datasets) numpy array of diff values."""
    import numpy as np
    grid = np.full((len(parsers), len(datasets)), float("nan"))
    ds_index = {d: i for i, d in enumerate(datasets)}
    p_index  = {p: i for i, p in enumerate(parsers)}
    for r in rows:
        p = r["Parser"]
        d = r["dataset"]
        if p not in p_index or d not in ds_index:
            continue
        val = r.get(diff_col, "")
        try:
            grid[p_index[p], ds_index[d]] = float(val)
        except (ValueError, TypeError):
            pass
    return grid


def _save_heatmap(grid, parsers, datasets, title, output_path):
    """Save a single heatmap as its own PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed -- skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    vmax = max(0.01, float(np.nanmax(np.abs(grid))))
    vmax = min(vmax, 0.15)

    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    for pi in range(len(parsers)):
        for di in range(len(datasets)):
            val = grid[pi, di]
            if not np.isnan(val):
                txt = f"{val:+.3f}"
                txt_color = "black" if abs(val) < vmax * 0.6 else "white"
                ax.text(di, pi, txt, ha="center", va="center",
                        fontsize=7, color=txt_color)
            else:
                ax.text(di, pi, "-", ha="center", va="center",
                        fontsize=8, color="#aaaaaa")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(parsers)))
    ax.set_yticklabels(parsers, fontsize=10)
    ax.set_title(
        title + "\nWhite = matches literature  |  Red = actual lower  |  Blue = actual higher",
        fontsize=11,
        pad=8,
    )
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved   : {output_path}")


def plot_comparison(rows: list) -> None:
    """
    Save two separate heatmap PNG files:
      parser_verification_f1.png        -- diff_f1  (actual - readme)
      parser_verification_accuracy.png  -- diff_accuracy (actual - readme)

    Rows = parsers, columns = datasets that appear in at least one README.
    """
    try:
        import numpy as np
    except ImportError:
        print("[WARN] numpy not installed -- skipping plot.", file=sys.stderr)
        return

    parsers  = ONLINE_PARSERS
    datasets = sorted({
        r["dataset"] for r in rows
        if r.get("f1-readme") or r.get("accuracy-readme")
    })

    grid_f1  = _build_grid(rows, "diff_f1",       parsers, datasets)
    grid_acc = _build_grid(rows, "diff_accuracy",  parsers, datasets)

    _save_heatmap(
        grid_f1, parsers, datasets,
        "",
        os.path.join(BASE_DIR, "parser_verification_f1.png"),
    )
    _save_heatmap(
        grid_acc, parsers, datasets,
        "",
        os.path.join(BASE_DIR, "parser_verification_accuracy.png"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Base directory : {BASE_DIR}")
    print(f"Summary stats  : {SUMMARY_STATS_DIR}")
    print(f"Output CSV     : {OUTPUT_CSV}\n")

    output_rows = []

    for parser in ONLINE_PARSERS:
        print(f"=== {parser} ===")

        readme_path = README_TEMPLATE.format(parser=parser)
        readme_data = parse_readme(readme_path)
        print(f"  README datasets  ({len(readme_data)}): {sorted(readme_data.keys())}")

        actual_data = read_actual_results(parser)
        print(f"  Actual datasets  ({len(actual_data)}): {sorted(actual_data.keys())}")

        # Union of all known datasets for this parser
        all_datasets = sorted(set(readme_data.keys()) | set(actual_data.keys()))

        for dataset in all_datasets:
            readme_entry = readme_data.get(dataset, {})
            actual_entry = actual_data.get(dataset, {})

            f1_r  = readme_entry.get("f1", "")
            acc_r = readme_entry.get("accuracy", "")
            f1_a  = actual_entry.get("f1", "")
            acc_a = actual_entry.get("accuracy", "")

            output_rows.append(
                {
                    "Parser":          parser,
                    "dataset":         dataset,
                    "f1-readme":       f1_r,
                    "accuracy-readme": acc_r,
                    "f1-actual":       f1_a,
                    "accuracy-actual": acc_a,
                    "diff_f1":         _diff(f1_a, f1_r),
                    "diff_accuracy":   _diff(acc_a, acc_r),
                }
            )

        print(f"  Combined rows    : {len(all_datasets)}")
        print()

    # Write output CSV
    fieldnames = [
        "Parser",
        "dataset",
        "f1-readme",
        "accuracy-readme",
        "f1-actual",
        "accuracy-actual",
        "diff_f1",
        "diff_accuracy",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Written {len(output_rows)} rows to {OUTPUT_CSV}")

    # Generate separate plots
    plot_comparison(output_rows)

    # Print preview
    w = [10, 15, 12, 12, 12, 12, 12, 12]
    hdr = (
        f"{'Parser':<{w[0]}} {'Dataset':<{w[1]}} {'F1-readme':>{w[2]}}"
        f" {'Acc-readme':>{w[3]}} {'F1-actual':>{w[4]}} {'Acc-actual':>{w[5]}}"
        f" {'diff_f1':>{w[6]}} {'diff_acc':>{w[7]}}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for row in output_rows:
        print(
            f"{row['Parser']:<{w[0]}} {row['dataset']:<{w[1]}}"
            f" {str(row['f1-readme']):>{w[2]}} {str(row['accuracy-readme']):>{w[3]}}"
            f" {str(row['f1-actual']):>{w[4]}} {str(row['accuracy-actual']):>{w[5]}}"
            f" {str(row['diff_f1']):>{w[6]}} {str(row['diff_accuracy']):>{w[7]}}"
        )


if __name__ == "__main__":
    main()
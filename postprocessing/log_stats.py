"""
Scans all ground-truth structured CSV files under loghub_2k/ and produces
a LaTeX table snippet describing each dataset.

Columns:
  - Dataset name
  - Average Content length (characters)
  - Vocabulary size (unique whitespace-split tokens in Content)
  - Total key=value pairs found across all Content columns
  - Number of lines that contain at least one key=value pair
  - Number of unique EventTemplate values
  - Average number of wildcards (<*>) per EventTemplate line

Usage:
    uv run python postprocessing/log_stats.py
    # Output written to postprocessing/dataset_stats_table.tex
"""
import re
import os
import glob
import sys
import pandas as pd

# Ensure the repo root is on sys.path so we can import from logparser
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Use the SAME regex and counting logic as the evaluator/benchmarking code.
from logparser.logparser.utils.evaluator import _KV_PAIR_RE, _count_kv_lines

# Root where the ground-truth CSVs live
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "logparser", "data", "loghub_2k")
DATA_ROOT = os.path.abspath(DATA_ROOT)

# Regex matching a single wildcard token in a template
_WILDCARD_RE = re.compile(r'<[^>]+>')

# Only these core loghub_2k datasets (matching the benchmarks).
# Excludes e.g. Welkin_old, Welkin_synthetic, Welkin_v2_old.
DATASETS = [
    "Android", "Apache", "BGL", "Hadoop", "HDFS", "HealthApp", "HPC",
    "Linux", "Mac", "OpenSSH", "OpenStack", "Proxifier", "Spark",
    "Thunderbird", "Welkin", "Windows", "Zookeeper",
]


def extract_name(filepath):
    """Extract a clean dataset name from a structured CSV path."""
    parent = os.path.basename(os.path.dirname(filepath))
    return parent


def compute_stats(filepath):
    """Return a dict of stats for one structured CSV file."""
    df = pd.read_csv(filepath)

    total_lines = len(df)

    # Average length of the Content column
    if "Content" in df.columns:
        content_lengths = df["Content"].dropna().astype(str).str.len()
        avg_line_length = round(content_lengths.mean(), 1) if len(content_lengths) > 0 else 0.0
    else:
        avg_line_length = 0.0

    # Count key=value pairs and lines containing them
    # Use _count_kv_lines from evaluator.py for the line counting (identical logic to benchmarking.py)
    total_kv_pairs = 0
    kv_lines = 0

    if "Content" in df.columns:
        kv_lines = _count_kv_lines(df["Content"])
        # Sum total KV pairs (not just lines) — same regex, counts all occurrences
        total_kv_pairs = df["Content"].dropna().astype(str).apply(
            lambda x: len(_KV_PAIR_RE.findall(x))
        ).sum()

    # Unique Event Templates
    if "EventTemplate" in df.columns:
        num_templates = df["EventTemplate"].nunique()
    else:
        num_templates = 0

    # Vocabulary size: unique alphabetic words in Content (tokens containing
    # at least one letter). This excludes pure numbers, timestamps, IPs, etc.
    if "Content" in df.columns:
        all_words = set()
        for cell in df["Content"].dropna().astype(str):
            for token in cell.split():
                if re.search(r'[a-zA-Z]', token):
                    all_words.add(token)
        vocab_size = len(all_words)
    else:
        vocab_size = 0

    # Average wildcards per line (in the EventTemplate column)
    if "EventTemplate" in df.columns:
        wc_counts = df["EventTemplate"].dropna().astype(str).apply(
            lambda t: len(_WILDCARD_RE.findall(t))
        )
        avg_wildcards = round(wc_counts.mean(), 2) if len(wc_counts) > 0 else 0.0
    else:
        avg_wildcards = 0.0

    return {
        "name": extract_name(filepath),
        "total_lines": total_lines,
        "avg_line_length": avg_line_length,
        "vocab_size": vocab_size,
        "total_kv_pairs": total_kv_pairs,
        "kv_lines": kv_lines,
        "num_templates": num_templates,
        "avg_wildcards": avg_wildcards,
    }


def format_number(n):
    """Format an integer with LaTeX thousands separator."""
    s = f"{n:,}"
    return s.replace(",", "{,}")


def main():
    pattern = os.path.join(DATA_ROOT, "**", "*.log_structured.csv")
    files = glob.glob(pattern, recursive=True)

    # Filter out custom/detail files
    files = [
        f for f in files
        if "_custom_structured" not in f and "_detailed_comparison" not in f
    ]

    stats_list = []
    for fp in sorted(files):
        try:
            stats = compute_stats(fp)
            stats_list.append(stats)
            print(f"  OK  {stats['name']:20s}  {stats['total_lines']} lines, "
                  f"{stats['num_templates']} templates, "
                  f"{stats['kv_lines']} KV-lines, "
                  f"{stats['avg_wildcards']} avg wildcards")
        except Exception as e:
            print(f"  ERR {os.path.basename(fp):20s}  {e}")

    # Filter to only the core datasets and sort by preferred order
    stats_list = [s for s in stats_list if s["name"] in DATASETS]
    stats_list.sort(key=lambda s: DATASETS.index(s["name"]))

    # ---- Compute per-column maxima for underlining ----
    max_vals = {
        "avg_line_length": max(s["avg_line_length"] for s in stats_list),
        "vocab_size":      max(s["vocab_size"] for s in stats_list),
        "total_kv_pairs":  max(s["total_kv_pairs"] for s in stats_list),
        "kv_lines":        max(s["kv_lines"] for s in stats_list),
        "num_templates":   max(s["num_templates"] for s in stats_list),
        "avg_wildcards":   max(s["avg_wildcards"] for s in stats_list),
    }

    # ---- Generate LaTeX output ----
    def fmt_cell(raw_val, col_key):
        """Format a single cell: underline if max, otherwise plain."""
        if col_key == "avg_wildcards":
            val_str = f"{raw_val:.2f}"
        elif col_key == "avg_line_length":
            val_str = f"{raw_val:.1f}"
        elif col_key == "num_templates":
            val_str = str(raw_val)
        else:
            val_str = format_number(raw_val)
        if raw_val == max_vals[col_key]:
            val_str = f"\\underline{{{val_str}}}"
        return val_str

    lines_out = []
    lines_out.append("% Auto-generated by postprocessing/log_stats.py")
    lines_out.append("% Datasets from loghub_2k ground-truth CSVs")
    lines_out.append(r"\begin{tabular}{lrrrrrr}")
    lines_out.append(r"\toprule")
    line = (
        r"Dataset & "
        r"\makecell{Avg.\\length} & "
        r"\makecell{Vocab.\\size} & "
        r"\makecell{KV\\pairs} & "
        r"\makecell{KV\\lines} & "
        r"\makecell{Templates} & "
        r"\makecell{Avg.\\wildcards} \\"
    )
    lines_out.append(line)
    lines_out.append(r"\midrule")

    cols = ["avg_line_length", "vocab_size", "total_kv_pairs", "kv_lines", "num_templates", "avg_wildcards"]

    for s in stats_list:
        cells = [fmt_cell(s[col], col) for col in cols]
        if s["name"] == "Welkin":
            row = "\\textbf{Welkin} & " + " & ".join(cells) + " \\\\"
        else:
            row = s["name"] + " & " + " & ".join(cells) + " \\\\"
        lines_out.append(row)

    lines_out.append(r"\bottomrule")
    lines_out.append(r"\end{tabular}")

    # Write to output file
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "dataset_stats_table.tex")
    with open(output_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"\nLaTeX table written to: {output_path}")
    print(f"  ({len(stats_list)} datasets)")


if __name__ == "__main__":
    main()
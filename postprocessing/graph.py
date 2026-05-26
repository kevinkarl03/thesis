import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


def load_benchmark_data(input_dir):
    
    os.chdir(input_dir)
    all_files = glob.glob("*_benchmark_result.csv")
    dataframes = []

    for filename in all_files:
        parser_name = os.path.basename(filename).split('_')[0]
        df = pd.read_csv(filename)
        df['Parser'] = parser_name
        dataframes.append(df)

    df_all = pd.concat(dataframes, ignore_index=True)
    sns.set_theme(style="whitegrid")
    
    return df_all


def graph_heatmap(df_all, output_dir='../../postprocessing/plots'):
    
    pivot_df = df_all.pivot(index="Dataset", columns="Parser", values="Accuracy")
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('Accuracy Heatmap: Dataset vs Parser')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'))
    plt.close()

    pivot_df = df_all.pivot(index="Dataset", columns="Parser", values="F1_measure")
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('F1-Measure Heatmap: Dataset vs Parser')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'f1_heatmap.png'))
    plt.close()


def graph_k8s_comparison(df_all, output_dir='../../postprocessing/plots'):
    k8s_data = df_all[df_all['Dataset'].str.contains('Welkin', case=False)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=k8s_data, x='Dataset', y='Accuracy', hue='Parser')
    plt.title('Parsing Accuracy on Welkin')
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'welkin_accuracy.png'))
    plt.close()


def graph_overall_ranking(df_all, output_dir='../../postprocessing/plots'):
    
    avg_perf = df_all.groupby('Parser')[['Accuracy', 'F1_measure']].mean().reset_index()
    avg_melted = avg_perf.melt(id_vars='Parser', var_name='Metric', value_name='Score')

    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_melted, x='Parser', y='Score', hue='Metric')
    plt.title('Overall Average Performance (Loghub 2k)')
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'overall_ranking.png'))
    plt.close()

def graph_welkin_ranking(df_all, output_dir='../../postprocessing/plots'):
    
    welkin_data = df_all[df_all['Dataset'].str.contains('Welkin', case=False)]
    avg_perf = welkin_data.groupby('Parser')[['Accuracy', 'F1_measure']].mean().reset_index()
    avg_melted = avg_perf.melt(id_vars='Parser', var_name='Metric', value_name='Score')

    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_melted, x='Parser', y='Score', hue='Metric')
    plt.title('Parser Ranking on Welkin Dataset')
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'welkin_ranking.png'))
    plt.close()

def graph_drain_accuracy_delta(merged, output_dir='../../postprocessing/plots'):
    """Separate figure: Accuracy Δ in percentage points (key-value − base) per dataset."""
    datasets = merged['Dataset'].tolist()
    diffs_pp = (merged['Accuracy_diff'] * 100).tolist()  # convert to percentage points

    n_improved = sum(1 for d in diffs_pp if d > 0)
    n_same     = sum(1 for d in diffs_pp if d == 0)
    n_worse    = sum(1 for d in diffs_pp if d < 0)

    colors = ['#2ca02c' if v > 0 else '#d62728' if v < 0 else '#7f7f7f' for v in diffs_pp]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(datasets))
    ax.bar(x, diffs_pp, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy Δ (percentage points)', fontsize=11)

    for i, v in enumerate(diffs_pp):
        label = f'+{v:.2f}pp' if v > 0 else f'{v:.2f}pp'
        y_pos = v + (0.08 if v >= 0 else -0.25)
        ax.text(i, y_pos, label, ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=7, rotation=90)

    avg_pp = merged["Accuracy_diff"].mean() * 100
    summary = (f'Improved: {n_improved}   |   Unchanged: {n_same}   |   '
               f'Degraded: {n_worse}')
    ax.set_title(f'Accuracy Δ — Drain base vs key-value\n{summary}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drain_accuracy_delta.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved drain_accuracy_delta.png")


def graph_drain_f1_delta(merged, output_dir='../../postprocessing/plots'):
    """Separate figure: F1 Δ in percentage points (key-value − base) per dataset."""
    datasets = merged['Dataset'].tolist()
    diffs_pp = (merged['F1_diff'] * 100).tolist()  # convert to percentage points

    n_improved = sum(1 for d in diffs_pp if d > 0)
    n_same     = sum(1 for d in diffs_pp if d == 0)
    n_worse    = sum(1 for d in diffs_pp if d < 0)

    colors = ['#2ca02c' if v > 0 else '#d62728' if v < 0 else '#7f7f7f' for v in diffs_pp]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(datasets))
    ax.bar(x, diffs_pp, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('F1 Δ (percentage points)', fontsize=11)

    for i, v in enumerate(diffs_pp):
        label = f'+{v:.2f}pp' if v > 0 else f'{v:.2f}pp'
        y_pos = v + (0.08 if v >= 0 else -0.25)
        ax.text(i, y_pos, label, ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=7, rotation=90)

    avg_pp = merged["F1_diff"].mean() * 100
    summary = (f'Improved: {n_improved}   |   Unchanged: {n_same}   |   '
               f'Degraded: {n_worse}')
    ax.set_title(f'F1-measure Δ — Drain base vs key-value\n{summary}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drain_f1_delta.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved drain_f1_delta.png")


def graph_drain_kv_pct(merged, output_dir='../../postprocessing/plots'):
    """
    Horizontal bar chart showing the percentage of lines that contain
    key=value pairs per dataset. This shows which datasets actually have
    the potential to benefit from custom klog parsing.
    """
    merged_sorted = merged.sort_values('KV_pct_base', ascending=True).reset_index(drop=True)

    datasets = merged_sorted['Dataset'].tolist()
    kv_pcts  = merged_sorted['KV_pct_base'].tolist()

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = plt.cm.YlOrRd([p / 100 for p in kv_pcts])
    bars = ax.barh(datasets, kv_pcts, color=colors, edgecolor='white', linewidth=0.5)

    # Annotate percentage on each bar
    for bar, pct in zip(bars, kv_pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=9)

    ax.set_xlabel('Lines with key=value pairs (%)', fontsize=11)
    ax.set_title('Which datasets have key=value content?',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drain_kv_pct.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved drain_kv_pct.png")

    # Console summary
    print(f"\n  Datasets with KV content (KV_pct > 0):")
    for d, p in zip(datasets, kv_pcts):
        if p > 0:
            print(f"    {d:20s}  {p}%")
    print(f"  Datasets with no KV content: {sum(1 for p in kv_pcts if p == 0)}")


def graph_drain_kv_vs_improvement(merged, output_dir='../../postprocessing/plots'):
    """
    Scatter plot: KV% on x-axis vs Accuracy Δ (percentage points) on y-axis.
    Each point is a dataset, annotated with its name.
    Dashed vertical line separates datasets with zero KV lines (left) from those with KV content (right).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    x = merged['KV_pct_base'].values
    y = merged['Accuracy_diff'].values
    y_pp = y * 100  # convert to percentage points
    labels = merged['Dataset'].tolist()

    # Color: improved (green) vs unchanged (gray)
    colors = ['#2ca02c' if d > 0 else '#7f7f7f' for d in y]
    sizes  = [120 if d > 0 else 80 for d in y]

    ax.scatter(x, y_pp, c=colors, s=sizes, edgecolors='black', linewidth=0.5, zorder=5)

    # Annotate each point
    for xi, yi_pp, label in zip(x, y_pp, labels):
        offset_y = 0.05 if yi_pp >= 0 else -0.05
        va = 'bottom' if yi_pp >= 0 else 'top'
        ax.annotate(label, (xi, yi_pp), textcoords='offset points',
                    xytext=(4, offset_y if yi_pp >= 0 else -offset_y),
                    fontsize=8, ha='left', va=va)

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)

    ax.set_xlabel('Lines with key=value pairs (%)', fontsize=12)
    ax.set_ylabel('Accuracy Δ (percentage points)', fontsize=12)
    ax.set_title('Does key-value content enable accuracy improvements?',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # Summary annotation
    improved = sum(1 for d in y if d > 0)
    with_kv = sum(1 for xv in x if xv > 0)
    text = (f'Datasets with KV content: {with_kv}/{len(merged)}\n'
            f'Improved: {improved}/{with_kv}  (of those with KV content)')
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drain_kv_vs_improvement.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved drain_kv_vs_improvement.png")


def graph_drain_keyvalue_comparison(df_all, output_dir='../../postprocessing/plots'):
    """Generate all plots for Drain key-value comparison."""
    drain_data = df_all[df_all['Parser'] == 'Drain'].copy()

    base_data = drain_data[~drain_data['Dataset'].str.endswith('-key-value')].copy()
    kv_data   = drain_data[drain_data['Dataset'].str.endswith('-key-value')].copy()
    kv_data['BaseDataset'] = kv_data['Dataset'].str.replace('-key-value', '', regex=False)

    merged = base_data[['Dataset', 'Accuracy', 'F1_measure', 'KV_pct']].rename(
        columns={'Accuracy': 'Accuracy_base', 'F1_measure': 'F1_base', 'KV_pct': 'KV_pct_base'}
    ).merge(
        kv_data[['BaseDataset', 'Accuracy', 'F1_measure']].rename(
            columns={'Accuracy': 'Accuracy_kv', 'F1_measure': 'F1_kv'}
        ),
        left_on='Dataset',
        right_on='BaseDataset',
        how='inner'
    )

    merged['Accuracy_diff'] = merged['Accuracy_kv'] - merged['Accuracy_base']
    merged['F1_diff']       = merged['F1_kv']       - merged['F1_base']
    merged = merged.sort_values('Accuracy_diff', ascending=False).reset_index(drop=True)

    graph_drain_accuracy_delta(merged, output_dir)
    graph_drain_f1_delta(merged, output_dir)
    graph_drain_kv_pct(merged, output_dir)
    graph_drain_kv_vs_improvement(merged, output_dir)

    # Console summary
    print(f"\n  Average accuracy Δ: {merged['Accuracy_diff'].mean():+.6f}")
    print(f"  Average F1      Δ: {merged['F1_diff'].mean():+.6f}")
    n_imp = sum(1 for d in merged['Accuracy_diff'] if d > 0)
    n_same = sum(1 for d in merged['Accuracy_diff'] if d == 0)
    print(f"  Improved: {n_imp}/{len(merged)},  Unchanged: {n_same}/{len(merged)}")


if __name__ == "__main__":
    input_directory = "Benchmark_results/_Parser_summary_stats"
    output_directory = "../../postprocessing/plots"
    
    # Load benchmark data
    df_all = load_benchmark_data(input_directory)
    
    # Generate all graphs
    graph_heatmap(df_all, output_directory)
    
    graph_k8s_comparison(df_all, output_directory)
    
    graph_overall_ranking(df_all, output_directory)

    graph_welkin_ranking(df_all, output_directory)

    graph_drain_keyvalue_comparison(df_all, output_directory)

    print("="*50)
    print("== All Graphs created ==")
    print("="*50)
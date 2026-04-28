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
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'))
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




if __name__ == "__main__":
    input_directory = "Benchmark_results/_Parser_summary_stats"
    output_directory = "../../postprocessing/plots"
    
    # Load benchmark data
    df_all = load_benchmark_data(input_directory)
    
    # Generate all graphs
    graph_heatmap(df_all, output_directory)
    
    graph_k8s_comparison(df_all, output_directory)
    
    graph_overall_ranking(df_all, output_directory)
    print("="*50)
    print("== All Grapsh created ==")
    print("="*50)

    
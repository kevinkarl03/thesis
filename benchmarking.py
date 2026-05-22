# =========================================================================
# Main file for running evaluation that is done in this study, this will 
# be able to run some selected parsers on all datasets, making for easy 
# comparison of accuracy.
# =========================================================================
import re
import sys
import importlib
import os
import pandas as pd
import inspect
from logparser.logparser.utils import evaluator

# When True, skip all "-key-value" dataset variants so only the base
# (non-key-value) datasets are benchmarked.
SKIP_KEY_VALUE_VARIANTS = True

# Regex matching a key=value pair used by the custom klog preprocessor.
# Matches both quoted (key="value") and unquoted (key=value) forms.
_KV_PAIR_RE = re.compile(r'\w+="[^"]*"|\w+=[a-zA-Z0-9_./-]+')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_parser_assets(name):
    try:
        parser_mod = importlib.import_module(f"logparser.logparser.{name}")
        parser_class = getattr(parser_mod, "LogParser")
        
        settings_mod = importlib.import_module(f"logparser.logparser.{name}.settings")
        settings = getattr(settings_mod, "benchmark_settings")  

        return parser_class, settings
    except Exception as e:
        print(f"Error loading assets for {name}: {e}")
        return None, None


def run_parser(parser_name):
    # This runs the entire benchmark on all dataset with the specified parser

    print(f"\n" + "="*50)
    print(f" STARTING BENCHMARK FOR: {parser_name}")
    print("="*50)

    input_dir_root = "logparser/data/loghub_2k/"
    BASE_OUT_DIR = "Benchmark_results/"
    PARSER_RESULTS_DIR = os.path.join(BASE_OUT_DIR, "_Parser_summary_stats")
    
    LogParser, benchmark_settings = load_parser_assets(parser_name)
    if not LogParser:
        return

    output_dir = os.path.join(BASE_OUT_DIR, f"{parser_name}_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(PARSER_RESULTS_DIR):            
        os.makedirs(PARSER_RESULTS_DIR)

    benchmark_result = []

    for dataset, setting in benchmark_settings.items():
        # Skip key-value variants when the flag is enabled
        if SKIP_KEY_VALUE_VARIANTS and dataset.endswith("-key-value"):
            print(f"  Skipping {dataset} (key-value variant disabled)")
            continue

        print(f"\n--- Processing {dataset} ---")

        indir = os.path.join(input_dir_root, os.path.dirname(setting["log_file"]))
        log_file = os.path.basename(setting["log_file"])

        all_potential_params = {
            "log_format": setting["log_format"],
            "indir": indir,
            "outdir": output_dir,
            "rex": setting["regex"],
            "logname": dataset
        }
        
        # Extra fields (st, depth, threshold, delimiter...)
        extra_fields = {k: v for k, v in setting.items() if k not in ['log_file', 'log_format', 'regex']}
        all_potential_params.update(extra_fields)

        signature = inspect.signature(LogParser.__init__)
        valid_params = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self': continue
            if param_name in all_potential_params:
                valid_params[param_name] = all_potential_params[param_name]
            elif param_name == 'logname':
                valid_params['logname'] = dataset

        try:
            parser_instance = LogParser(**valid_params)
            parser_instance.parse(log_file)

            # Evaluation
            groundtruth = os.path.join(indir, log_file + "_structured.csv")

            if parser_instance.custom_klog_parser:
                parsed_result = os.path.join(output_dir, log_file + "_custom_structured.csv")
            else:
                parsed_result = os.path.join(output_dir, log_file + "_structured.csv")

            f1_measure, accuracy, precision, recall = evaluator.evaluate(
                groundtruth=groundtruth,
                parsedresult=parsed_result,
            )

            # Count lines containing key=value pairs in the Content column
            groundtruth_df = pd.read_csv(groundtruth)
            gt_kv_count = groundtruth_df["Content"].dropna().apply(
                lambda x: bool(_KV_PAIR_RE.search(str(x)))
            ).sum()
            gt_total = len(groundtruth_df)
            gt_kv_pct = round(100 * gt_kv_count / gt_total, 1) if gt_total > 0 else 0.0

            benchmark_result.append([
                dataset, f1_measure, accuracy, precision, recall,
                gt_total, gt_kv_count, gt_kv_pct,
            ])
            
            # Generate detailed per-line comparison
            print(f"\nGenerating detailed comparison for {dataset}...")
            try:
                groundtruth_df = pd.read_csv(groundtruth)
                parsed_df = pd.read_csv(parsed_result)
                
                # Merge on LineId to compare templates
                if 'LineId' in groundtruth_df.columns and 'LineId' in parsed_df.columns:
                    comparison_df = pd.merge(
                        groundtruth_df[['LineId', 'Content', 'EventTemplate']],
                        parsed_df[['LineId', 'EventTemplate']],
                        on='LineId',
                        suffixes=('_groundtruth', '_parsed')
                    )
                    
                    # Normalize templates by stripping surrounding quotes for fair comparison
                    def normalize_template(t):
                        if isinstance(t, str):
                            # Strip leading/trailing quotes that are CSV artifacts
                            return t.strip('"').strip("'")
                        return t
                    
                    groundtruth_norm = comparison_df['EventTemplate_groundtruth'].apply(normalize_template)
                    parsed_norm = comparison_df['EventTemplate_parsed'].apply(normalize_template)
                    comparison_df['Correct'] = groundtruth_norm == parsed_norm
                    
                    # Save detailed comparison
                    detailed_file = os.path.join(output_dir, log_file + "_detailed_comparison.csv")
                    comparison_df.to_csv(detailed_file, index=False)
                    print(f"Detailed comparison saved to: {detailed_file}")
                    print(f"Total lines: {len(comparison_df)}")
                    print(f"Correct matches: {comparison_df['Correct'].sum()}")
                    print(f"Incorrect matches: {(~comparison_df['Correct']).sum()}")
                    
                    # Show some examples of incorrect matches
                    incorrect = comparison_df[~comparison_df['Correct']]
                    # if len(incorrect) > 0:
                    #     print(f"\nFirst 10  incorrect matches:")
                    #     for _, row in incorrect.head(10).iterrows():
                    #         print(f"\n  Line {row['LineId']}:")
                    #         print(f"    Content: {row['Content'][:100]}...")
                    #         print(f"    Expected: {row['EventTemplate_groundtruth']}")
                    #         print(f"    Got:      {row['EventTemplate_parsed']}")
            except Exception as e:
                print(f"Warning: Could not generate detailed comparison: {e}")
            
        except Exception as e:
            print(f"Failed to parse {dataset}: {e}")

    df_result = pd.DataFrame(benchmark_result, columns=[
        "Dataset", "F1_measure", "Accuracy", "Precision", "Recall",
        "Total_lines", "KV_lines", "KV_pct",
    ])
    print(f"\nResults for {parser_name}:")
    print(df_result)
    
    summary_file = os.path.join(PARSER_RESULTS_DIR, f"{parser_name}_benchmark_result.csv")
    df_result.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")



if __name__ == "__main__":


    # parsers_to_run = ["AEL", "Brain", "Drain", "IPLoM", "LenMa", "LFA", "LKE", "LogCluster", "LogMine", "Logram", "LogSig", "MoLFI", "SHISO", "SLCT", "Spell", "ULP"]

    # Simpler parsers that i can run out of the box
    # parsers_to_run = ["AEL", "Brain", "Drain", "IPLoM", "LenMa", "LFA", "LogCluster", "LogMine", "Logram", "LogSig", "MoLFI", "SHISO", "SLCT", "Spell"]
    online_parsers = ["Brain", "Drain", "LenMa", "Logram", "SHISO", "Spell"]
    # drain = ["Drain"]
    

    for name in online_parsers:
        run_parser(name)
    
    print("\n" + "#"*50)
    print(" ALL BENCHMARKS COMPLETED!")
    print("#"*50)
# =========================================================================
# Main file for running evaluation that is done in this study, this will 
# be able to run some selected parsers on all datasets, making for easy 
# comparison of accuracy.
# =========================================================================


import sys
import importlib
from logparser.logparser.utils import evaluator
import os
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))




parser_name = "Drain" 
input_dir = "logparser/data/loghub_2k/"
BASE_OUT_DIR = "Benchmark_results/"
PARSER_RESULTS_DIR = BASE_OUT_DIR + "_Parser_results"




def load_parser_assets(name):
    try:
        parser_mod = importlib.import_module(f"logparser.logparser.{name}")
        parser_class = getattr(parser_mod, "LogParser")
        
        settings_mod = importlib.import_module(f"logparser.logparser.{name}.settings")
        settings = getattr(settings_mod, "benchmark_settings")  

        return parser_class, settings
    except Exception as e:
        print(f"Error loading assets for {name}: {e}")
        sys.exit(1)

LogParser, benchmark_settings = load_parser_assets(parser_name)


input_dir = "logparser/data/loghub_2k/"
output_dir = os.path.join("Benchmark_results", f"{parser_name}_results")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


benchmark_result = []
for dataset, setting in benchmark_settings.items():
    print(f"\n=== Evaluation on {dataset} using {parser_name} ===")
    
    indir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
    log_file = os.path.basename(setting["log_file"])

    
    params = {
        "log_format": setting["log_format"],
        "indir": indir,
        "outdir": output_dir,
        "rex": setting["regex"]
    }
    
    extra_params = {k: v for k, v in setting.items() if k not in ['log_file', 'log_format', 'regex']}
    params.update(extra_params)

    try:
        parser_instance = LogParser(**params)
        parser_instance.parse(log_file)

        # Evaluera
        groundtruth = os.path.join(indir, log_file + "_structured.csv")
        parsed_result = os.path.join(output_dir, log_file + "_structured.csv")
        
        f1_measure, accuracy, precision, recall = evaluator.evaluate(
            groundtruth=groundtruth,
            parsedresult=parsed_result,
        )
        benchmark_result.append([dataset, f1_measure, accuracy, precision, recall])
        
    except Exception as e:
        print(f"Failed to parse {dataset}: {e}")


df_result = pd.DataFrame(benchmark_result, columns=["Dataset", "F1_measure", "Accuracy", "Precision", "Recall"])
print(df_result)

if not os.path.exists(PARSER_RESULTS_DIR):            
    os.makedirs(PARSER_RESULTS_DIR)


df_result.to_csv(f"{PARSER_RESULTS_DIR}/{parser_name}_benchmark_result.csv")
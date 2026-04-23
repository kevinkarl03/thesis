#!/usr/bin/env python

import sys
sys.path.append('../../')
from logparser.Drain import LogParser
from logparser.utils import evaluator
import os
import pandas as pd



# input_dir  = '../../data/loghub_2k/Spark/' # The input directory of log file
input_dir  = '../../data/welkin/' # The input directory of log file
output_dir = input_dir + 'results/'  # The output directory of parsing results
log_file   = 'Welkin_klog_2k.log'  # The input log file name
# Regular expression list for optional preprocessing (default: [])


regex = [
    # 1. Datum och UTC-tid (t.ex. 2026-03-24 16:01:24 +0000 UTC)
    r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(\.\d+)?\s\+\d{4}\sUTC',

    # 2. Versionsnummer (v1.34.1)
    r'v?\d+\.\d+\.\d+',

    # 3. Logger och Key-Value par (logger="...", controller="...")
    r'(?<=="")([^""]+)(?="")',
    
    # 4. Tidsangivelser med enheter (1.739269429s, 8.77µs, 124ms)
    r'\d+(\.\d+)?(s|ms|µs|ns)',

    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
]


log_format = '<LevelDate> <Time>       <Pid> <Source>] <Content>'
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
bechmark_result = []
F1_measure, accuracy = evaluator.evaluate(
        groundtruth=os.path.join(input_dir, "Welkin_2k_CLEAN.csv"),
        parsedresult=os.path.join(output_dir, log_file + "_structured.csv"),
    )
bechmark_result.append(["Welkin", F1_measure, accuracy])


print("\n=== Overall evaluation results ===")
df_result = pd.DataFrame(bechmark_result, columns=["Dataset", "F1_measure", "Accuracy"])
df_result.set_index("Dataset", inplace=True)
print(df_result)


# Welkin     0.903184     0.735

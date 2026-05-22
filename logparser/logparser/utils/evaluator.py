# =========================================================================
# Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import re

import pandas as pd
from scipy.special import comb

# Regex matching a key=value pair used by the custom klog preprocessor.
# Matches both quoted (key="value") and unquoted (key=value) forms.
_KV_PAIR_RE = re.compile(r'\w+="[^"]*"|\w+=[a-zA-Z0-9_./-]+')


def _count_kv_lines(series):
    """Count how many lines in the series contain a key=value pair."""
    return series.dropna().apply(lambda x: bool(_KV_PAIR_RE.search(str(x)))).sum()


def evaluate(groundtruth, parsedresult):
    """Evaluation function to benchmark log parsing accuracy

    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """
    import os

    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)

    # Count lines containing key=value pairs in the Content column
    gt_kv_count = _count_kv_lines(df_groundtruth["Content"])
    parsed_kv_count = _count_kv_lines(df_parsedlog["Content"])

    # Remove invalid groundtruth event Ids
    non_empty_log_ids = df_groundtruth[~df_groundtruth["EventId"].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]
    (precision, recall, f_measure, accuracy) = get_accuracy(
        df_groundtruth["EventId"], df_parsedlog["EventId"]
    )
    print(
        "Precision: {:.4f}, Recall: {:.4f}, F1_measure: {:.4f}, Parsing_Accuracy: {:.4f}".format(
            precision, recall, f_measure, accuracy
        )
    )
    print(
        "KV-pair ground truth: {} / {} ({:.1f}%)\t"
        "KV-pair parsed result: {} / {} ({:.1f}%)".format(
            gt_kv_count, len(df_groundtruth) if not df_groundtruth.empty else 0,
            100*gt_kv_count/len(df_groundtruth) if not df_groundtruth.empty else 0.0,
            parsed_kv_count, len(df_parsedlog) if not df_parsedlog.empty else 0,
            100*parsed_kv_count/len(df_parsedlog) if not df_parsedlog.empty else 0.0,
        )
    )

    # Write key-value pair stats to a companion file next to parsedresult
    kv_stats_file = parsedresult.rstrip(".csv") + "_kv_stats.csv"
    df_stats = pd.DataFrame([{
        "groundtruth_file": os.path.basename(groundtruth),
        "parsedresult_file": os.path.basename(parsedresult),
        "gt_total_lines": len(df_groundtruth),
        "gt_kv_lines": gt_kv_count,
        "gt_kv_pct": round(100*gt_kv_count/len(df_groundtruth), 1) if not df_groundtruth.empty else 0.0,
        "parsed_total_lines": len(df_parsedlog),
        "parsed_kv_lines": parsed_kv_count,
        "parsed_kv_pct": round(100*parsed_kv_count/len(df_parsedlog), 1) if not df_parsedlog.empty else 0.0,
    }])
    df_stats.to_csv(kv_stats_file, index=False)

    return f_measure, accuracy, precision, recall


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """Compute accuracy metrics between log parsing results and ground truth

    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                logIds.size
                == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

# METRIC INTERPRETATION:
# - Accuracy (Grouping Accuracy): Strict measure. A log is 'correct' only if its 
#   entire cluster matches the Ground Truth perfectly.
# - Precision: Measures 'Over-merging'. Low precision means different templates 
#   were incorrectly grouped together.
# - Recall: Measures 'Over-splitting'. Low recall means the same template was 
#   incorrectly split into multiple clusters.
# - F1-measure: The harmonic mean of Precision and Recall, representing the 
#   overall balance of the parser.

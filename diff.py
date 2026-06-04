import pandas as pd
import argparse


def analyze_template_errors(ground_truth_path, parsed_result_path, output_file="template_diff_analysis.csv"):
    """
    Compare ground truth and parsed log files, output only lines where EventTemplate differs.
    
    For each differing line, shows:
    - LineId
    - Content (original log message)
    - EventTemplate_gt (expected template)
    - EventTemplate_parsed (actual parsed template)
    """
    # Read the files
    gt = pd.read_csv(ground_truth_path)
    parsed = pd.read_csv(parsed_result_path)
    
    # Clean up parsed file - strip quotes from Content and EventTemplate
    content_cols = ["Content", "EventTemplate"]
    for col in content_cols:
        if col in parsed.columns:
            parsed[col] = parsed[col].astype(str).str.strip('"')
    
    # Also clean ground truth for consistent comparison
    for col in content_cols:
        if col in gt.columns:
            gt[col] = gt[col].astype(str).str.strip('"')
    
    # Merge on LineId
    comparison = gt.merge(parsed, on='LineId', suffixes=('_gt', '_parsed'))
    
    # Filter rows where EventTemplate differs
    errors = comparison[
        comparison['EventTemplate_gt'].fillna('') != comparison['EventTemplate_parsed'].fillna('')
    ].copy()
    
    # Select relevant columns
    cols_to_keep = [
        'LineId', 
        # 'Content_gt', 
        'EventTemplate_gt', 
        'EventTemplate_parsed'
    ]
    
    # Check that columns exist
    final_cols = [c for c in cols_to_keep if c in errors.columns]
    errors_cleaned = errors[final_cols]
    
    # Rename for clarity
    rename_map = {
        'Content_gt': 'Content',
        'EventTemplate_gt': 'EventTemplate_GroundTruth',
        'EventTemplate_parsed': 'EventTemplate_Parsed'
    }
    errors_cleaned = errors_cleaned.rename(columns=rename_map)
    
    # Save to CSV
    errors_cleaned.to_csv(output_file, index=False)
    
    # Print summary
    total_lines = len(gt)
    diff_lines = len(errors_cleaned)
    accuracy = ((total_lines - diff_lines) / total_lines) * 100 if total_lines > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"TEMPLATE DIFF ANALYSIS")
    print(f"{'='*60}")
    print(f"Ground truth file: {ground_truth_path}")
    print(f"Parsed file:       {parsed_result_path}")
    print(f"Total lines:       {total_lines}")
    print(f"Lines differing:   {diff_lines}")
    print(f"Lines matching:    {total_lines - diff_lines}")
    print(f"Template Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to:  {output_file}")
    print(f"{'='*60}\n")
    
    return errors_cleaned


def analyze_content_errors(ground_truth_path, parsed_result_path, output_file="content_diff_analysis.csv"):
    """
    Compare ground truth and parsed log files, output only lines where Content differs.
    This is useful when preprocessing changes the content.
    """
    gt = pd.read_csv(ground_truth_path)
    parsed = pd.read_csv(parsed_result_path)
    
    # Clean up
    for col in ["Content", "EventTemplate"]:
        if col in parsed.columns:
            parsed[col] = parsed[col].astype(str).str.strip('"')
        if col in gt.columns:
            gt[col] = gt[col].astype(str).str.strip('"')
    
    # Merge
    comparison = gt.merge(parsed, on='LineId', suffixes=('_gt', '_parsed'))
    
    # Filter where Content differs
    errors = comparison[
        comparison['Content_gt'].fillna('') != comparison['Content_parsed'].fillna('')
    ].copy()
    
    # Select columns
    cols_to_keep = ['LineId', 'Content_gt', 'Content_parsed', 'EventTemplate_gt', 'EventTemplate_parsed']
    final_cols = [c for c in cols_to_keep if c in errors.columns]
    errors_cleaned = errors[final_cols].rename(columns={
        'Content_gt': 'Content_GroundTruth',
        'Content_parsed': 'Content_Parsed',
        'EventTemplate_gt': 'EventTemplate_GroundTruth',
        'EventTemplate_parsed': 'EventTemplate_Parsed'
    })
    
    # Save
    errors_cleaned.to_csv(output_file, index=False)
    
    print(f"Content differences: {len(errors_cleaned)} lines")
    print(f"Saved to: {output_file}")
    
    return errors_cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare parsed log results with ground truth")
    # parser.add_argument("--parsed", default="Benchmark_results/Drain_results/Welkin_2k_v2.log_custom_structured.csv",
    #                    help="Path to parsed result CSV")
    parser.add_argument("--parsed", default="Benchmark_results/Drain_results/Welkin_2k.log_structured.csv",
                       help="Path to parsed result CSV")
    parser.add_argument("--ground-truth", default="logparser/data/loghub_2k/Welkin/Welkin_2k.log_structured.csv",
                       help="Path to ground truth CSV")
    parser.add_argument("--output", default="template_diff_analysis.csv",
                       help="Output file for differences")
    parser.add_argument("--mode", choices=["template", "content", "both"], default="template",
                       help="What to compare: template (default), content, or both")
    
    args = parser.parse_args()
    
    if args.mode in ["template", "both"]:
        print("\n=== Template Comparison ===")
        analyze_template_errors(args.ground_truth, args.parsed, args.output)
    
    if args.mode in ["content", "both"]:
        print("\n=== Content Comparison ===")
        analyze_content_errors(args.ground_truth, args.parsed, "content_diff_analysis.csv")
import pandas as pd

import pandas as pd

# Welkin v1 har 1433 rader

def analyze_template_errors(ground_truth_path, parsed_result_path):
    # Läs in filerna
    gt = pd.read_csv(ground_truth_path)
    parsed = pd.read_csv(parsed_result_path)
    
    content_cols = ["Content", "EventTemplate"]

    for col in parsed.columns:

        if col in content_cols:
            parsed[col] = parsed[col].astype(str).str.strip('"')

    # Slå ihop filerna på LineId
    # Vi antar att båda filerna har kolumnerna 'LineId', 'Content' och 'EventTemplate'
    comparison = gt.merge(parsed, on='LineId', suffixes=('_gt', '_parsed'))
    
    # Filtrera fram rader där mallarna (EventTemplate) skiljer sig åt
    # Vi använder .fillna('') för att hantera eventuella tomma rader utan att krascha
    errors = comparison[
        comparison['EventTemplate_gt'].fillna('') != comparison['EventTemplate_parsed'].fillna('')
    ].copy()
    
    # Välj ut endast de relevanta kolumnerna för analys
    # Vi behåller Content_gt för att se vad rådatan var
    cols_to_keep = [
        'LineId', 
        'Content_gt', 
        'EventTemplate_gt', 
        'EventTemplate_parsed'
    ]
    
    # Kontrollera att kolumnerna finns (vissa verktyg döper om Content)
    final_cols = [c for c in cols_to_keep if c in errors.columns]
    errors_cleaned = errors[final_cols]
    
    # Spara till CSV
    errors_cleaned.to_csv("template_diff_analysis.csv", index=False)
    
    print(f"Analys klar: Hittade {len(errors_cleaned)} rader där mallarna skiljer sig.")
    print("Resultatet har sparats till 'template_diff_analysis.csv'")
    
    return errors_cleaned


if __name__ == "__main__":

    parsed = "Benchmark_results/Drain_results/Welkin_2k_v2.log_structured.csv"

    truth = "logparser/data/loghub_2k/Welkin_v2/Welkin_2k_v2.log_structured.csv"

    analyze_template_errors(truth, parsed)
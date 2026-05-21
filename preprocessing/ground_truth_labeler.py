# #!/usr/bin/env python

import sys
sys.path.append('../../')
import os
import pandas as pd




import pandas as pd
import csv

def clean_csv_quotations(input_file, output_file):
    # 1. Läs in filen
    # Vi antar att den är sparad med QUOTE_ALL (därför ser den ut som ditt exempel)
    df = pd.read_csv(input_file)

    # Kolumner som ska vara helt rena (utan några citattecken alls)
    clean_cols = ["LineId", "LevelDate", "Time", "Pid", "Source", "EventId"]
    
    # Kolumner som ska behålla sina inre citattecken (men inte trippla)
    content_cols = ["Content", "EventTemplate"]

    for col in df.columns:
        if col in clean_cols:
            # Ta bort alla citattecken och gör till sträng/int
            df[col] = df[col].astype(str).str.replace('"', '')
        elif col in content_cols:
            # Ta bort de "trippla" citattecknen genom att ersätta "" med "
            # Pandas läser oftast in "" som ", så vi säkerställer att de är rena
            df[col] = df[col].astype(str).str.strip('"')

    # 2. Spara med QUOTE_MINIMAL
    # Detta gör att LineId, Time etc. INTE får citattecken, 
    # medan Content BARA får det om det behövs (t.ex. om det finns ett komma)
    df.to_csv(
        output_file, 
        index=False, 
        quoting=csv.QUOTE_MINIMAL, 
        quotechar='"', 
        doublequote=True
    )

    print(f"Filen har städats och sparats som: {output_file}")


    df = pd.read_csv(output_file)

    # 1. Kolla efter helt identiska rader
    duplicates = df[df.duplicated()]
    print(f"Antal helt identiska rader: {len(duplicates)}")

    # 2. Kolla efter dubbletter i LineId specifikt
    id_duplicates = df[df.duplicated(['LineId'], keep=False)]
    print("Rader med samma LineId:")
    print(id_duplicates)

    # 3. Kolla om det finns en tom rad i slutet
    print(f"Faktiskt antal rader i DataFrame: {len(df)}")


# if __name__ == "__main__":
#     INPUT = '../../data/welkin/Welkin_2k_dataset.csv'
#     OUTPUT = '../../data/welkin/Welkin_2k_CLEAN.csv'
#     clean_csv_quotations(INPUT, OUTPUT)



import pandas as pd
import hashlib
import csv


def merge_manual_labels(parsed_file, labeled_file, output_file):
    print(f"1. Läser in bas-filen (den parsade loggen)...")
    df_parsed = pd.read_csv(parsed_file)

    print(f"2. Läser in din manuella Excel-fil...")
    df_labeled = pd.read_excel(labeled_file)

    # Vi skapar mappen: { "Exakt loggmeddelande": "Din manuella template" }
    label_map = dict(zip(
        df_labeled['Content'].astype(str).str.strip(), 
        df_labeled['Manual_EventTemplate'].astype(str).str.strip()
    ))

    print(f"3. Letar efter exakta matcher mellan logg och Excel...")

    # Här sker magin: För varje rad i bas-filen kollar vi om 'Content' finns i vår Excel-mapp
    df_parsed['EventTemplate'] = df_parsed['Content'].astype(str).str.strip().map(label_map)

    # 4. Fixa EventId (E1, E2, E3...) baserat på de templates som faktiskt hittades
    unique_templates = []
    for t in df_parsed['EventTemplate']:
        if pd.notna(t) and t not in unique_templates:
            unique_templates.append(t)
    
    event_id_map = {template: f"E{i+1}" for i, template in enumerate(unique_templates)}
    df_parsed['EventId'] = df_parsed['EventTemplate'].map(event_id_map).fillna("UNKNOWN")

    # Ta bort ParameterList för att hålla filen ren
    if 'ParameterList' in df_parsed.columns:
        df_parsed = df_parsed.drop(columns=['ParameterList'])

    # 5. Spara resultatet (med de fula men säkra citattecknen)
    df_parsed.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

    print("-" * 30)
    print(f"KLART! Sparad till: {output_file}")
    
    # Debug: Visa vad som saknas
    missing_df = df_parsed[df_parsed['EventTemplate'].isna()]
    if not missing_df.empty:
        print(f"VARNING: {len(missing_df)} rader hittade ingen matchning i Excel-filen.")
        print("Här är de första unika meddelandena som saknas:")
        for msg in missing_df['Content'].unique()[:50]:
            print(f"{msg}")
    else:
        print("Snyggt! Alla rader matchades perfekt.")

if __name__ == "__main__":
    # Dina fil-sökvägar
    STRUCTURED = 'Benchmark_results/Drain_results/Welkin_2k.log_structured.csv'
    
    GROUND_TRUTH = 'audit_sheet_READY_TO_EDIT.ods'
    OUTPUT = 'Welkin_2k_dataset_v3.csv'

    merge_manual_labels(STRUCTURED, GROUND_TRUTH, OUTPUT)
    clean_csv_quotations(OUTPUT, OUTPUT)

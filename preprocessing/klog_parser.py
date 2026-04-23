# 98c71f14,"""Starting endpoint slice <*> <*>",7 är knas den tar in för mycket
# 88b7ac1a,"""Starting <*> <*> controller""",21 Tar också in för mycket
# 13aea5fd,"""Waiting for caches to sync"" <*> <*> <*>",7
# e79a5219,"""Waiting for caches to sync"" <*> <*>",30
# GroupVersion Missar några





import os
import re
import pandas as pd

def extract_klogs(input_dir, output_file):
    # Regex for klog: Starts with I, W, E, or F, followed by 4 digits (MMDD), 
    # then a space and the timestamp (HH:MM:SS.micro)
    klog_pattern = re.compile(r'^[IWEF]\d{4}\s\d{2}:\d{2}:\d{2}\.\d+')

    klog_lines_found = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the directory (including subfolders)
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                # Avoid reading the output file if it's in the same directory
                if file == output_file:
                    continue
                
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        for line in infile:
                            if klog_pattern.match(line):
                                outfile.write(line)
                                klog_lines_found += 1
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    print(f"Success! Extracted {klog_lines_found} klog lines into {output_file}")

# --- CONFIGURATION ---
TARGET_DIRECTORY = './logs'  # Change this to your directory path
RESULT_FILE = 'extracted_klog.log'
SORTED_RESULTS_FILE = 'extracted_sorted_klog.log'
FILE_2K = 'extracted_klog_2k.log'



from datetime import datetime

def sort_klog_file(input_file, output_file, year=2026):
    lines_with_dates = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # klog header format: I0414 15:52:42.332897
            # We take the first 21 characters for the timestamp
            try:
                # Extract parts: 'I' + '0414 15:52:42.332897'
                header_part = line[1:22].strip()
                
                # Convert to datetime object for accurate sorting
                # %m%d %H:%M:%S.%f handles MMDD and the microsecond time
                dt = datetime.strptime(f"{year}{header_part}", "%Y%m%d %H:%M:%S.%f")
                
                lines_with_dates.append((dt, line))
            except (ValueError, IndexError):
                # If a line doesn't match the format exactly, we skip it
                # or append it to the previous log line if it's a multi-line log
                continue

    # Sort based on the datetime object (the first element in our tuple)
    lines_with_dates.sort(key=lambda x: x[0])

    # Write the sorted lines back out
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, line in lines_with_dates:
            f.write(line)

    print(f"Sorted {len(lines_with_dates)} lines into {output_file}")

# --- EXECUTION ---


import collections
import random

def get_diverse_sample(input_file,output_file ,target_count=2000):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Create a 'Signature' for each line by ignoring the first 4 metadata blocks
    # Line: I0317 14:46:27.815342 1 main.go:154] "Version"...
    # Parts: [0]I0317 [1]14:46:27 [2]1 [3]main.go:154] [4]"Version"
    
    line_groups = collections.defaultdict(list)
    
    for line in lines:
        parts = line.split()
        if len(parts) > 4:
            # We use the Code Source and the first word of the message as the key
            # e.g., "main.go:154] Version"
            signature = f"{parts[3]} {parts[4]}"
        else:
            signature = line # Fallback for weird lines
            
        line_groups[signature].append(line)

    print(f"Detected {len(line_groups)} unique log templates.")

    final_sample = []
    
    # Stratified Selection: 
    # We want to take a few samples from the 'heavy' groups
    # and almost all samples from the 'rare' groups.
    
    # 1. Sort groups by rarity (smallest first)
    sorted_signatures = sorted(line_groups.items(), key=lambda x: len(x[1]))
    
    for sig, examples in sorted_signatures:
        if len(examples) <= 7:
            # It's rare! Keep all of them.
            final_sample.extend(examples)
        else:
            # It's common! Only take a small representative sample (e.g., 10%)
            # This ensures the 'Heartbeats' don't drown out the 'Errors'
            sample_size = max(2, int(len(examples) * 0.15))
            final_sample.extend(random.sample(examples, min(sample_size, 50)))

    # Final polish: shuffle so the 2000 lines aren't ordered by "rarity"
    random.shuffle(final_sample)
    

    print(f"Created a diverse pool of {len(final_sample)} lines.")
    final_sample = final_sample[:target_count]

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in final_sample:
            f.write(line)
    return


def pre_mask(text):
    # Mask common things automatically
    text = re.sub(r'(\d+\.){3}\d+', '<*>', text) # IPs
    text = re.sub(r'0x[a-fA-F0-9]+', '<*>', text) # Hex
    text = re.sub(r'backman-wc-[\w-]+', '<*>', text) # Your specific node/pod pattern
    return text




    

if __name__ == "__main__":
    # extract_klogs(TARGET_DIRECTORY, RESULT_FILE)
    # sort_klog_file(RESULT_FILE, RESULT_FILE)
    # get_diverse_sample(RESULT_FILE, FILE_2K , target_count=2000)
    # sort_klog_file(FILE_2K, FILE_2K)




    STRUCTURED = './logparser/data/welkin/results/extracted_klog_2k.log_structured.csv'
    PATH_TO_AUDIT = './logparser/data/welkin/results/manual_audit.csv'

    df = pd.read_csv(STRUCTURED)

    # Group by Source AND Content to find unique occurrences
    # We count them so you know which templates are the most 'chatty'
    audit_df = df.groupby(['Source', 'Content']).size().reset_index(name='Occurrences')

    # Sort by Source so you can see all logs from 'main.go' together
    audit_df = audit_df.sort_values(by=['Source', 'Occurrences'], ascending=[True, False])

    # Add a blank column for your manual Template corrections
    audit_df['Manual_EventTemplate'] = ""

    audit_df.to_csv(PATH_TO_AUDIT, index=False)
    print(f"Audit sheet created with {len(audit_df)} unique source/content pairs.")


    df = pd.read_csv(PATH_TO_AUDIT)
    df['Manual_EventTemplate'] = df['Content'].apply(pre_mask)
    df.to_csv('audit_sheet_READY_TO_EDIT.csv', index=False)



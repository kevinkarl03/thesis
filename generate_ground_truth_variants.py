#!/usr/bin/env python3
"""
Script to generate 5 new ground truth files from Welkin_v3 log file.
Each file will contain 2000 lines with varying numbers of templates.

Key features:
- LevelDate in chronological order (grouped by date)
- All-unique timestamps per line, in chronological order
- No consecutive repeated EventIds (guaranteed via ordered deque interleaving)
- EventTemplate retains the actual template text with <*> wildcards
- Templates renumbered sequentially as EventId (E1, E2, E3, ...)
"""

import csv
import random
import os
from collections import defaultdict, OrderedDict, deque

def read_original_log(filepath):
    templates = defaultdict(list)
    header = None
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 8:
                template = row[7].strip()
                templates[template].append(row)
    return header, templates

def generate_unique_times(num_times):
    times = set()
    while len(times) < num_times:
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        micro = random.randint(0, 999999)
        times.add(f"{hour:02d}:{minute:02d}:{second:02d}.{micro:06d}")
    return sorted(times)

def interleave_rows(rows):
    """
    Rearrange rows so no two consecutive rows have the same template.
    
    Uses the following strategy:
    1. Shuffle within each template group
    2. Arrange groups by size (largest first)
    3. Take one from each group in round-robin fashion
    4. When a group runs out, remove it from rotation
    
    This guarantees no consecutive duplicates regardless of template distribution.
    """
    if not rows:
        return rows
    
    # Group by template
    groups = defaultdict(deque)
    for row in rows:
        template = row[7].strip()
        groups[template].append(row)
    
    # Shuffle within groups
    for template in groups:
        lst = list(groups[template])
        random.shuffle(lst)
        groups[template] = deque(lst)
    
    # Sort groups by size (largest first) for round-robin order
    template_order = sorted(groups.keys(), key=lambda t: len(groups[t]), reverse=True)
    
    # Interleave: at each step, pick the group with most remaining items
    # that's different from the last one used
    result = []
    last_template = None
    
    # Track remaining count per template
    remaining = {t: len(groups[t]) for t in template_order}
    
    while any(remaining.values()):
        # Find template with most remaining that != last_template
        best_template = None
        best_count = -1
        
        for t in template_order:
            if remaining[t] > 0 and t != last_template:
                if remaining[t] > best_count:
                    best_count = remaining[t]
                    best_template = t
        
        # If forced to use last_template, pick the one with most remaining
        if best_template is None:
            for t in template_order:
                if remaining[t] > 0 and remaining[t] > best_count:
                    best_count = remaining[t]
                    best_template = t
        
        if best_template is None:
            break
        
        row = groups[best_template].popleft()
        result.append(row)
        remaining[best_template] -= 1
        last_template = best_template
    
    return result

def generate_ground_truth_file(header, templates, target_templates, output_path):
    # Select templates
    available_templates = list(templates.keys())
    if target_templates > len(available_templates):
        target_templates = len(available_templates)
    
    selected_templates = random.sample(available_templates, target_templates)
    
    # Create mapping: original template -> new EventId (E1, E2, ...)
    template_renumbering = OrderedDict()
    for i, template in enumerate(selected_templates, 1):
        template_renumbering[template] = f"E{i}"
    
    # Distribute 2000 lines across templates
    rows_per_template = {t: 1 for t in selected_templates}
    remaining = 2000 - len(selected_templates)
    
    # Use exponential distribution for realistic frequencies
    weights = [random.expovariate(0.3) for _ in selected_templates]
    total_weight = sum(weights)
    counts = [int(remaining * w / total_weight) for w in weights]
    
    # Adjust to exactly 2000
    current_total = sum(counts) + len(selected_templates)
    diff = 2000 - current_total
    if diff > 0:
        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
        for idx in sorted_indices[:diff]:
            counts[idx] += 1
    
    for template, count in zip(selected_templates, counts):
        rows_per_template[template] += count
    
    # Build selected rows
    all_rows = []
    for template, count in rows_per_template.items():
        available = list(templates[template])
        if count > len(available):
            chosen = random.choices(available, k=count)
        else:
            chosen = random.sample(available, count)
        all_rows.extend(chosen)
    
    # Group by date to maintain chronological order
    date_groups = defaultdict(list)
    for row in all_rows:
        date_part = row[1][1:]
        date_groups[date_part].append(row)
    
    sorted_dates = sorted(date_groups.keys())
    
    # Interleave within each date group
    final_rows = []
    for date in sorted_dates:
        interleaved = interleave_rows(date_groups[date])
        final_rows.extend(interleaved)
    
    # Verify no consecutive duplicates exist
    final_rows = interleave_rows(final_rows)
    
    # Generate unique times
    unique_times = generate_unique_times(2000)
    
    # Write output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i, row in enumerate(final_rows, 1):
            original_template = row[7].strip()
            new_event_id = template_renumbering[original_template]
            
            new_row = [
                i,                         # LineId (sequential)
                row[1],                    # LevelDate (original)
                unique_times[i-1],         # Unique time
                row[3],                    # Pid (original)
                row[4],                    # Source (original)
                row[5],                    # Content (original)
                new_event_id,              # EventId (renumbered)
                original_template          # EventTemplate (original text)
            ]
            writer.writerow(new_row)
    
    return len(selected_templates)

def main():
    input_file = "logparser/data/loghub_2k/Welkin_v3/Welkin_2k_v3.log_structured.csv"
    output_dir = "ground_truth_variants"
    os.makedirs(output_dir, exist_ok=True)
    
    template_counts = [300, 250, 200, 150, 100]
    
    print(f"Reading original log file: {input_file}")
    header, templates = read_original_log(input_file)
    
    total_lines = sum(len(v) for v in templates.values())
    print(f"Original file has {total_lines} lines, {len(templates)} unique templates")
    
    for i, target_count in enumerate(template_counts, 1):
        output_file = os.path.join(output_dir, f"Welkin_2k_v3_ground_truth_{target_count}templates.csv")
        print(f"\nGenerating file {i}/5: {output_file}")
        print(f"  Target templates: {target_count}")
        
        actual_count = generate_ground_truth_file(header, templates, target_count, output_file)
        print(f"  Templates used: {actual_count}")
        print(f"  Lines: 2000")
    
    print(f"\n✓ All 5 ground truth files generated in '{output_dir}' directory")

if __name__ == "__main__":
    random.seed(42)
    main()
# -*- coding: utf-8 -*-
import csv

input_path = r'D:\VScode\VScodeProjects\Fond\output\ablation_results.csv'
output_path = r'D:\VScode\VScodeProjects\Fond\output\ablation_results_fixed.csv'

rows = []
with open(input_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0].startswith('- '):
            row[0] = row[0][2:]  # remove "- " prefix
        rows.append(row)

with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f'Fixed version saved to {output_path}')
print(f'Total rows: {len(rows)}')

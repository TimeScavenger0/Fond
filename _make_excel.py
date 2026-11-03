# -*- coding: utf-8 -*-
import json
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

with open(r"D:\VScode\VScodeProjects\Fond\output\eval_baseline.json", "r", encoding="utf-8") as f:
    data = json.load(f)

wb = Workbook()
ws = wb.active
ws.title = "Baseline Results"

# Headers
headers = ["Letter", "Final Loss", "Time (s)", "CLIP Score Pre", "CLIP Score Post", "CLIP Score Delta"]
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF", size=11)
thin_border = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin')
)

for col, h in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=h)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.border = thin_border

# Data rows
for i, row in enumerate(data, 2):
    ws.cell(row=i, column=1, value=row["letter"]).alignment = Alignment(horizontal='center')
    ws.cell(row=i, column=2, value=round(row["final_loss"], 6))
    ws.cell(row=i, column=3, value=round(row["time"], 1))
    # CLIP scores placeholder (to be computed)
    ws.cell(row=i, column=4, value="TBD")
    ws.cell(row=i, column=5, value="TBD")
    ws.cell(row=i, column=6, value="TBD")
    for col in range(1, 7):
        ws.cell(row=i, column=col).border = thin_border

# Summary row
avg_loss = sum(r["final_loss"] for r in data) / len(data)
avg_time = sum(r["time"] for r in data) / len(data)
summary_row = len(data) + 2
ws.cell(row=summary_row, column=1, value="Average").font = Font(bold=True)
ws.cell(row=summary_row, column=2, value=round(avg_loss, 6)).font = Font(bold=True)
ws.cell(row=summary_row, column=3, value=round(avg_time, 1)).font = Font(bold=True)
for col in range(1, 7):
    ws.cell(row=summary_row, column=col).border = thin_border
    ws.cell(row=summary_row, column=col).fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")

# Column widths
ws.column_dimensions['A'].width = 10
ws.column_dimensions['B'].width = 14
ws.column_dimensions['C'].width = 12
ws.column_dimensions['D'].width = 16
ws.column_dimensions['E'].width = 16
ws.column_dimensions['F'].width = 16

# Add a second sheet for config
ws2 = wb.create_sheet("Config")
config_data = [
    ["Parameter", "Value"],
    ["Font", "gwfonts/Arimo-Regular.ttf"],
    ["Style Prompt", "italic, slanted font"],
    ["Iterations", 300],
    ["Learning Rate", 1.5],
    ["use_aug", True],
    ["fclip_loss_w", 2.0],
    ["conformal_loss_w", 8.0],
    ["tone_loss_w", 5.0],
    ["Two-stage LR", "Yes (60% switch)"],
    ["Best-of-3 sampling", "Yes"],
    ["Anchor gradient scale", 0.3],
]
for r, row in enumerate(config_data, 1):
    for c, val in enumerate(row, 1):
        cell = ws2.cell(row=r, column=c, value=val)
        if r == 1:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
ws2.column_dimensions['A'].width = 24
ws2.column_dimensions['B'].width = 30

out = r"D:\VScode\VScodeProjects\Fond\output\baseline_results.xlsx"
wb.save(out)
print(f"Saved to {out}")

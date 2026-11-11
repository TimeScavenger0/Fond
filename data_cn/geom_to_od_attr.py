# -*- coding: utf-8 -*-
"""把中文几何实测映射到 O'Donovan 0-100 量纲（fontclip 环境，只读）。

量纲对齐方式：O'Donovan 37 属性本质是字体间的相对位置，这里用 24 个字体的
相对百分位（rank percentile）映射到 0-100，不虚构绝对量纲；原始实测值并列保留。
"""
import os
import csv
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "font_level_attributes.csv")

rows = []
with open(src, "r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

n = len(rows)


def pct(values, ascending=True):
    x = np.array(values, float)
    if not ascending:
        x = -x
    ranks = rankdata(x, method="average")
    return np.round((ranks - 1) / (n - 1) * 100.0, 2)


slant = np.array([abs(float(r["slant_deg"])) for r in rows])
width = np.array([float(r["width_ratio"]) for r in rows])
black = np.array([float(r["blackness"]) for r in rows])
halfw = np.array([float(r["stroke_halfw"]) for r in rows])

italic_pct = pct(slant)                 # 越倾斜 italic 越高
wide_pct = pct(width)                   # 越宽 wide 越高
thin_pct = pct(black, ascending=False)  # 黑度越低越细 thin 越高
strong_pct = pct(black)                 # 越粗越重 strong 越高

out_rows = []
for i, r in enumerate(rows):
    out_rows.append({
        "file": r["file"], "family": r["family"],
        "slant_deg": r["slant_deg"], "italic_pct": italic_pct[i],
        "width_ratio": r["width_ratio"], "wide_pct": wide_pct[i],
        "blackness": r["blackness"], "stroke_halfw": r["stroke_halfw"],
        "thin_pct": thin_pct[i], "strong_pct": strong_pct[i],
        "source": "chinese_geometric_measured_rank",
    })

cols = ["file", "family", "slant_deg", "italic_pct", "width_ratio",
        "wide_pct", "blackness", "stroke_halfw", "thin_pct",
        "strong_pct", "source"]
out = os.path.join(HERE, "cn_geometric_od_attributes.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out_rows)

print("fonts:", n, "->", out)
print("\ncheck monotonic ordering (top/bottom):")
for key, pctcol in [("italic", italic_pct), ("wide", wide_pct),
                    ("thin", thin_pct), ("strong", strong_pct)]:
    order = np.argsort(-pctcol)
    top = rows[order[0]]["file"]
    bot = rows[order[-1]]["file"]
    print("  %-7s highest=%-18s lowest=%s" % (key, top, bot))

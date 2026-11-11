# -*- coding: utf-8 -*-
"""汇总 7 个客观属性为统一 0-100 表（fontclip 环境，只读）。

几何 4 项（italic/wide/thin/strong）取自 cn_geometric_od_attributes.csv；
serif 用上轮廓突变斜率 max_slope 正向；
monospace 用拉丁 advance width 变异系数 latin_cv 反向（中文 CV 全 0 无区分）；
angular 用显著曲率峰中位数 angular_peak 正向。
原始实测指标全部保留，rank 百分位口径与 geom_to_od_attr.py 一致。
"""
import os
import csv
import json
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
N = 24


def pct(values, ascending=True):
    x = np.array(values, float)
    if not ascending:
        x = -x
    ranks = rankdata(x, method="average")
    return np.round((ranks - 1) / (N - 1) * 100.0, 2)


# 几何 4 项
geo = []
with open(os.path.join(HERE, "cn_geometric_od_attributes.csv"),
          "r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        geo.append(r)

with open(os.path.join(HERE, "serif_robust.json"), "r", encoding="utf-8") as f:
    serif = json.load(f)
with open(os.path.join(HERE, "monospace_cv.json"), "r", encoding="utf-8") as f:
    mono = json.load(f)
with open(os.path.join(HERE, "angular_metrics.json"), "r", encoding="utf-8") as f:
    ang = json.load(f)

files = [r["file"] for r in geo]
serif_raw = [serif[fn]["max_slope_n"] for fn in files]
lift_raw = [serif[fn]["peak_lift_n"] for fn in files]
cv_raw = [mono[fn]["latin_cv"] for fn in files]
ang_raw = [ang[fn]["angular_peak"] for fn in files]

serif_pct = pct(serif_raw, ascending=True)
mono_pct = pct(cv_raw, ascending=False)
ang_pct = pct(ang_raw, ascending=True)

out_rows = []
for i, r in enumerate(files):
    g = geo[i]
    out_rows.append({
        "file": g["file"],
        "family": g["family"],
        "italic_pct": g["italic_pct"],
        "wide_pct": g["wide_pct"],
        "thin_pct": g["thin_pct"],
        "strong_pct": g["strong_pct"],
        "serif_pct": serif_pct[i],
        "monospace_pct": mono_pct[i],
        "angular_pct": ang_pct[i],
        "slant_deg": g["slant_deg"],
        "width_ratio": g["width_ratio"],
        "blackness": g["blackness"],
        "stroke_halfw": g["stroke_halfw"],
        "serif_max_slope": serif_raw[i],
        "serif_peak_lift": lift_raw[i],
        "latin_cv": cv_raw[i],
        "angular_peak": ang_raw[i],
        "source": "chinese_objective_measured_rank",
    })

cols = ["file", "family", "italic_pct", "wide_pct", "thin_pct",
        "strong_pct", "serif_pct", "monospace_pct", "angular_pct",
        "slant_deg", "width_ratio", "blackness", "stroke_halfw",
        "serif_max_slope", "serif_peak_lift", "latin_cv",
        "angular_peak", "source"]
out = os.path.join(HERE, "cn_objective_attributes.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out_rows)

print("fonts:", N, "->", out)
# BOM 核对
with open(out, "rb") as f:
    print("BOM:", f.read(3) == b"\xef\xbb\xbf")
print("\nserif 最高/最低:")
o = np.argsort(-serif_pct)
print("  high=%s low=%s" % (files[o[0]], files[o[-1]]))
print("monospace 最高/最低:")
o = np.argsort(-mono_pct)
print("  high=%s low=%s" % (files[o[0]], files[o[-1]]))
print("angular 最高/最低:")
o = np.argsort(-ang_pct)
print("  high=%s low=%s" % (files[o[0]], files[o[-1]]))

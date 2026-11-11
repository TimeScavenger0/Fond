# -*- coding: utf-8 -*-
"""对 probe 指标做字体级聚合与合理性检查（只读，不修改原始数据）。"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "probe_metrics.json"), "r", encoding="utf-8") as f:
    recs = json.load(f)

keys = ["slant_ecc", "slant_regress", "width_ratio", "blackness", "stroke_halfw"]

# 1) 缺失/异常检查
nan_cnt = {k: 0 for k in keys}
for r in recs:
    for k in keys:
        if r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])):
            nan_cnt[k] += 1
print("records:", len(recs), "nan counts:", nan_cnt)

# 2) 两种倾斜角一致性
ecc = np.array([r["slant_ecc"] for r in recs], float)
reg = np.array([r["slant_regress"] for r in recs], float)
valid = ~(np.isnan(ecc) | np.isnan(reg))
corr = np.corrcoef(ecc[valid], reg[valid])[0, 1]
diff = np.abs(ecc[valid] - reg[valid])
print("slant corr(ecc, regress) = %.4f" % corr)
print("slant abs diff  mean=%.3f  median=%.3f  max=%.3f" % (
    diff.mean(), np.median(diff), diff.max()))

# 3) 字体级聚合（中位数）
fonts = {}
for r in recs:
    fonts.setdefault(r["file"], []).append(r)

rows = []
for fbase, items in fonts.items():
    agg = {k: float(np.median([x[k] for x in items])) for k in keys}
    rows.append((fbase, items[0].get("family", ""), agg))

# 按 blackness 排序输出字体级对比
print("\n%-20s %-14s %7s %7s %7s %8s %8s" % (
    "file", "family", "slantE", "slantR", "wRatio", "black", "halfw"))
for fbase, fam, agg in sorted(rows, key=lambda t: -t[2]["blackness"]):
    print("%-20s %-14s %7.2f %7.2f %7.3f %8.4f %8.3f" % (
        fbase, fam[:14], agg["slant_ecc"], agg["slant_regress"],
        agg["width_ratio"], agg["blackness"], agg["stroke_halfw"]))

# -*- coding: utf-8 -*-
"""按字符检查指标分布，确认无异常值、聚合稳健（只读）。"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "probe_metrics.json"), "r", encoding="utf-8") as f:
    recs = json.load(f)

chars = sorted(set(r["char"] for r in recs), key=lambda c: ord(c))
for k in ["width_ratio", "blackness", "stroke_halfw", "slant_regress"]:
    print("\n=== %s by char ===" % k)
    print("%-4s %8s %8s %8s %8s" % ("char", "min", "median", "max", "std"))
    for ch in chars:
        vals = np.array([r[k] for r in recs if r["char"] == ch], float)
        print("%-4s %8.3f %8.3f %8.3f %8.3f" % (
            ch, vals.min(), np.median(vals), vals.max(), vals.std()))

# blackness 与 stroke_halfw 的跨字体一致性（字体级中位数）
fonts = {}
for r in recs:
    fonts.setdefault(r["file"], []).append(r)
bl, sw = [], []
for items in fonts.values():
    bl.append(np.median([x["blackness"] for x in items]))
    sw.append(np.median([x["stroke_halfw"] for x in items]))
print("\nfont-level corr(blackness, stroke_halfw) = %.4f" % np.corrcoef(bl, sw)[0, 1])

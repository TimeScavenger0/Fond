# -*- coding: utf-8 -*-
"""对比字体级 slant 的不同聚合方式，确认结论对简单字符噪声稳健（只读）。"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "probe_metrics.json"), "r", encoding="utf-8") as f:
    recs = json.load(f)

COMPLEX = ["永", "国", "木", "水"]
fonts = {}
for r in recs:
    fonts.setdefault(r["file"], []).append(r)

print("%-20s %10s %10s %10s" % ("file", "medAll6", "medComplex", "meanAll6"))
diffs = []
for fbase in sorted(fonts):
    items = fonts[fbase]
    allv = np.array([x["slant_regress"] for x in items], float)
    cxv = np.array([x["slant_regress"] for x in items if x["char"] in COMPLEX], float)
    m_all, m_cx, mean_all = np.median(allv), np.median(cxv), np.mean(allv)
    diffs.append(abs(m_all - m_cx))
    print("%-20s %10.3f %10.3f %10.3f" % (fbase, m_all, m_cx, mean_all))

print("\n|medianAll - medianComplex|  max=%.3f mean=%.3f" % (max(diffs), np.mean(diffs)))

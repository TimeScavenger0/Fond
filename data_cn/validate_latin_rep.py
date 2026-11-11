# -*- coding: utf-8 -*-
"""验证中文字体的拉丁字形能否代表其中文风格（只读）。

对每个中文字体分别测量拉丁字符与中文字符的几何指标，比较一致性。
"""
import os
import json
import numpy as np

from geom_annotate import render_glyph, measure_all
from visual_match import MATCH_CHARS
from build_annotations import REP_CHARS

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
    inv = json.load(f)
cn_fonts = [r for r in inv if r.get("has_chinese")]


def agg(font_path, chars):
    rows = []
    for ch in chars:
        b = render_glyph(ch, font_path)
        if b is None:
            continue
        rows.append(measure_all(b))
    out = {}
    for k in ["slant_regress", "width_ratio", "blackness"]:
        vals = np.array([r[k] for r in rows], float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(np.median(vals)) if len(vals) else np.nan
    return out


print("%-20s | %-26s | %-26s" % ("file", "LATIN(slant,width,black)",
                                  "CHINESE(slant,width,black)"))
latin_w, chinese_w, latin_b, chinese_b, latin_s, chinese_s = [], [], [], [], [], []
for r in cn_fonts:
    la = agg(r["path"], MATCH_CHARS)
    ca = agg(r["path"], REP_CHARS)
    print("%-20s | %8.2f %7.3f %7.3f | %8.2f %7.3f %7.3f" % (
        r["file"], la["slant_regress"], la["width_ratio"], la["blackness"],
        ca["slant_regress"], ca["width_ratio"], ca["blackness"]))
    latin_s.append(la["slant_regress"]); chinese_s.append(ca["slant_regress"])
    latin_w.append(la["width_ratio"]); chinese_w.append(ca["width_ratio"])
    latin_b.append(la["blackness"]); chinese_b.append(ca["blackness"])

for name, x, y in [("slant", latin_s, chinese_s),
                   ("width", latin_w, chinese_w),
                   ("blackness", latin_b, chinese_b)]:
    x, y = np.array(x), np.array(y)
    m = ~(np.isnan(x) | np.isnan(y))
    print("corr latin-vs-chinese %-9s = %.4f" % (name, np.corrcoef(x[m], y[m])[0, 1]))

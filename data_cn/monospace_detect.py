# -*- coding: utf-8 -*-
"""等宽检测：读取 advance width，计算变异系数 CV（fontclip 环境，只读）。

CV = std/mean，越接近 0 越等宽。分别测量拉丁与中文，因为 O'Donovan 的
monospace 主要描述拉丁打字机/编程字体，而中文方块字天然等宽。
"""
import os
import json
import numpy as np
from fontTools.ttLib import TTFont

from build_annotations import REP_CHARS

HERE = os.path.dirname(os.path.abspath(__file__))
LATIN_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
    inv = json.load(f)
fonts = [r for r in inv if r.get("has_chinese")]


def advance_widths(font_path, chars):
    tt = TTFont(font_path, lazy=True)
    cmap = tt.getBestCmap()
    hmtx = tt["hmtx"]
    widths = []
    for ch in chars:
        gname = cmap.get(ord(ch))
        if gname is not None:
            widths.append(hmtx[gname][0])
    return np.array(widths, float)


print("%-20s %18s %18s" % ("file", "latin CV", "chinese CV"))
res = {}
for r in fonts:
    wl = advance_widths(r["path"], LATIN_CHARS)
    wc = advance_widths(r["path"], REP_CHARS)
    cv_l = float(wl.std() / wl.mean()) if wl.mean() else np.nan
    cv_c = float(wc.std() / wc.mean()) if wc.mean() else np.nan
    res[r["file"]] = {"latin_cv": round(cv_l, 4),
                      "chinese_cv": round(cv_c, 4)}
    print("%-20s %18.4f %18.4f" % (r["file"], cv_l, cv_c))

out = os.path.join(HERE, "monospace_cv.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("\njson ->", out)

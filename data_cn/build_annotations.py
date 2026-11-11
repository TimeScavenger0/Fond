# -*- coding: utf-8 -*-
"""扩展代表性字符集，输出字体级几何属性标注（fontclip 环境，纯 CPU，只读测量）。

输出（均为 utf-8-sig，Excel 可直接打开）：
  char_level_metrics.csv   每个 (字体, 字符) 的实测指标
  font_level_attributes.csv 字体级中位数聚合，作为 LoRA 训练的属性标签
"""
import os
import csv
import json
import numpy as np
from fontTools.ttLib import TTFont

from geom_annotate import render_glyph, measure_all

HERE = os.path.dirname(os.path.abspath(__file__))

# 44 个结构代表字符：独体 / 左右 / 上下 / 包围 / 品字复杂
REP_CHARS = [
    # 独体
    "永", "人", "口", "木", "水", "火", "山", "日", "月", "田",
    "土", "大", "女", "王", "心", "文",
    # 左右结构
    "林", "明", "好", "和", "的", "双", "羽", "朋",
    # 上下结构
    "字", "李", "吴", "花", "分", "要", "安", "会",
    # 包围结构
    "国", "回", "同", "问", "区", "医", "这", "建",
    # 品字 / 复杂
    "鑫", "森", "品", "众",
]

METRIC_COLS = ["slant_ecc_raw", "slant_deg", "width_ratio",
               "blackness", "stroke_halfw"]


def main():
    with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
        inventory = json.load(f)
    fonts = [r for r in inventory if r.get("has_chinese")]

    char_rows = []
    missing = []
    for r in fonts:
        font_path, fbase = r["path"], r["file"]
        cmap = TTFont(font_path, lazy=True).getBestCmap() or {}
        for ch in REP_CHARS:
            if ord(ch) not in cmap:
                missing.append((fbase, ch))
                continue
            binary = render_glyph(ch, font_path)
            if binary is None:
                missing.append((fbase, ch))
                continue
            m = measure_all(binary)
            char_rows.append({
                "file": fbase, "family": r.get("family", ""),
                "char": ch, "codepoint": "u%x" % ord(ch),
                "slant_ecc_raw": m["slant_ecc"],
                "slant_deg": m["slant_regress"],
                "width_ratio": m["width_ratio"],
                "blackness": m["blackness"],
                "stroke_halfw": m["stroke_halfw"],
            })

    # 字符级明细
    char_csv = os.path.join(HERE, "char_level_metrics.csv")
    with open(char_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "family", "char",
                                          "codepoint"] + METRIC_COLS)
        w.writeheader()
        w.writerows(char_rows)

    # 字体级聚合（中位数）
    by_font = {}
    for row in char_rows:
        by_font.setdefault(row["file"], []).append(row)
    font_rows = []
    for r in fonts:
        fbase = r["file"]
        items = by_font.get(fbase, [])
        if not items:
            continue
        agg = {c: round(float(np.median([x[c] for x in items])), 4)
               for c in METRIC_COLS}
        font_rows.append({
            "file": fbase, "family": r.get("family", ""),
            "units_per_em": r["units_per_em"], "is_variable": r["is_variable"],
            "num_chars_measured": len(items), **agg,
        })

    font_csv = os.path.join(HERE, "font_level_attributes.csv")
    fcols = ["file", "family", "units_per_em", "is_variable",
             "num_chars_metric"] + METRIC_COLS
    with open(font_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols)
        w.writeheader()
        for row in font_rows:
            w.writerow({k: row.get(k if k != "num_chars_metric"
                                   else "num_chars_measured", "")
                        for k in fcols})

    # JSON 备份
    with open(os.path.join(HERE, "font_level_attributes.json"), "w",
              encoding="utf-8") as f:
        json.dump(font_rows, f, ensure_ascii=False, indent=2)

    print("fonts:", len(fonts), "rep chars:", len(REP_CHARS),
          "char-level rows:", len(char_rows))
    print("missing (font,char) pairs:", len(missing))
    if missing:
        for x in missing[:20]:
            print("  MISSING", x)
    print("char csv ->", char_csv)
    print("font csv ->", font_csv)


if __name__ == "__main__":
    main()

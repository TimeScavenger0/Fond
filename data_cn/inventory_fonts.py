# -*- coding: utf-8 -*-
"""盘点系统字体：用 fontTools 核验每个 ttf 的真实中文覆盖与字体元信息。
只做只读测量，不修改任何字体文件。输出 JSON 供后续管线使用。"""
import os
import json
from fontTools.ttLib import TTFont

FONTS_DIR = r"C:\Windows\Fonts"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font_inventory.json")

# 用于核验中文覆盖的测试字符（覆盖常用字与不同结构）
TEST_CHARS = ["永", "人", "国", "字", "体", "中", "文", "永", "鑫"]
TEST_CPS = [ord(c) for c in dict.fromkeys(TEST_CHARS)]


def family_name(font):
    """优先取 name 表中的英文/中文家族名。"""
    name = font["name"]
    for nid in (1, 4, 16):
        try:
            record = name.getBestFamilyName() if nid == 1 else None
            if record:
                return record
        except Exception:
            pass
    # 兜底：遍历 nameID=1
    for rec in name.names:
        if rec.nameID == 1:
            try:
                return rec.toUnicode()
            except Exception:
                continue
    return ""


def main():
    rows = []
    for fn in sorted(os.listdir(FONTS_DIR)):
        if not fn.lower().endswith(".ttf"):
            continue
        path = os.path.join(FONTS_DIR, fn)
        row = {"file": fn, "path": path}
        try:
            font = TTFont(path, fontNumber=0, lazy=True)
        except Exception as e:
            row.update(ok=False, error="open:%s" % e)
            rows.append(row)
            continue

        # cmap 覆盖
        cmap = font.getBestCmap() or {}
        covered = [cp for cp in TEST_CPS if cp in cmap]
        row["num_glyphs"] = int(font["maxp"].numGlyphs)
        row["units_per_em"] = int(font["head"].unitsPerEm)
        row["post_italic_angle"] = float(getattr(font["post"], "italicAngle", 0.0))
        row["is_variable"] = "fvar" in font
        if row["is_variable"]:
            try:
                row["fvar_axes"] = [a.axisTag for a in font["fvar"].axes]
            except Exception:
                row["fvar_axes"] = []
        try:
            row["family"] = family_name(font)
        except Exception:
            row["family"] = ""
        # 中文覆盖：测试码点命中数与比例
        row["cn_test_hits"] = len(covered)
        row["cn_test_total"] = len(TEST_CPS)
        row["has_chinese"] = len(covered) == len(TEST_CPS)
        # 全量 cmap 中 CJK 统一表意文字数量
        cjk = [cp for cp in cmap if 0x4E00 <= cp <= 0x9FFF]
        row["cjk_glyph_count"] = len(cjk)
        row["ok"] = True
        font.close()
        rows.append(row)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    cn = [r for r in rows if r.get("has_chinese")]
    print("total ttf scanned:", len(rows))
    print("fonts with full Chinese coverage:", len(cn))
    for r in cn:
        print("  %-20s cjk=%-6d upm=%-5d italic=%6s var=%s family=%s" % (
            r["file"], r["cjk_glyph_count"], r["units_per_em"],
            r["post_italic_angle"], r["is_variable"], r.get("family", "")))
    failed = [r for r in rows if not r.get("ok")]
    if failed:
        print("failed to open:", [r["file"] for r in failed])
    print("json ->", OUT_JSON)


if __name__ == "__main__":
    main()

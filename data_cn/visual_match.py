# -*- coding: utf-8 -*-
"""跨语言视觉最近邻匹配（fontclip 环境，纯 CPU，只读测量）。

用中文字体和 148 个标注拉丁字体渲染同一组拉丁字符，标准化后计算
像素 L1 距离（不同像素占比，越小越相似），输出每个中文字体的 top-k。
"""
import os
import json
import numpy as np

from geom_annotate import render_glyph

HERE = os.path.dirname(os.path.abspath(__file__))
ATTR = r"D:\VScode\VScodeProjects\Fond\attributeData"
GW = r"D:\VScode\VScodeProjects\Fond\gwfonts"
TOP_K = 5

# 结构代表拉丁字符：尖顶/直竖/圆/对称/小写
MATCH_CHARS = ["A", "H", "O", "E", "D", "M", "S", "n"]


def font_feature(font_path):
    """渲染多个字符，返回展平的 0/1 特征向量列表。"""
    vecs = []
    for ch in MATCH_CHARS:
        b = render_glyph(ch, font_path)
        if b is None:
            return None
        vecs.append((b > 0).astype(np.uint8).reshape(-1))
    return vecs


def distance(feat_a, feat_b):
    """多字符平均像素差异比例。"""
    ds = [np.mean(a != b) for a, b in zip(feat_a, feat_b)]
    return float(np.mean(ds))


def main():
    with open(os.path.join(ATTR, "all_font_to_attribute_values.json"),
              "r", encoding="utf-8") as f:
        attr_data = json.load(f)
    with open(os.path.join(HERE, "font_inventory.json"),
              "r", encoding="utf-8") as f:
        inv = json.load(f)
    cn_fonts = [r for r in inv if r.get("has_chinese")]

    # 拉丁标注字体特征
    print("rendering labeled Latin fonts...")
    latin_feats = {}
    for name in sorted(attr_data.keys()):
        ttf = os.path.join(GW, name + ".ttf")
        feat = font_feature(ttf)
        if feat is not None:
            latin_feats[name] = feat
    print("Latin features:", len(latin_feats))

    results = []
    for r in cn_fonts:
        fbase, fpath = r["file"], r["path"]
        feat = font_feature(fpath)
        if feat is None:
            continue
        scored = sorted(((distance(feat, lf), name)
                         for name, lf in latin_feats.items()))
        top = [{"rank": i + 1, "font": name, "distance": round(d, 4)}
               for i, (d, name) in enumerate(scored[:TOP_K])]
        results.append({"file": fbase, "family": r.get("family", ""),
                        "top_matches": top,
                        "all_distances": {name: round(d, 4)
                                          for d, name in scored}})
        print("%-20s best=%s d=%.4f" % (fbase, top[0]["font"],
                                        top[0]["distance"]))

    out = os.path.join(HERE, "visual_match.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\njson ->", out)


if __name__ == "__main__":
    main()

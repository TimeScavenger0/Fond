# -*- coding: utf-8 -*-
"""构建中文 LoRA 训练数据集（fontclip 环境，纯 CPU）。

输出到 data_cn/dataset/：
  cn_images/<font_key>.png   单字网格（上）+ 整段短文（下）合成一张近正方形白底图
  cn_font_to_attribute_values.json 及 train/validation/test 划分
  dataset_meta.json
留空属性（R²<0.4）不写入 JSON，故不参与 prompt 采样；字体级划分防泄漏。
"""
import os
import csv
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geom_annotate import render_glyph
from build_annotations import REP_CHARS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset")
IMG_DIR = os.path.join(OUT, "cn_images")

# 单字网格布局
CELL = 120
COLS = 8
# 整段中文短文（固定，常用字、结构多样、排版整齐）
CN_PARAGRAPH = ["春眠不觉晓", "处处闻啼鸟", "夜来风雨声", "花落知多少"]
PARA_FONT_SIZE = 90
SEED = 123


def font_key_of(file_name):
    return os.path.splitext(file_name)[0]


def build_grid(font_path):
    rows = int(np.ceil(len(REP_CHARS) / float(COLS)))
    grid = Image.new("L", (COLS * CELL, rows * CELL), 255)
    for i, ch in enumerate(REP_CHARS):
        b = render_glyph(ch, font_path, canvas=256, target_h=200)
        if b is None:
            continue
        # render_glyph 返回黑底白字（笔画=255），反相为白底黑字，与短文极性一致
        glyph = Image.fromarray(255 - b)
        g = glyph.resize((CELL - 16, CELL - 16), Image.LANCZOS)
        r, c = divmod(i, COLS)
        grid.paste(g, (c * CELL + 8, r * CELL + 8))
    return grid


def build_paragraph(font_path):
    font = ImageFont.truetype(font_path, PARA_FONT_SIZE)
    try:
        if hasattr(font, "set_variation_by_axes"):
            font.set_variation_by_axes([a.defaultValue
                                        for a in font.get_variation_axes()])
    except Exception:
        pass
    line_w = max(font.getbbox(line)[2] for line in CN_PARAGRAPH)
    line_h = PARA_FONT_SIZE + 18
    w = line_w + 80
    h = line_h * len(CN_PARAGRAPH) + 40
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(CN_PARAGRAPH):
        lw = font.getbbox(line)[2]
        d.text(((w - lw) // 2, 20 + i * line_h), line, font=font, fill=0)
    return img


def compose(font_path):
    grid = build_grid(font_path)
    para = build_paragraph(font_path)
    w = max(grid.width, para.width)
    # 统一宽度，居中
    def pad(im):
        c = Image.new("L", (w, im.height), 255)
        c.paste(im, ((w - im.width) // 2, 0))
        return c
    g, p = pad(grid), pad(para)
    out = Image.new("L", (w, g.height + p.height + 20), 255)
    out.paste(g, (0, 0))
    out.paste(p, (0, g.height + 20))
    return out


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(os.path.join(HERE, "font_inventory.json"),
              "r", encoding="utf-8") as f:
        inventory = [r for r in json.load(f) if r.get("has_chinese")]

    # 读取伪标签（空值跳过）
    labels = {}
    with open(os.path.join(HERE, "cn_37attr_pseudo_labels.csv"),
              "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = font_key_of(r["file"])
            labels[key] = {a: r[a] for a in r
                           if a not in ("file", "family") and r[a] != ""}

    all_attr = {}
    rendered = []
    for rec in inventory:
        key = font_key_of(rec["file"])
        img = compose(rec["path"])
        img.save(os.path.join(IMG_DIR, key + ".png"))
        all_attr[key] = labels[key]
        rendered.append((key, img.size))

    keys = list(all_attr.keys())
    random.Random(SEED).shuffle(keys)
    n = len(keys)
    splits = {
        "train": keys[: int(n * 0.6)],
        "validation": keys[int(n * 0.6): int(n * 0.8)],
        "test": keys[int(n * 0.8):],
    }
    for name, ks in splits.items():
        with open(os.path.join(OUT, name + "_font_to_attribute_values.json"),
                  "w", encoding="utf-8") as f:
            json.dump({k: all_attr[k] for k in ks}, f,
                      ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "cn_font_to_attribute_values.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_attr, f, ensure_ascii=False, indent=2)

    meta = {"seed": SEED, "fonts": n,
            "splits": {k: len(v) for k, v in splits.items()},
            "attributes_per_font": {k: len(v) for k, v in all_attr.items()},
            "image_sizes": {k: list(s) for k, s in rendered},
            "paragraph": CN_PARAGRAPH, "grid_chars": len(REP_CHARS)}
    with open(os.path.join(OUT, "dataset_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("fonts rendered:", n)
    print("splits:", meta["splits"])
    print("images ->", IMG_DIR)
    print("sample size:", rendered[0])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""生成中文字体与 top-1 匹配拉丁字体的字符对比图（fontclip 环境，只读）。"""
import os
import json
from PIL import Image, ImageDraw, ImageFont

from geom_annotate import render_glyph
from visual_match import MATCH_CHARS, GW

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 72
LABEL_W = 230
PAD = 6

with open(os.path.join(HERE, "visual_match.json"), "r", encoding="utf-8") as f:
    results = json.load(f)
with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
    inv = json.load(f)
path_by_file = {r["file"]: r["path"] for r in inv
                if r.get("has_chinese")}

try:
    label_font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 15)
except Exception:
    label_font = ImageFont.load_default()

n = len(results)
row_h = CELL + PAD
width = LABEL_W + (CELL + PAD) * len(MATCH_CHARS) + LABEL_W + 40
canvas = Image.new("RGB", (width, row_h * n + 20), "white")
draw = ImageDraw.Draw(canvas)

for ri, res in enumerate(results):
    fbase = res["file"]
    top = res["top_matches"][0]
    latin_name, dist = top["font"], top["distance"]
    y = ri * row_h + 10
    draw.text((6, y + CELL // 2 - 10), fbase[:22], fill="black",
              font=label_font)

    cn_path = path_by_file[fbase]
    latin_path = os.path.join(GW, latin_name + ".ttf")
    x = LABEL_W
    for ch in MATCH_CHARS:
        b = render_glyph(ch, cn_path)
        im = Image.fromarray(b).resize((CELL, CELL))
        canvas.paste(im, (x, y))
        x += CELL + PAD

    draw.text((x + 4, y + 8), latin_name[:20], fill="blue",
              font=label_font)
    draw.text((x + 4, y + 30), "d=%.3f" % dist, fill="red",
              font=label_font)
    x += LABEL_W + 40

    for ch in MATCH_CHARS:
        b = render_glyph(ch, latin_path)
        im = Image.fromarray(b).resize((CELL, CELL))
        canvas.paste(im, (x, y))
        x += CELL + PAD

    if ri % 2 == 0:
        draw.rectangle([0, y - 4, width, y + CELL + 4], outline=None)

out = os.path.join(HERE, "match_compare.png")
canvas.save(out)
print("saved", out, "size", canvas.size)

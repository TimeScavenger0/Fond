# -*- coding: utf-8 -*-
"""固定字号渲染横画，可视化笔画末端衬线（fontclip 环境，只读）。

不做 bbox 高度拉伸，保持笔画真实比例；渲染"一"放大，观察右端三角鳞饰。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = r"C:\Windows\Fonts"
TEST = [
    ("STSONG.TTF", "song"),
    ("STFANGSO.TTF", "fangsong"),
    ("simhei.ttf", "hei"),
    ("STKAITI.TTF", "kai"),
    ("SIMYOU.TTF", "youyuan"),
    ("方正粗黑宋简体.ttf", "cuheisong"),
]
SIZE = 260
CANVAS = 320


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    d = ImageDraw.Draw(im)
    d.text((CANVAS // 2, CANVAS // 2), ch, font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


# 拼接"一"的放大图
cell = 200
strip = Image.new("RGB", (cell * len(TEST), cell + 24), "white")
dd = ImageDraw.Draw(strip)
for i, (fbase, tag) in enumerate(TEST):
    b = render_raw("一", os.path.join(FONT_DIR, fbase))
    ys, xs = np.where(b > 0)
    # 裁剪横画并留边
    y0, y1 = max(ys.min() - 6, 0), min(ys.max() + 7, CANVAS)
    x0, x1 = max(xs.min() - 6, 0), min(xs.max() + 7, CANVAS)
    crop = Image.fromarray(b[y0:y1, x0:x1] * 255)
    crop = crop.resize((cell, cell), Image.NEAREST)
    strip.paste(crop, (i * cell, 24))
    dd.text((i * cell + 4, 4), tag, fill="black")

out = os.path.join(HERE, "serif_tip_compare.png")
strip.save(out)
print("saved", out, strip.size)

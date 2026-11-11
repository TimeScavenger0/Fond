# -*- coding: utf-8 -*-
"""分析横画右上轮廓曲线，区分尖三角与平滑弯曲（fontclip 环境，只读）。"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = r"C:\Windows\Fonts"
TEST = [
    ("STSONG.TTF", "song"),
    ("STKAITI.TTF", "kai"),
    ("simhei.ttf", "hei"),
    ("方正粗黑宋简体.ttf", "cuheisong"),
]
SIZE, CANVAS = 260, 320


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def top_curve(b):
    xs = np.where(b > 0)[1]
    x0, x1 = xs.min(), xs.max()
    curve = {}
    for x in range(x0, x1 + 1):
        col = np.where(b[:, x] > 0)[0]
        if len(col):
            curve[x] = col.min()
    return curve, x0, x1


for fbase, tag in TEST:
    b = render_raw("一", os.path.join(FONT_DIR, fbase))
    curve, x0, x1 = top_curve(b)
    w = x1 - x0
    # 右端 25% 区域的上轮廓，每 5% 采样
    print("\n=== %s ===" % tag)
    vals = []
    for frac in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00]:
        x = x0 + int(frac * w)
        if x in curve:
            vals.append((frac, curve[x]))
    base = np.median([curve[x] for x in range(x0 + int(0.4 * w),
                                             x0 + int(0.65 * w))])
    print("  body top baseline y = %.1f" % base)
    for frac, y in vals:
        print("  x frac %.2f top_y=%6.1f  lift=%.1f" % (frac, y, base - y))

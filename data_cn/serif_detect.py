# -*- coding: utf-8 -*-
"""衬线指数：横画右端三角尖高出主体上边缘的高度 / 笔画宽度（fontclip 环境）。

先在 6 个代表字体上验证区分度，打印中间量。
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
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def serif_index(b):
    ys, xs = np.where(b > 0)
    x0, x1 = xs.min(), xs.max()
    w = x1 - x0
    mx0, mx1 = x0 + int(0.40 * w), x0 + int(0.70 * w)

    tops, widths = [], []
    for x in range(mx0, mx1):
        col = np.where(b[:, x] > 0)[0]
        if len(col):
            tops.append(col.min())
            widths.append(col.max() - col.min() + 1)
    top_base = np.median(tops)
    body_w = np.median(widths)

    rx0 = x0 + int(0.88 * w)
    rys = np.where(b[:, rx0:x1 + 1] > 0)[0]
    tip_y = rys.min()
    tip_h = top_base - tip_y
    return float(tip_h / body_w), float(tip_h), float(body_w)


for fbase, tag in TEST:
    b = render_raw("一", os.path.join(FONT_DIR, fbase))
    idx, tip_h, body_w = serif_index(b)
    print("%-10s serif_index=%6.3f (tip_h=%6.2f body_w=%6.2f)" % (
        tag, idx, tip_h, body_w))

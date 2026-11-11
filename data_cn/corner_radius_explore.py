# -*- coding: utf-8 -*-
"""转折圆角半径检测：口/国 外轮廓四角拟合圆（fontclip 环境，只读）。"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = r"C:\Windows\Fonts"
TEST = [
    ("simhei.ttf", "hei"),
    ("SIMYOU.TTF", "youyuan"),
    ("STSONG.TTF", "song"),
    ("STKAITI.TTF", "kai"),
    ("FZYTK.TTF", "yao"),
    ("STHUPO.TTF", "hupo"),
]
SIZE, CANVAS = 260, 320


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def fit_circle(points):
    # 最小二乘拟合圆（代数法）
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(c + cx * cx + cy * cy, 0))
    return r


def corner_radii(b):
    conts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    c = max(conts, key=cv2.contourArea).reshape(-1, 2)
    x0, y0 = c[:, 0].min(), c[:, 1].min()
    x1, y1 = c[:, 0].max(), c[:, 1].max()
    w, h = x1 - x0, y1 - y0
    radii = []
    # 四个角：取靠近各角落、落在边长 35% 内的点
    corners = [
        (x0, y0, c[(c[:, 0] < x0 + 0.35 * w) & (c[:, 1] < y0 + 0.35 * h)]),
        (x1, y0, c[(c[:, 0] > x1 - 0.35 * w) & (c[:, 1] < y0 + 0.35 * h)]),
        (x0, y1, c[(c[:, 0] < x0 + 0.35 * w) & (c[:, 1] > y1 - 0.35 * h)]),
        (x1, y1, c[(c[:, 0] > x1 - 0.35 * w) & (c[:, 1] > y1 - 0.35 * h)]),
    ]
    for _, _, pts in corners:
        if len(pts) >= 5:
            radii.append(fit_circle(pts))
    return np.array(radii), (w, h)


for fbase, tag in TEST:
    fp = os.path.join(FONT_DIR, fbase)
    for ch in ["口", "国"]:
        b = render_raw(ch, fp)
        radii, (w, h) = corner_radii(b)
        norm = radii / min(w, h)
        print("%-9s %s  r=%s  r/side=%s" % (
            tag, ch, np.round(radii, 1), np.round(norm, 3)))
    print()

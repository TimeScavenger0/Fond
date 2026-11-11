# -*- coding: utf-8 -*-
"""弧长曲率曲线可视化：口 外轮廓（fontclip 环境，只读）。"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = r"C:\Windows\Fonts"
TEST = [("simhei.ttf", "hei"), ("SIMYOU.TTF", "youyuan"),
        ("STSONG.TTF", "song"), ("STKAITI.TTF", "kai")]
SIZE, CANVAS = 260, 320


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def resample_contour(b, n=240):
    conts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    c = max(conts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    d = d / d[-1]
    grid = np.linspace(0, 1, n, endpoint=False)
    pts = np.c_[np.interp(grid, d, c[:, 0]),
                np.interp(grid, d, c[:, 1])]
    return pts


def curvature(pts, k=6):
    n = len(pts)
    t1 = pts - np.roll(pts, k, axis=0)
    t2 = np.roll(pts, -k, axis=0) - pts
    a = np.arctan2(t1[:, 1], t1[:, 0])
    bb = np.arctan2(t2[:, 1], t2[:, 0])
    dtheta = (bb - a + np.pi) % (2 * np.pi) - np.pi
    ds = np.linalg.norm(t1, axis=1) + np.linalg.norm(t2, axis=1)
    return 2 * dtheta / np.maximum(ds, 1e-6)


fig, axes = plt.subplots(len(TEST), 1, figsize=(10, 9), sharex=True)
for ax, (fbase, tag) in zip(axes, TEST):
    b = render_raw("口", os.path.join(FONT_DIR, fbase))
    pts = resample_contour(b)
    kappa = curvature(pts)
    ax.plot(np.arange(len(kappa)), kappa, lw=1.2)
    ax.set_title(tag)
    ax.set_ylabel("curvature")
    ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(HERE, "curvature_curves.png")
plt.savefig(out, dpi=110)
print("saved", out)

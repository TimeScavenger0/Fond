# -*- coding: utf-8 -*-
"""衬线检测候选特征探索：横细竖粗（fontclip 环境，只读）。

骨架点按切线方向分类水平/垂直，取距离场宽度，比较竖宽/横宽。
宋体横细竖粗比值大，黑体均匀接近 1。
"""
import os
import numpy as np
import cv2
from skimage.morphology import skeletonize

from geom_annotate import render_glyph

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = [
    ("STSONG.TTF", "song"),
    ("STFANGSO.TTF", "fangsong"),
    ("simhei.ttf", "hei"),
    ("STKAITI.TTF", "kai"),
    ("STHUPO.TTF", "hupo"),
    ("FZYTK.TTF", "yao"),
]
CHARS = ["永", "木", "林", "国", "明"]


def stroke_width_ratio(binary):
    b = (binary > 0).astype(np.uint8)
    dist = cv2.distanceTransform(b, cv2.DIST_L2, 3)
    skel = skeletonize(b > 0).astype(np.uint8)
    # 骨架切线方向：对骨架图 Sobel，切线垂直于梯度
    gx = cv2.Sobel(skel, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(skel, cv2.CV_64F, 0, 1, ksize=3)
    ys, xs = np.where(skel > 0)
    horiz_w, vert_w = [], []
    for y, x in zip(ys, xs):
        vx, vy = -gy[y, x], gx[y, x]  # 切线方向
        n = (vx * vx + vy * vy) ** 0.5
        if n < 1e-6:
            continue
        c = abs(vx) / n  # 切线水平分量
        w = dist[y, x]
        if c > 0.866:      # 与水平夹角 <30°
            horiz_w.append(w)
        elif c < 0.5:      # 与水平夹角 >60°
            vert_w.append(w)
    if len(horiz_w) < 5 or len(vert_w) < 5:
        return np.nan, np.nan, np.nan
    h = float(np.median(horiz_w))
    v = float(np.median(vert_w))
    return v / h, h, v


for fbase, tag in TEST:
    fp = os.path.join(r"C:\Windows\Fonts", fbase)
    print("\n=== %s (%s) ===" % (fbase, tag))
    ratios = []
    for ch in CHARS:
        b = render_glyph(ch, fp)
        r, h, v = stroke_width_ratio(b)
        ratios.append(r)
        print("  %s vert/horiz width ratio=%6.3f (h=%.2f v=%.2f)" % (ch, r, h, v))
    print("  median ratio = %.3f" % np.nanmedian(ratios))

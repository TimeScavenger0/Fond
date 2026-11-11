# -*- coding: utf-8 -*-
"""棱角检测探索：轮廓顶点内角与锐角占比（fontclip 环境，只读）。"""
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
CHARS = ["永", "木", "国", "口", "林"]


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def corner_angles(b):
    conts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in conts:
        peri = cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, 0.03 * peri, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        for i in range(len(poly)):
            p = poly[i]
            a = poly[i - 1] - p
            q = poly[(i + 1) % len(poly)] - p
            na, nq = np.linalg.norm(a), np.linalg.norm(q)
            if na < 1e-6 or nq < 1e-6:
                continue
            cosv = np.clip(np.dot(a, q) / (na * nq), -1, 1)
            angles.append(np.degrees(np.arccos(cosv)))
    return np.array(angles)


for fbase, tag in TEST:
    fp = os.path.join(FONT_DIR, fbase)
    all_ang = []
    for ch in CHARS:
        b = render_raw(ch, fp)
        all_ang.append(corner_angles(b))
    ang = np.concatenate(all_ang)
    sharp = np.mean(ang < 80)
    print("%-10s n=%4d mean_angle=%6.1f  sharp(<80)=%.3f  median=%6.1f" % (
        tag, len(ang), ang.mean(), sharp, np.median(ang)))

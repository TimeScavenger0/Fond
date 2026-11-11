# -*- coding: utf-8 -*-
"""棱角指数：曲率峰尖锐度 = 峰值 / 半高宽（fontclip 环境，只读）。

窄高峰（直角）尖锐度大，宽低峰（圆角）尖锐度小。先在代表字体验证。
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = r"C:\Windows\Fonts"
TEST = [("simhei.ttf", "hei"), ("SIMYOU.TTF", "youyuan"),
        ("STSONG.TTF", "song"), ("STKAITI.TTF", "kai"),
        ("FZYTK.TTF", "yao"), ("STHUPO.TTF", "hupo")]
SIZE, CANVAS = 260, 320
CHARS = ["口", "国", "回"]


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def resample(b, n=240):
    conts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    c = max(conts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    d = d / d[-1]
    grid = np.linspace(0, 1, n, endpoint=False)
    return np.c_[np.interp(grid, d, c[:, 0]), np.interp(grid, d, c[:, 1])]


def curvature(pts, k=6):
    t1 = pts - np.roll(pts, k, axis=0)
    t2 = np.roll(pts, -k, axis=0) - pts
    a = np.arctan2(t1[:, 1], t1[:, 0])
    bb = np.arctan2(t2[:, 1], t2[:, 0])
    dtheta = (bb - a + np.pi) % (2 * np.pi) - np.pi
    ds = np.linalg.norm(t1, axis=1) + np.linalg.norm(t2, axis=1)
    return np.abs(2 * dtheta / np.maximum(ds, 1e-6))


def peak_sharpness(kappa):
    n = len(kappa)
    pk = np.r_[kappa[-1], kappa, kappa[0]]
    peaks = []
    for i in range(1, n + 1):
        if pk[i] >= pk[i - 1] and pk[i] > pk[i + 1] and pk[i] > 0.02:
            # 半高宽
            half = 0.5 * pk[i]
            l = i
            while l > 0 and pk[l] > half:
                l -= 1
            r = i
            while r < len(pk) - 1 and pk[r] > half:
                r += 1
            fwhm = max(r - l, 1)
            peaks.append(pk[i] / fwhm)
    return np.array(peaks)


for fbase, tag in TEST:
    fp = os.path.join(FONT_DIR, fbase)
    sharps = []
    for ch in CHARS:
        b = render_raw(ch, fp)
        kappa = curvature(resample(b))
        sharps.append(peak_sharpness(kappa))
    s = np.concatenate(sharps)
    print("%-9s npeaks=%3d median_sharpness=%7.4f mean=%7.4f" % (
        tag, len(s), np.median(s), s.mean()))

# -*- coding: utf-8 -*-
"""全量棱角检测：多字符显著曲率峰峰值中位数（fontclip 环境，只读）。"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from angular_detect import resample, curvature

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE, CANVAS = 260, 320
CHARS = ["口", "国", "回", "同", "问", "区", "永", "木", "林", "明"]


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def peak_values(kappa):
    n = len(kappa)
    pk = np.r_[kappa[-1], kappa, kappa[0]]
    out = []
    for i in range(1, n + 1):
        if pk[i] >= pk[i - 1] and pk[i] > pk[i + 1] and pk[i] > 0.02:
            out.append(pk[i])
    return out


with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
    inv = json.load(f)
fonts = [r for r in inv if r.get("has_chinese")]

results = {}
print("%-20s %12s %8s" % ("file", "angular_peak", "npeaks"))
for r in fonts:
    peaks = []
    for ch in CHARS:
        b = render_raw(ch, r["path"])
        peaks.extend(peak_values(curvature(resample(b))))
    peaks = np.array(peaks)
    val = float(np.median(peaks))
    results[r["file"]] = {"angular_peak": round(val, 4),
                          "n_peaks": len(peaks)}
    print("%-20s %12.4f %8d" % (r["file"], val, len(peaks)))

out = os.path.join(HERE, "angular_metrics.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\njson ->", out)

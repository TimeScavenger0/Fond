# -*- coding: utf-8 -*-
"""衬线检测多字符稳健性：一/二/三 的独立横段（fontclip 环境，只读）。"""
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE, CANVAS = 260, 320


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def analyze_segment(b):
    xs = np.where(b > 0)[1]
    x0, x1 = xs.min(), xs.max()
    curve = {x: np.where(b[:, x] > 0)[0].min()
             for x in range(x0, x1 + 1) if b[:, x].any()}
    w = x1 - x0

    def xat(frac):
        return x0 + int(frac * w)

    body_cols = [x for x in range(xat(0.40), xat(0.65)) if x in curve]
    base = np.median([curve[x] for x in body_cols])
    body_w = np.median([(np.where(b[:, x] > 0)[0].ptp() + 1)
                        for x in body_cols])
    lift = {x: base - curve[x] for x in curve if x >= xat(0.65)}
    peak = max(lift.values())
    xs_r = sorted(lift.keys())
    dx, max_slope = 2, 0.0
    for x in xs_r:
        if x + dx in lift:
            max_slope = max(max_slope, (lift[x + dx] - lift[x]) / dx)
    return peak / body_w, max_slope / body_w


def segments(b):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 3 * h and area > 30:  # 只保留横向条段
            seg = b[max(y - 4, 0):y + h + 5,
                    max(x - 4, 0):x + w + 5]
            out.append(seg)
    return out


with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
    inv = json.load(f)
fonts = [r for r in inv if r.get("has_chinese")]

print("%-20s %12s %12s %8s" % ("file", "peak_lift", "max_slope", "nseg"))
results = {}
for r in fonts:
    peaks, slopes = [], []
    for ch in ["一", "二", "三"]:
        b = render_raw(ch, r["path"])
        for seg in segments(b):
            p, s = analyze_segment(seg)
            peaks.append(p); slopes.append(s)
    results[r["file"]] = {
        "peak_lift_n": round(float(np.median(peaks)), 3),
        "max_slope_n": round(float(np.median(slopes)), 3),
        "n_seg": len(peaks),
    }
    print("%-20s %12.3f %12.3f %8d" % (
        r["file"], np.median(peaks), np.median(slopes), len(peaks)))

out = os.path.join(HERE, "serif_robust.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\njson ->", out)

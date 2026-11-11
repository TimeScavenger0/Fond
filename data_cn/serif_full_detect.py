# -*- coding: utf-8 -*-
"""全量衬线检测：上轮廓右端的突变尖峰（fontclip 环境，只读）。

输出三个候选量（归一化到笔画宽度）：
  body_lift  主体 frac0.72 处抬升（楷体大、宋体近 0）
  peak_lift  右端最大抬升
  max_slope  上轮廓最大局部上升斜率（刻画突变尖，宋体/粗黑宋大、楷体小）
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = r"C:\Windows\Fonts"
SIZE, CANVAS = 260, 320


def render_raw(ch, font_path):
    im = Image.new("L", (CANVAS, CANVAS), 255)
    f = ImageFont.truetype(font_path, SIZE)
    ImageDraw.Draw(im).text((CANVAS // 2, CANVAS // 2), ch,
                            font=f, fill=0, anchor="mm")
    return (np.array(im) < 128).astype(np.uint8)


def analyze(b):
    xs = np.where(b > 0)[1]
    x0, x1 = xs.min(), xs.max()
    curve = {}
    for x in range(x0, x1 + 1):
        col = np.where(b[:, x] > 0)[0]
        if len(col):
            curve[x] = col.min()
    w = x1 - x0

    def xat(frac):
        return x0 + int(frac * w)

    body_cols = [x for x in range(xat(0.40), xat(0.65)) if x in curve]
    base = np.median([curve[x] for x in body_cols])
    body_w = np.median([
        (np.where(b[:, x] > 0)[0].ptp() + 1) for x in body_cols])

    lift = {x: base - curve[x] for x in curve if x >= xat(0.65)}
    xs_r = sorted(lift.keys())
    peak_lift = max(lift.values())
    body_lift = lift.get(xat(0.72), 0)

    dx = 2
    max_slope = 0.0
    for x in xs_r:
        if (x + dx) in lift:
            max_slope = max(max_slope, (lift[x + dx] - lift[x]) / dx)

    return {
        "body_lift_n": round(float(body_lift / body_w), 3),
        "peak_lift_n": round(float(peak_lift / body_w), 3),
        "max_slope_n": round(float(max_slope / body_w), 3),
        "body_w": round(float(body_w), 1),
    }


with open(os.path.join(HERE, "font_inventory.json"), "r", encoding="utf-8") as f:
    inv = json.load(f)
fonts = [r for r in inv if r.get("has_chinese")]

results = {}
print("%-20s %10s %10s %10s" % ("file", "body_lift", "peak_lift", "max_slope"))
for r in fonts:
    b = render_raw("一", r["path"])
    a = analyze(b)
    results[r["file"]] = a
    print("%-20s %10.3f %10.3f %10.3f" % (
        r["file"], a["body_lift_n"], a["peak_lift_n"], a["max_slope_n"]))

out = os.path.join(HERE, "serif_metrics.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\njson ->", out)

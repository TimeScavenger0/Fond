# -*- coding: utf-8 -*-
"""中文字体几何自动标注管线（fontclip 环境，纯 CPU，只读测量）。

标准化渲染：每个字符统一目标高度、居中到固定画布，保证跨字体可比。
四个可靠几何指标：
  slant_ecc      整体倾斜角（cv2.fitEllipse，度；正=右倾）
  slant_regress  按行重心回归的倾斜角（度），用于交叉验证
  width_ratio    前景包围盒宽 / 高
  blackness      前景像素 / 画布像素
  stroke_halfw   距离场中位半宽（像素），约为笔画宽度的一半
"""
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CANVAS = 256
TARGET_H = 180
RENDER_SIZE = 300
BIN_THRESH = 128

# 小字符集：覆盖独体、包围、对称、交叉等结构。
# 不含单笔画的"一"：退化结构会让 bbox/距离场/行回归全部失真。
PROBE_CHARS = ["永", "人", "国", "口", "木", "水"]


def render_glyph(char, font_path, canvas=CANVAS, target_h=TARGET_H):
    """高分辨率渲染 -> 按 bbox 等比缩放到统一高度 -> 居中 -> 二值化。"""
    font = ImageFont.truetype(font_path, RENDER_SIZE)
    # 可变字体：显式取默认（Regular）实例
    try:
        if hasattr(font, "set_variation_by_axes"):
            axes = [a.defaultValue for a in font.get_variation_axes()]
            font.set_variation_by_axes(axes)
    except Exception:
        pass

    layer = Image.new("L", (RENDER_SIZE * 2, RENDER_SIZE * 2), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(layer)
    d.text((RENDER_SIZE // 2, RENDER_SIZE // 2), char, font=font, fill=255)
    bbox = layer.getbbox()
    if bbox is None:
        return None
    glyph = layer.crop(bbox)
    gw, gh = glyph.size
    scale = target_h / float(gh)
    new_w = max(1, int(round(gw * scale)))
    glyph = glyph.resize((new_w, target_h), Image.LANCZOS)

    out = Image.new("L", (canvas, canvas), 0)
    ox = (canvas - new_w) // 2
    oy = (canvas - target_h) // 2
    out.paste(glyph, (ox, oy))
    arr = np.array(out, dtype=np.uint8)
    binary = (arr >= BIN_THRESH).astype(np.uint8) * 255
    return binary


def _foreground_pts(binary):
    ys, xs = np.where(binary > 0)
    return np.column_stack([xs, ys]).astype(np.float32)


def measure_slant_ecc(binary):
    """fitEllipse 主轴角归一化：正=右倾。"""
    pts = _foreground_pts(binary)
    if len(pts) < 5:
        return float("nan")
    (_, _), (ma, MA), angle = cv2.fitEllipse(pts)
    # angle: 主轴与 x 轴夹角 0-180；垂直主导时≈90
    slant = 90.0 - float(angle)
    # 归一化到 (-90, 90]
    while slant > 90:
        slant -= 180
    while slant <= -90:
        slant += 180
    return slant


def measure_slant_regress(binary):
    """逐行前景重心 x 对 y 线性回归，斜率转角度。"""
    ys, xs = np.where(binary > 0)
    if len(ys) < 5:
        return float("nan")
    # 只取前景足够多的行，避免端点噪声
    rows = {}
    for y, x in zip(ys, xs):
        rows.setdefault(int(y), []).append(x)
    yy, xx = [], []
    for y, xl in rows.items():
        if len(xl) >= 3:
            yy.append(y)
            xx.append(float(np.mean(xl)))
    if len(yy) < 8:
        return float("nan")
    # 有效行必须覆盖字形纵向跨度的大部分，否则少量噪声即放大斜率
    y_span = float(max(yy) - min(yy))
    glyph_h = float(ys.max() - ys.min() + 1)
    if y_span < 0.6 * glyph_h:
        return float("nan")
    yy, xx = np.array(yy, float), np.array(xx, float)
    slope = np.polyfit(yy, xx, 1)[0]  # dx/dy
    return float(np.degrees(np.arctan(slope)))


def measure_width_ratio(binary):
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return float("nan")
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    return w / h


def measure_blackness(binary):
    return float((binary > 0).sum()) / float(binary.size)


def measure_stroke_halfw(binary):
    """距离场：前景像素到背景的距离，取中位数即笔画半宽。"""
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    inside = dist[binary > 0]
    if inside.size == 0:
        return float("nan")
    return float(np.median(inside))


def measure_all(binary):
    return {
        "slant_ecc": round(measure_slant_ecc(binary), 4),
        "slant_regress": round(measure_slant_regress(binary), 4),
        "width_ratio": round(measure_width_ratio(binary), 4),
        "blackness": round(measure_blackness(binary), 4),
        "stroke_halfw": round(measure_stroke_halfw(binary), 4),
    }


def main():
    inv_path = os.path.join(HERE, "font_inventory.json")
    with open(inv_path, "r", encoding="utf-8") as f:
        inventory = json.load(f)
    fonts = [r for r in inventory if r.get("has_chinese")]

    debug_dir = os.path.join(HERE, "probe_render")
    os.makedirs(debug_dir, exist_ok=True)
    records = []
    for r in fonts:
        font_path, fbase = r["path"], r["file"]
        for ch in PROBE_CHARS:
            binary = render_glyph(ch, font_path)
            if binary is None:
                continue
            metrics = measure_all(binary)
            records.append({"file": fbase, "family": r.get("family", ""),
                            "char": ch, "cp": "u%x" % ord(ch), **metrics})
            cv2.imwrite(os.path.join(debug_dir, "%s_u%x.png" % (
                os.path.splitext(fbase)[0], ord(ch))), binary)

    out_json = os.path.join(HERE, "probe_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("fonts:", len(fonts), "probe chars:", len(PROBE_CHARS),
          "records:", len(records))
    print("render debug ->", debug_dir)
    print("json ->", out_json)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""位图转矢量桥接管线（fontclip 环境，纯 CPU）。

把 FontDiffuser 生成的 96x96 位图转换为可编辑 SVG：轮廓提取 -> 多边形简化
-> 线段 Path（even-odd 保留内孔）-> diffvg 可微拟合 -> 导出。
支持单图与批量字符处理。
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import pydiffvg
from PIL import Image
from skimage import measure

from b2v_svg import save_svg_full

CANVAS = 96
TOL = 1.0


def _build_paths(bitmap_path, size=CANVAS):
    arr = np.array(Image.open(bitmap_path).convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    shapes = []
    for c in cnts:
        coords = c.reshape(-1, 2)[:, ::-1].astype(np.float64)
        simp = measure.approximate_polygon(coords, tolerance=TOL)
        if np.allclose(simp[0], simp[-1]):
            simp = simp[:-1]
        xy = np.ascontiguousarray(simp[:, ::-1])
        if len(xy) < 2:
            continue
        points = torch.tensor(xy, dtype=torch.float32)
        n = len(xy)
        path = pydiffvg.Path(
            num_control_points=torch.zeros(n, dtype=torch.int32),  # 0=线段
            points=points, is_closed=True, stroke_width=torch.tensor(1.0))
        shapes.append(path)

    if not shapes:
        raise ValueError("no contour in %s" % bitmap_path)
    group = pydiffvg.ShapeGroup(
        shape_ids=torch.arange(len(shapes), dtype=torch.int32),
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        use_even_odd_rule=torch.tensor(True))
    return shapes, [group]


def _render_rgb(shapes, groups, size=CANVAS):
    scene = pydiffvg.RenderFunction.serialize_scene(size, size, shapes, groups)
    img = pydiffvg.RenderFunction.apply(size, size, 2, 2, 0, None, *scene)
    a = img[:, :, 3:4]
    return a * img[:, :, :3] + (1 - a)


def bitmap_to_svg(bitmap_path, out_svg_path, refine_steps=200, size=CANVAS):
    """单张位图 -> 可编辑 SVG。返回指标 dict。"""
    shapes, groups = _build_paths(bitmap_path, size)
    target = torch.from_numpy(
        np.array(Image.open(bitmap_path).convert("RGB"))).float() / 255.0

    point_vars = []
    for path in shapes:
        path.points.requires_grad = True
        point_vars.append(path.points)
    optim = torch.optim.Adam(point_vars, lr=1.0)

    init_loss, final_loss = None, None
    for t in range(refine_steps):
        optim.zero_grad()
        cur = _render_rgb(shapes, groups, size)
        loss = (cur - target).pow(2).mean()
        if init_loss is None:
            init_loss = float(loss.item())
        loss.backward()
        optim.step()
        with torch.no_grad():
            for path in shapes:
                path.points.data[:, 0].clamp_(0, size)
                path.points.data[:, 1].clamp_(0, size)
        final_loss = float(loss.item())

    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
    save_svg_full(out_svg_path, size, size, shapes, groups)
    npt = sum(int(s.points.shape[0]) for s in shapes)
    return {"paths": len(shapes), "points": npt,
            "init_loss": init_loss, "final_loss": final_loss}


def batch_bitmap_to_svg(bitmap_dir, out_dir, manifest, refine_steps=200, size=CANVAS):
    """对 manifest 中 ok 的字符批量转换。返回更新后的记录列表。"""
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for rec in manifest.get("chars", []):
        if rec.get("status") != "ok" or not rec.get("file"):
            records.append({**rec, "svg": None})
            continue
        src = os.path.join(bitmap_dir, rec["file"])
        svg_name = "u%x.svg" % ord(rec["char"])
        dst = os.path.join(out_dir, svg_name)
        try:
            metric = bitmap_to_svg(src, dst, refine_steps=refine_steps, size=size)
            records.append({**rec, "svg": svg_name, **metric})
        except Exception as e:
            records.append({**rec, "svg": None, "svg_error": str(e)})
    return records

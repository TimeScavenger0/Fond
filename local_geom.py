# -*- coding: utf-8 -*-
"""画笔局部几何精修（确定性、可控、可解释）。

对画笔圈内的轮廓控制点，沿距离场（SDF）单位内法线方向做确定性收缩，
实现均匀变细（thinner）。距离场梯度平滑，相邻点移动自然协调，
不产生 CLIP 式高频锯齿，也不会侵入空白；圈外点严格冻结。

支持反向（thicker，沿法线向外扩张）。
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import pydiffvg
import cv2

CANVAS = 200


def render_binary(shapes, groups, size=CANVAS):
    """渲染矢量为二值图：笔画=1，背景=0。"""
    scene = pydiffvg.RenderFunction.serialize_scene(size, size, shapes, groups)
    img = pydiffvg.RenderFunction.apply(size, size, 2, 2, 0, None, *scene)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(size, size, 3) * (1 - img[:, :, 3:4])
    gray = (img[:, :, :3].detach().mean(dim=2).numpy() < 0.5).astype(np.uint8)
    return gray


def build_normal_field(binary):
    """由二值图计算内部距离场及其单位内法线（指向笔画内部）。"""
    dfield = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.gradient(dfield)
    norm = np.sqrt(gx ** 2 + gy ** 2)
    ux = np.where(norm > 1e-6, gx / (norm + 1e-9), 0.0)
    uy = np.where(norm > 1e-6, gy / (norm + 1e-9), 0.0)
    return dfield, ux, uy


def _bilinear(field, x, y, size=CANVAS):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    x0c = min(max(x0, 0), size - 1)
    x1c = min(max(x1, 0), size - 1)
    y0c = min(max(y0, 0), size - 1)
    y1c = min(max(y1, 0), size - 1)
    tx = min(max(x - x0, 0.0), 1.0)
    ty = min(max(y - y0, 0.0), 1.0)
    v = (field[y0c, x0c] * (1 - tx) * (1 - ty) +
         field[y0c, x1c] * tx * (1 - ty) +
         field[y1c, x0c] * (1 - tx) * ty +
         field[y1c, x1c] * tx * ty)
    return v


def local_thin(shapes, mask, amount, mode="thin", size=CANVAS):
    """对 mask 圈内控制点沿距离场内法线做确定性几何变形。

    mode="thin"：沿 +n 向内收缩（变细）；mode="thick"：沿 -n 向外扩张（变粗）。
    amount 为非负移动距离（像素）。返回新的 shapes（深拷贝，不修改输入）。
    """
    groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.arange(len(shapes), dtype=torch.int32),
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        use_even_odd_rule=torch.tensor(True),
    )]
    binary = render_binary(shapes, groups, size)
    dfield, ux, uy = build_normal_field(binary)

    out = []
    for s in shapes:
        ns = pydiffvg.Path(
            num_control_points=s.num_control_points.clone(),
            points=s.points.clone(),
            is_closed=s.is_closed,
            stroke_width=s.stroke_width.clone(),
        )
        pts = ns.points.detach().cpu().numpy()
        for j in range(pts.shape[0]):
            x, y = pts[j, 0], pts[j, 1]
            xi, yi = int(round(x)), int(round(y))
            inside = (0 <= yi < size) and (0 <= xi < size) and bool(mask[yi, xi])
            if not inside:
                continue
            nx = _bilinear(ux, x, y, size)
            ny = _bilinear(uy, x, y, size)
            if mode == "thin":
                # 向内收缩，且不越过笔画中心（半宽峰值），避免两侧交叉
                peak = _normal_peak(dfield, nx, ny, x, y, size)
                m = min(amount, max(peak - 0.5, 0.0))
            else:
                m = -amount  # 向外扩张
            pts[j, 0] = x + nx * m
            pts[j, 1] = y + ny * m
        ns.points = torch.tensor(pts, dtype=torch.float32)
        out.append(ns)
    return out


def _normal_peak(dfield, nx, ny, x, y, size=CANVAS):
    """沿内法线方向扫描距离场，返回笔画中心峰值距离（半宽）。"""
    peak = 0.0
    for t in np.arange(0.5, 12.0, 0.5):
        qx, qy = x + nx * t, y + ny * t
        if not (0 <= qx < size and 0 <= qy < size):
            break
        d = _bilinear(dfield, qx, qy, size)
        if d >= peak:
            peak = d
        else:
            break
    return peak

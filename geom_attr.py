# -*- coding: utf-8 -*-
"""矢量域确定性几何属性变换（属性滑杆的连续强度执行器）。

几何属性（italic / wide 等）在矢量控制点上有明确、可解释的对应：
滑杆值 -> 几何变换参数，严格单调、可逆、CPU 瞬时完成，不依赖 CLIP 隐式优化。

坐标约定：pydiffvg / SVG 坐标系，y 向下。
"""
import torch


def shear_points(points, k, yc):
    """右倾 shear（italic）。k>0 右倾（顶部向右、底部向左）。

    x' = x + k*(yc - y)，y' = y。
    points: [N,2] 张量 (x, y)。
    """
    pts = points.detach().clone()
    x = pts[:, 0]
    y = pts[:, 1]
    nx = x + k * (yc - y)
    return torch.stack([nx, y], dim=1)


def scale_x_points(points, sx, xc):
    """横向缩放（wide / narrow）。sx>1 变宽，sx<1 变窄。"""
    pts = points.detach().clone()
    x = pts[:, 0]
    y = pts[:, 1]
    nx = (x - xc) * sx + xc
    return torch.stack([nx, y], dim=1)


def apply_to_shapes(shapes, fn):
    """对每个 path 的 points 应用纯函数 fn(points)->new_points。"""
    for path in shapes:
        path.points = torch.nn.Parameter(fn(path.points))
    return shapes


def shear_shapes(shapes, k, height):
    yc = height / 2.0
    return apply_to_shapes(shapes, lambda pts: shear_points(pts, k, yc))


def scale_x_shapes(shapes, sx, width):
    xc = width / 2.0
    return apply_to_shapes(shapes, lambda pts: scale_x_points(pts, sx, xc))


def rescale_points(points, sx, sy):
    """整体线性缩放坐标（用于 96 坐标系桥接 SVG 适配 200 优化画布）。"""
    pts = points.detach().clone()
    return torch.stack([pts[:, 0] * sx, pts[:, 1] * sy], dim=1)


def rescale_shapes(shapes, sx, sy):
    return apply_to_shapes(shapes, lambda pts: rescale_points(pts, sx, sy))

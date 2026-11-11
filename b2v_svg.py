"""位图转矢量桥接的正确 SVG 导出。

修复 pydiffvg.save_svg 只保存每个 ShapeGroup 第一个 shape 的限制：
把一个 group 内的所有 Path（外轮廓 + 内孔）合并为同一个 <path> 的多个子路径，
并通过 fill-rule="evenodd" 使内孔正确镂空。
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pydiffvg


def _fmt(v):
    return ("%.3f" % v).rstrip("0").rstrip(".")


def path_sub_d(path):
    """单个 pydiffvg.Path -> 一个子路径的 d 字符串（含 M，闭合加 Z）。"""
    ncp = path.num_control_points.data.cpu().numpy()
    pts = path.points.data.cpu().numpy()
    npts = pts.shape[0]
    s = "M " + _fmt(pts[0, 0]) + " " + _fmt(pts[0, 1])
    pid = 1
    for j in range(len(ncp)):
        if ncp[j] == 0:  # 线段
            p = pid % npts
            s += " L " + _fmt(pts[p, 0]) + " " + _fmt(pts[p, 1])
            pid += 1
        elif ncp[j] == 1:  # 二次贝塞尔
            p1 = (pid + 1) % npts
            s += (" Q " + _fmt(pts[pid, 0]) + " " + _fmt(pts[pid, 1]) +
                  " " + _fmt(pts[p1, 0]) + " " + _fmt(pts[p1, 1]))
            pid += 2
        elif ncp[j] == 2:  # 三次贝塞尔
            p2 = (pid + 2) % npts
            s += (" C " + _fmt(pts[pid, 0]) + " " + _fmt(pts[pid, 1]) +
                  " " + _fmt(pts[pid + 1, 0]) + " " + _fmt(pts[pid + 1, 1]) +
                  " " + _fmt(pts[p2, 0]) + " " + _fmt(pts[p2, 1]))
            pid += 3
    if bool(path.is_closed):
        s += " Z"
    return s


def save_svg_full(filename, width, height, shapes, shape_groups):
    root = ET.Element("svg")
    root.set("version", "1.1")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))

    for group in shape_groups:
        sub_ds = []
        for sid in group.shape_ids:
            sub_ds.append(path_sub_d(shapes[int(sid)]))
        node = ET.SubElement(root, "path")
        node.set("d", " ".join(sub_ds))
        if bool(group.use_even_odd_rule):
            node.set("fill-rule", "evenodd")
        if group.fill_color is not None:
            c = group.fill_color.data.cpu().numpy()
            node.set("fill", "rgb({},{},{})".format(
                int(round(255 * c[0])), int(round(255 * c[1])), int(round(255 * c[2]))))
        else:
            node.set("fill", "none")
        node.set("stroke", "none")

    rough = ET.tostring(root, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty)

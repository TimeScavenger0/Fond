# -*- coding: utf-8 -*-
"""受控 shear 标定：对真实中文字符施加已知均匀倾斜，检验两种 slant 度量
能否单调、准确地恢复角度（只读，不修改原始数据）。"""
import os
import numpy as np
import cv2

from geom_annotate import render_glyph, measure_slant_ecc, measure_slant_regress

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE_ANGLES = [-15, -10, -5, 0, 5, 10, 15]
CALIB_CHARS = ["永", "国", "木", "人"]
# 用几个有代表性的字体
CALIB_FONTS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\simfang.ttf",
    r"C:\Windows\Fonts\SIMYOU.TTF",
]


def apply_shear(binary, angle_deg):
    """以画布中心为锚点施加水平 shear（正=顶部右倾）。"""
    h, w = binary.shape
    tan = np.tan(np.radians(angle_deg))
    # 先平移使中心为原点，shear，再平移回来
    cx, cy = w / 2.0, h / 2.0
    M = np.array([[1.0, tan, -tan * cy],
                  [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderValue=0)


def main():
    ecc_rows, reg_rows = [], []
    for fp in CALIB_FONTS:
        fbase = os.path.basename(fp)
        for ch in CALIB_CHARS:
            base = render_glyph(ch, fp)
            if base is None:
                continue
            for t in TRUE_ANGLES:
                sheared = base if t == 0 else apply_shear(base, t)
                ecc_rows.append((fbase, ch, t, measure_slant_ecc(sheared)))
                reg_rows.append((fbase, ch, t, measure_slant_regress(sheared)))

    def summarize(rows, name):
        print("\n=== %s ===" % name)
        # 按真实角度聚合：测量值的均值/标准差
        print("%8s %8s %8s %8s" % ("true", "mean", "std", "bias"))
        for t in TRUE_ANGLES:
            vals = np.array([v for (_, _, tt, v) in rows if tt == t], float)
            vals = vals[~np.isnan(vals)]
            print("%8.1f %8.3f %8.3f %8.3f" % (
                t, vals.mean(), vals.std(), vals.mean() - t))
        # 每个字符序列是否单调
        mono, total = 0, 0
        for fp in CALIB_FONTS:
            for ch in CALIB_CHARS:
                seq = [v for (fb, c, tt, v) in rows
                       if fb == os.path.basename(fp) and c == ch]
                if len(seq) == len(TRUE_ANGLES):
                    total += 1
                    if all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
                        mono += 1
        print("monotonic sequences: %d/%d" % (mono, total))

    summarize(ecc_rows, "ECC fitEllipse")
    summarize(reg_rows, "row-centroid regression")


if __name__ == "__main__":
    main()

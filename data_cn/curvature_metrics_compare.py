# -*- coding: utf-8 -*-
"""曲率衍生指标系统对比（fontclip 环境，只读）。"""
import os
import numpy as np
from scipy.stats import kurtosis

from angular_detect import (render_raw, resample, curvature,
                            FONT_DIR, TEST, CHARS)

HERE = os.path.dirname(os.path.abspath(__file__))
print("%-9s %9s %9s %9s %9s" % (
    "tag", "straight", "kurtosis", "max_k", "p90"))
for fbase, tag in TEST:
    fp = os.path.join(FONT_DIR, fbase)
    all_k = []
    for ch in CHARS:
        b = render_raw(ch, fp)
        all_k.append(curvature(resample(b)))
    k = np.concatenate(all_k)
    straight = np.mean(k < 0.012)
    kurt = kurtosis(k)
    print("%-9s %9.3f %9.2f %9.3f %9.3f" % (
        tag, straight, kurt, k.max(), np.percentile(k, 90)))

# -*- coding: utf-8 -*-
"""分析 O'Donovan 37 属性相关结构与客观->主观可预测性（fontclip 环境，只读）。"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = ["angular", "artistic", "attention-grabbing", "attractive", "bad",
         "boring", "calm", "capitals", "charming", "clumsy", "complex",
         "cursive", "delicate", "disorderly", "display", "dramatic",
         "formal", "fresh", "friendly", "gentle", "graceful", "happy",
         "italic", "legible", "modern", "monospace", "playful",
         "pretentious", "serif", "sharp", "sloppy", "soft", "strong",
         "technical", "thin", "warm", "wide"]
OBJECTIVE = ["angular", "italic", "monospace", "serif", "strong", "thin", "wide"]

with open(os.path.join(HERE, "..", "attributeData",
                       "all_font_to_attribute_values.json"),
          "r", encoding="utf-8") as f:
    data = json.load(f)

fonts = list(data.keys())
M = np.array([[float(data[fn][a]) for a in ATTRS] for fn in fonts])
print("labeled fonts:", len(fonts), "attributes:", len(ATTRS))

# 标准化
X = (M - M.mean(0)) / M.std(0, ddof=0)
idx = {a: i for i, a in enumerate(ATTRS)}
obj_idx = [idx[a] for a in OBJECTIVE]


def r2_loo(y, Xd):
    # 留一交叉验证 R²（带截距）
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        keep = np.ones(n, bool); keep[i] = False
        A = np.c_[Xd[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        pred[i] = Xd[i] @ coef[:-1] + coef[-1]
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / ss


print("\n%-18s %8s %10s" % ("attribute", "max|corr|", "LOO_R2"))
scores = {}
for a in ATTRS:
    j = idx[a]
    if a in OBJECTIVE:
        continue
    corrs = [np.corrcoef(X[:, j], X[:, k])[0, 1] for k in obj_idx]
    r2 = r2_loo(X[:, j], X[:, obj_idx])
    scores[a] = (max(abs(np.array(corrs))), r2)

for a, (mc, r2) in sorted(scores.items(), key=lambda kv: -kv[1][1]):
    print("%-18s %8.2f %10.3f" % (a, mc, r2))

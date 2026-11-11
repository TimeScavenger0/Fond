# -*- coding: utf-8 -*-
"""生成中文 37 属性伪标签（fontclip 环境，只读）。

客观 7 项直接用中文实测百分位；
主观项在 148 拉丁字体上用 7 客观属性（0-100）训练线性回归，
再把中文客观百分位输入预测，clip 到 0-100；
R²<0.4 的主观项不硬造，留空。来源与 R² 写入 meta JSON。
"""
import os
import csv
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
R2_MIN = 0.4

# 拉丁标注
with open(os.path.join(HERE, "..", "attributeData",
                       "all_font_to_attribute_values.json"),
          "r", encoding="utf-8") as f:
    od = json.load(f)
latin_fonts = list(od.keys())
L = np.array([[float(od[fn][a]) for a in ATTRS] for fn in latin_fonts])
idx = {a: i for i, a in enumerate(ATTRS)}
obj_i = [idx[a] for a in OBJECTIVE]
Xl = L[:, obj_i]

# 中文客观实测
cn = []
with open(os.path.join(HERE, "cn_objective_attributes.csv"),
          "r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        cn.append(r)
# 中文 7 客观百分位，按 OBJECTIVE 顺序
Xc = np.array([[float(r[a + "_pct"]) for a in OBJECTIVE] for r in cn])


def loo_r2(y, X):
    n = len(y); pred = np.zeros(n)
    for i in range(n):
        keep = np.ones(n, bool); keep[i] = False
        A = np.c_[X[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        pred[i] = X[i] @ coef[:-1] + coef[-1]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


meta = {}
pred_matrix = np.full((len(cn), len(ATTRS)), np.nan)
for a in ATTRS:
    j = idx[a]
    if a in OBJECTIVE:
        # 实测直接填入
        pred_matrix[:, j] = Xc[:, OBJECTIVE.index(a)]
        meta[a] = {"source": "chinese_measured", "loo_r2": None}
        continue
    y = L[:, j]
    r2 = loo_r2(y, Xl)
    meta[a] = {"source": "latin_regression_pseudo", "loo_r2": round(float(r2), 3)}
    if r2 < R2_MIN:
        continue  # 不可靠，留空
    A = np.c_[Xl, np.ones(len(Xl))]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = Xc @ coef[:-1] + coef[-1]
    pred_matrix[:, j] = np.clip(yhat, 0, 100)

# 输出 CSV（空值写空字符串）
out = os.path.join(HERE, "cn_37attr_pseudo_labels.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "family"] + ATTRS)
    for i, r in enumerate(cn):
        row = [r["file"], r["family"]]
        for j in range(len(ATTRS)):
            v = pred_matrix[i, j]
            row.append("" if np.isnan(v) else round(float(v), 2))
        w.writerow(row)

with open(os.path.join(HERE, "pseudo_label_meta.json"), "w",
          encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

filled = sum(1 for a in ATTRS if not (meta[a]["loo_r2"] is not None
                                       and meta[a]["loo_r2"] < R2_MIN)
             )
print("fonts:", len(cn), "->", out)
print("filled attributes:", filled, "/ 37; blank:", 37 - filled)
blank_attrs = [a for a in ATTRS if a not in OBJECTIVE
               and meta[a]["loo_r2"] < R2_MIN]
print("blank (R2<0.4):", blank_attrs)
with open(out, "rb") as f:
    print("BOM:", f.read(3) == b"\xef\xbb\xbf")

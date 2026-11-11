"""
绘制新旧模型在整体验证集（13 字体）上的逐属性相关性对比条形图。
读取 cn_val_compare_per_attr.csv 的 all_13 部分，按新模型相关性排序输出。
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "cn_val_compare_per_attr.csv")
OUT = os.path.join(BASE, "cn_attr_compare.png")

simhei = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(simhei):
    font_manager.fontManager.addfont(simhei)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    rows = []
    with open(CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["subset"] == "all_13":
                rows.append((r["attribute"], float(r["old_corr"]), float(r["new_corr"])))
    rows.sort(key=lambda x: x[2])

    attrs = [r[0] for r in rows]
    old = [r[1] for r in rows]
    new = [r[2] for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(11, 10))
    h = 0.38
    ax.barh([i + h / 2 for i in y], old, height=h, color="#9bb8d3", label="旧：拉丁 LoRA")
    ax.barh([i - h / 2 for i in y], new, height=h, color="#1f6fb4", label="新：混合 LoRA")
    ax.set_yticks(list(y))
    ax.set_yticklabels(attrs)
    ax.set_xlabel("属性预测相关性（Pearson r）")
    ax.set_title("新旧 LoRA 在混合验证集（13 字体）上的逐属性相关性对比", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved:", OUT)


if __name__ == "__main__":
    main()

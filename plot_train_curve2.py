"""
绘制全模块（A+B+C+D）训练曲线：loss 与 val_correlation 双轴，标注最佳 epoch。
读取 train_curve2.csv，输出 train_curve2.png。
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "train_curve2.csv")
OUT = os.path.join(BASE, "train_curve2.png")

simhei = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(simhei):
    font_manager.fontManager.addfont(simhei)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    epochs, losses, corrs = [], [], []
    with open(CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            corrs.append(float(row["val_correlation"]))

    best_idx = max(range(len(corrs)), key=lambda i: corrs[i])
    best_epoch = epochs[best_idx]
    best_corr = corrs[best_idx]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    color1 = "#1f6fb4"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练 Loss（主 BCE + 辅助损失）", color=color1)
    ax1.plot(epochs, losses, color=color1, linewidth=1.8, label="训练 Loss")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    color2 = "#d9534f"
    ax2.set_ylabel("验证集属性相关性 val_correlation", color=color2)
    ax2.plot(epochs, corrs, color=color2, linewidth=1.8, label="val_correlation")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.scatter([best_epoch], [best_corr], color=color2, zorder=5, s=55)
    ax2.annotate(
        "最佳 epoch %d\nval_corr=%.4f" % (best_epoch, best_corr),
        xy=(best_epoch, best_corr),
        xytext=(best_epoch - 25, best_corr - 0.05),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.title("全模块（A+B+C+D）中文+拉丁混合 LoRA 续训曲线（100 epoch）", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("plot saved:", OUT)
    print("best epoch:", best_epoch, "best val_corr:", best_corr)


if __name__ == "__main__":
    main()

"""
解析 Featurize 混合 LoRA 训练日志 train_run.log。

日志含 tqdm 控制字符与空字节，普通文本工具无法直接读。这里用正则提取
每个 `EPOCH: N, loss: X, val_correlation: Y` 行，输出训练曲线摘要，
并把逐 epoch 数据存为 utf-8-sig CSV，便于后续核对与画图。
"""
import csv
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_run.log")
OUT_CSV = os.path.join(BASE, "train_curve.csv")

PATTERN = re.compile(
    r"EPOCH:\s*(\d+),\s*loss:\s*([0-9.eE+-]+),\s*val_correlation:\s*([0-9.eE+-]+)"
)


def main():
    # 日志含空字节与控制字符，用 latin-1 逐字节无损读入，再交给正则
    with open(LOG, "r", encoding="latin-1") as f:
        text = f.read()

    records = []
    for m in PATTERN.finditer(text):
        epoch = int(m.group(1))
        loss = float(m.group(2))
        val_corr = float(m.group(3))
        records.append((epoch, loss, val_corr))

    if not records:
        print("no EPOCH records found")
        return

    # 日志可能因多次运行重复，按 epoch 去重，保留最后一次出现
    dedup = {}
    for epoch, loss, val_corr in records:
        dedup[epoch] = (loss, val_corr)
    rows = sorted(dedup.items())

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_correlation"])
        for epoch, (loss, val_corr) in rows:
            writer.writerow([epoch, round(loss, 6), round(val_corr, 6)])

    first_epoch, (first_loss, first_corr) = rows[0]
    last_epoch, (last_loss, last_corr) = rows[-1]
    best_epoch, (best_loss, best_corr) = max(rows, key=lambda r: r[1][1])
    min_loss_epoch, (min_loss, min_loss_corr) = min(rows, key=lambda r: r[1][0])

    print("parsed epochs:", len(rows), "range:", first_epoch, "->", last_epoch)
    print("first: loss=%.4f val_corr=%.4f" % (first_loss, first_corr))
    print("last : loss=%.4f val_corr=%.4f" % (last_loss, last_corr))
    print("best val_corr=%.4f at epoch %d (loss=%.4f)" % (best_corr, best_epoch, best_loss))
    print("min loss=%.4f at epoch %d" % (min_loss, min_loss_epoch))
    print("csv saved:", OUT_CSV)


if __name__ == "__main__":
    main()

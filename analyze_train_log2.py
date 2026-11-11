"""
解析全模块（A+B+C+D）训练日志 train_run2.log。
与 S0 的 analyze_train_log.py 同口径，输出 train_curve2.csv 与关键摘要。
"""
import csv
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_run2.log")
OUT_CSV = os.path.join(BASE, "train_curve2.csv")

PATTERN = re.compile(
    r"EPOCH:\s*(\d+),\s*loss:\s*([0-9.eE+-]+),\s*val_correlation:\s*([0-9.eE+-]+)"
)


def main():
    with open(LOG, "r", encoding="latin-1") as f:
        text = f.read()

    records = []
    for m in PATTERN.finditer(text):
        records.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    if not records:
        print("no EPOCH records found")
        return

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
    print("min loss=%.4f at epoch %d (val_corr=%.4f)" % (min_loss, min_loss_epoch, min_loss_corr))
    print("csv saved:", OUT_CSV)


if __name__ == "__main__":
    main()

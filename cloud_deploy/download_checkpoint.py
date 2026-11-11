"""
下载训练所需基础权重 fontclip_lora_best.pt（GitHub Release v1.0）到 model_checkpoints/。

仅用标准库 urllib（云端 Linux 可直连 GitHub）。已存在且字节数正确则跳过；
下载后校验字节数，避免半截文件。VGG19 权重训练侧不需要，不在此下载。
"""
import os
import sys
import urllib.request

URL = "https://github.com/TimeScavenger0/Fond/releases/download/v1.0/fontclip_lora_best.pt"
EXPECTED_BYTES = 929836810

# 脚本位于 cloud_deploy/，项目根为其上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST_DIR = os.path.join(ROOT, "model_checkpoints")
DST = os.path.join(DST_DIR, "fontclip_lora_best.pt")


def report(block_num, block_size, total):
    got = block_num * block_size
    pct = min(got / total * 100, 100) if total else 0
    sys.stdout.write("\rdownloading: %.1f%%  %.1f/%.1f MB"
                     % (pct, got / 1e6, (total or EXPECTED_BYTES) / 1e6))
    sys.stdout.flush()


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    if os.path.exists(DST) and os.path.getsize(DST) == EXPECTED_BYTES:
        print("already exists with correct size, skip:", DST)
        return
    print("downloading from:", URL)
    urllib.request.urlretrieve(URL, DST, reporthook=report)
    print()
    actual = os.path.getsize(DST)
    if actual != EXPECTED_BYTES:
        raise RuntimeError("size mismatch: got %d, expected %d" % (actual, EXPECTED_BYTES))
    print("checkpoint ready:", DST, "%.1f MB" % (actual / 1e6))


if __name__ == "__main__":
    main()

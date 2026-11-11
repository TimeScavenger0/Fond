"""
验证 fond_cloud_bundle.zip 解压后能满足 train_cn_mixed.py 的云端路径重映射。

核对：
  - cloud_fonts 含 64 TTF，与三个 split font_paths 的 basename 一一对应，且 PIL 可加载
  - data_cn/mixed 含 9 JSON + 3 图片目录（64 PNG）
  - bundle_manifest 与实际文件一致
"""
import json
import os
import zipfile
from PIL import ImageFont

BASE = r"D:\VScode\VScodeProjects\Fond"
MIXED = os.path.join(BASE, "data_cn", "mixed")
DEPLOY = os.path.join(BASE, "cloud_deploy")
ZIP_PATH = os.path.join(DEPLOY, "fond_cloud_bundle.zip")
VERIFY_DIR = os.path.join(DEPLOY, "_verify")

SPLIT_PATH_FILES = [
    "mixed_font_paths.json",
    "mixed_val_font_paths.json",
    "mixed_test_font_paths.json",
]
IMAGE_DIRS = ["mixed_images", "mixed_val_images", "mixed_test_images"]


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    import shutil
    if os.path.exists(VERIFY_DIR):
        shutil.rmtree(VERIFY_DIR)
    os.makedirs(VERIFY_DIR)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(VERIFY_DIR)

    cloud_fonts = os.path.join(VERIFY_DIR, "cloud_fonts")
    mixed_staging = os.path.join(VERIFY_DIR, "data_cn", "mixed")

    # 1. 字体 basename 一一对应 + PIL 可加载
    expected_basenames = set()
    for pf in SPLIT_PATH_FILES:
        for src in load(os.path.join(MIXED, pf)):
            expected_basenames.add(os.path.basename(src))
    actual_fonts = {f for f in os.listdir(cloud_fonts) if f.lower().endswith(".ttf")}
    assert expected_basenames == actual_fonts, (
        "missing:", expected_basenames - actual_fonts,
        "extra:", actual_fonts - expected_basenames,
    )
    for f in sorted(actual_fonts):
        ImageFont.truetype(os.path.join(cloud_fonts, f), 150)  # dataset 构造同款调用
    print("font basenames match & PIL truetype OK:", len(actual_fonts))

    # 2. 模拟云端重映射（原始路径强制视为不存在）
    for pf in SPLIT_PATH_FILES:
        for src in load(os.path.join(MIXED, pf)):
            cand = os.path.join(cloud_fonts, os.path.basename(src))
            assert os.path.exists(cand), cand
    print("cloud font remap rule OK")

    # 3. JSON + 图片目录
    jsons = [f for f in os.listdir(mixed_staging) if f.endswith(".json")]
    assert len(jsons) == 9, jsons
    png_total = 0
    for d in IMAGE_DIRS:
        pdir = os.path.join(mixed_staging, d)
        assert os.path.isdir(pdir), pdir
        png_total += len([f for f in os.listdir(pdir) if f.endswith(".png")])
    assert png_total == 64, png_total
    print("json=9, png total:", png_total)

    # 4. manifest 一致
    manifest = load(os.path.join(VERIFY_DIR, "bundle_manifest.json"))
    real = []
    for root, _, files in os.walk(VERIFY_DIR):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, VERIFY_DIR).replace(os.sep, "/")
            if rel == "bundle_manifest.json":  # manifest 自身不在其 entries 中
                continue
            real.append({"path": rel, "bytes": os.path.getsize(full)})
    real.sort(key=lambda e: e["path"])
    assert real == manifest["entries"], "manifest mismatch"
    print("manifest entries match:", manifest["total_files"])

    print("\nBUNDLE VERIFY PASSED")


if __name__ == "__main__":
    main()

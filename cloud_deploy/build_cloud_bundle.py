"""
构建云端 GPU 训练数据 bundle（配套 train_cn_mixed.py 跨平台路径重映射）。

收集：
  - 64 个真实 TTF（24 中文 + 40 拉丁）-> cloud_fonts/（实际复制，非硬链接）
  - data_cn/mixed 全部 12 个 JSON + 三个图片目录（38+13+13=64 PNG，实际复制）
输出 fond_cloud_bundle.zip，含 bundle_manifest.json（文件清单与字节数）。

云端解压到 git clone 的项目根后，目录与 train_cn_mixed.py 的重映射规则对齐：
  cloud_fonts/、data_cn/mixed/。
"""
import json
import os
import shutil
import zipfile

BASE = r"D:\VScode\VScodeProjects\Fond"
MIXED = os.path.join(BASE, "data_cn", "mixed")
DEPLOY = os.path.join(BASE, "cloud_deploy")
STAGING = os.path.join(DEPLOY, "staging")
ZIP_PATH = os.path.join(DEPLOY, "fond_cloud_bundle.zip")

SPLIT_PATH_FILES = [
    "mixed_font_paths.json",
    "mixed_val_font_paths.json",
    "mixed_test_font_paths.json",
]
IMAGE_DIRS = ["mixed_images", "mixed_val_images", "mixed_test_images"]
JSON_FILES = [
    "mixed_font_to_attribute_values.json",
    "mixed_font_paths.json",
    "mixed_dataset_meta.json",
    "mixed_val_font_to_attribute_values.json",
    "mixed_val_font_paths.json",
    "mixed_val_meta.json",
    "mixed_test_font_to_attribute_values.json",
    "mixed_test_font_paths.json",
    "mixed_test_meta.json",
]


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 干净重建 staging
    if os.path.exists(STAGING):
        shutil.rmtree(STAGING)
    os.makedirs(STAGING)

    # ---- 1. 字体：三个 split 去重 ----
    fonts_staging = os.path.join(STAGING, "cloud_fonts")
    os.makedirs(fonts_staging)
    seen = set()
    font_records = []
    for pf in SPLIT_PATH_FILES:
        for src in load(os.path.join(MIXED, pf)):
            if src in seen:
                continue
            seen.add(src)
            assert os.path.exists(src), src
            dst = os.path.join(fonts_staging, os.path.basename(src))
            shutil.copy2(src, dst)
            font_records.append(os.path.basename(src))
    print("fonts collected:", len(font_records))

    # ---- 2. JSON + 图片目录 ----
    mixed_staging = os.path.join(STAGING, "data_cn", "mixed")
    os.makedirs(mixed_staging)
    for jf in JSON_FILES:
        shutil.copy2(os.path.join(MIXED, jf), os.path.join(mixed_staging, jf))
    png_count = 0
    for d in IMAGE_DIRS:
        src_dir = os.path.join(MIXED, d)
        dst_dir = os.path.join(mixed_staging, d)
        shutil.copytree(src_dir, dst_dir)
        png_count += len([f for f in os.listdir(dst_dir) if f.endswith(".png")])
    print("png collected:", png_count)

    # ---- 3. manifest ----
    entries = []
    for root, _, files in os.walk(STAGING):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, STAGING).replace(os.sep, "/")
            entries.append({"path": rel, "bytes": os.path.getsize(full)})
    entries.sort(key=lambda e: e["path"])
    manifest = {
        "bundle": "fond_cloud_bundle",
        "font_count": len(font_records),
        "png_count": png_count,
        "json_count": len(JSON_FILES),
        "total_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "entries": entries,
    }
    with open(os.path.join(STAGING, "bundle_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- 4. 打包 zip（含中文文件名，UTF-8）----
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    all_files = []
    for root, _, files in os.walk(STAGING):
        for f in files:
            all_files.append(os.path.join(root, f))
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for full in all_files:
            arc = os.path.relpath(full, STAGING)
            zf.write(full, arc)
    print("zip written:", ZIP_PATH, "%.2f MB" % (os.path.getsize(ZIP_PATH) / 1e6))

    # ---- 5. 验证 zip 完整性 ----
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad = zf.testzip()
        assert bad is None, bad
        names = zf.namelist()
    print("zip entries:", len(names), "testzip OK")
    print("\nBUNDLE BUILD PASSED")


if __name__ == "__main__":
    main()

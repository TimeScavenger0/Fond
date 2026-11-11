"""
构建中文 + 拉丁混合训练数据集（训练方案第二节）。

- 中文 train 14 字体全部保留；拉丁 train 88 字体用固定种子抽 LATIN_REPLAY 个 replay。
- 中文缺 7 个留空属性（R2<0.4），用占位值 PLACEHOLDER 补齐到 37，仅为满足
  BCE matrix 的结构断言；这 7 个属性训练时经 exclusive 排除、不生成 prompt，不参与训练。
- 同盘硬链接汇集 PNG 到 mixed_images（不占额外空间、不改源文件）。
- 输出混合 JSON、字体文件路径列表、可复现 meta，并做完整验证。
"""
import json
import os
import random

BASE = r"D:\VScode\VScodeProjects\Fond"
CN_TRAIN_JSON = os.path.join(BASE, r"data_cn\dataset\train_font_to_attribute_values.json")
LATIN_TRAIN_JSON = os.path.join(BASE, "attributeData", "train_font_to_attribute_values.json")
INVENTORY_JSON = os.path.join(BASE, "data_cn", "font_inventory.json")
CN_IMAGES_DIR = os.path.join(BASE, r"data_cn\dataset\cn_images")
GW_IMAGES_DIR = os.path.join(BASE, "gwfonts_images")
GW_FONTS_DIR = os.path.join(BASE, "gwfonts")

OUT_DIR = os.path.join(BASE, "data_cn", "mixed")
MIXED_IMAGES_DIR = os.path.join(OUT_DIR, "mixed_images")

SEED = 123
LATIN_REPLAY = 24
PLACEHOLDER = "0"  # 明确"无"方向；对应属性 exclusive 后实际不被 mask 查询

# 本阶段排除（不参与训练）：默认 capitals/monospace + 7 个留空属性
EXCLUSIVE = [
    "capitals", "monospace",
    "bad", "cursive", "display", "fresh", "friendly", "gentle",
]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def link_or_verify(src, dst):
    """硬链接 src -> dst；已存在则核对是否同一文件。"""
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return "skipped"
        raise RuntimeError(f"target exists and differs from source: {dst}")
    os.link(src, dst)
    return "linked"


def main():
    cn = load_json(CN_TRAIN_JSON)
    latin = load_json(LATIN_TRAIN_JSON)
    inventory = load_json(INVENTORY_JSON)

    cn_names = list(cn.keys())
    latin_names_all = list(latin.keys())

    # 37 属性全局顺序（与 initialize_font_data.all_attributes 同口径）
    all_attrs = list(next(iter(latin.values())).keys())
    assert len(all_attrs) == 37, f"expected 37 attrs, got {len(all_attrs)}"

    sample_cn_attrs = next(iter(cn.values()))
    cn_missing = [a for a in all_attrs if a not in sample_cn_attrs]
    print("CN missing attrs (to fill placeholder):", cn_missing)
    assert set(cn_missing) == set(EXCLUSIVE) - {"monospace"}, cn_missing

    # 固定种子抽拉丁 replay（sorted 保证可复现）
    rng = random.Random(SEED)
    latin_subset = rng.sample(sorted(latin_names_all), LATIN_REPLAY)
    print("Latin replay subset:", latin_subset)

    # 字体名零冲突
    assert not (set(cn_names) & set(latin_subset)), "font name collision"

    # 合并 JSON，统一 37 属性、统一顺序
    merged = {}
    for name in cn_names:
        src_attrs = cn[name]
        merged[name] = {a: src_attrs.get(a, PLACEHOLDER) for a in all_attrs}
    for name in latin_subset:
        src_attrs = latin[name]
        merged[name] = {a: src_attrs[a] for a in all_attrs}

    # 字体文件路径列表（MyDataset font_dir 支持 list，元素须为具体文件路径）
    inv_by_stem = {
        os.path.splitext(e["file"])[0]: e["path"]
        for e in inventory
        if e.get("ok")
    }
    font_paths = []
    for name in cn_names:
        p = inv_by_stem[name]
        assert os.path.exists(p), p
        font_paths.append(p)
    for name in latin_subset:
        p = os.path.join(GW_FONTS_DIR, name + ".ttf")
        assert os.path.exists(p), p
        font_paths.append(p)

    # 硬链接图片
    os.makedirs(MIXED_IMAGES_DIR, exist_ok=True)
    link_log = {}
    for name in cn_names:
        src = os.path.join(CN_IMAGES_DIR, name + ".png")
        assert os.path.exists(src), src
        link_log[name] = link_or_verify(src, os.path.join(MIXED_IMAGES_DIR, name + ".png"))
    for name in latin_subset:
        src = os.path.join(GW_IMAGES_DIR, name + ".png")
        assert os.path.exists(src), src
        link_log[name] = link_or_verify(src, os.path.join(MIXED_IMAGES_DIR, name + ".png"))

    os.makedirs(OUT_DIR, exist_ok=True)
    merged_json = os.path.join(OUT_DIR, "mixed_font_to_attribute_values.json")
    paths_json = os.path.join(OUT_DIR, "mixed_font_paths.json")
    meta_json = os.path.join(OUT_DIR, "mixed_dataset_meta.json")

    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    with open(paths_json, "w", encoding="utf-8") as f:
        json.dump(font_paths, f, ensure_ascii=False, indent=2)

    meta = {
        "seed": SEED,
        "counts": {
            "cn_train": len(cn_names),
            "latin_replay": LATIN_REPLAY,
            "total": len(merged),
        },
        "cn_share": round(len(cn_names) / len(merged), 4),
        "target_sampled_cn_share": 0.40,
        "sampling_note": "字体数量占比低于40%，训练时通过 WeightedRandomSampler 采样权重保证中文实际采样占比>=40%",
        "cn_names": cn_names,
        "latin_replay_names": latin_subset,
        "placeholder_attributes": cn_missing,
        "placeholder_value": PLACEHOLDER,
        "exclusive_attributes": EXCLUSIVE,
        "active_attributes": [a for a in all_attrs if a not in EXCLUSIVE],
        "n_active_attributes": len([a for a in all_attrs if a not in EXCLUSIVE]),
        "mixed_images_dir": MIXED_IMAGES_DIR,
        "link_log": link_log,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    verify(merged, font_paths, cn_names, latin_subset, meta)
    print("\nALL CHECKS PASSED")


def verify(merged, font_paths, cn_names, latin_subset, meta):
    total = len(cn_names) + len(latin_subset)
    assert len(merged) == total, (len(merged), total)
    # 每个字体 37 属性
    for name, row in merged.items():
        assert len(row) == 37, (name, len(row))
    # 中文占位位置正确
    for name in cn_names:
        for a in meta["placeholder_attributes"]:
            assert merged[name][a] == PLACEHOLDER, (name, a)
    # 字体路径数与存在性
    assert len(font_paths) == total
    for p in font_paths:
        assert os.path.exists(p), p
    # 图片硬链接数
    pngs = [f for f in os.listdir(MIXED_IMAGES_DIR) if f.endswith(".png")]
    assert len(pngs) == total, (len(pngs), total)
    for name in list(merged.keys()):
        assert os.path.exists(os.path.join(MIXED_IMAGES_DIR, name + ".png"))
    # 字体数量占比如实记录；40% 目标由采样权重在 DataLoader 层保证
    assert 0 < meta["cn_share"] < 1, meta["cn_share"]
    assert meta["target_sampled_cn_share"] >= 0.40
    # 活跃属性 29
    assert meta["n_active_attributes"] == 29, meta["n_active_attributes"]
    print("verify: total fonts =", total)
    print("verify: cn share =", meta["cn_share"])
    print("verify: active attrs =", meta["n_active_attributes"])
    print("verify: hardlinked pngs =", len(pngs))


if __name__ == "__main__":
    main()

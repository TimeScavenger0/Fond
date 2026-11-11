"""
构建中文 + 拉丁混合验证集与测试集（训练方案第六节，配套 build_mixed_dataset.py）。

- 中文 val/test 各 5 字体全部保留；拉丁 val/test 各 30 字体用固定种子抽 LATIN_EVAL 个。
- 中文缺 7 个留空属性，用占位 PLACEHOLDER 补齐到 37，exclusive 口径与训练集完全一致。
- 同盘硬链接汇集 PNG 到各自图片目录（不占额外空间、不改源文件）。
- 输出混合 JSON、字体路径、可复现 meta，并做完整验证。
"""
import json
import os
import random

BASE = r"D:\VScode\VScodeProjects\Fond"
INVENTORY_JSON = os.path.join(BASE, "data_cn", "font_inventory.json")
CN_IMAGES_DIR = os.path.join(BASE, r"data_cn\dataset\cn_images")
GW_IMAGES_DIR = os.path.join(BASE, "gwfonts_images")
GW_FONTS_DIR = os.path.join(BASE, "gwfonts")

OUT_DIR = os.path.join(BASE, "data_cn", "mixed")

SEED = 123
LATIN_EVAL = 8
PLACEHOLDER = "0"

EXCLUSIVE = [
    "capitals", "monospace",
    "bad", "cursive", "display", "fresh", "friendly", "gentle",
]

EVAL_SPLITS = {
    "val": {
        "cn": r"data_cn\dataset\validation_font_to_attribute_values.json",
        "latin": r"attributeData\validation_font_to_attribute_values.json",
    },
    "test": {
        "cn": r"data_cn\dataset\test_font_to_attribute_values.json",
        "latin": r"attributeData\test_font_to_attribute_values.json",
    },
}


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def link_or_verify(src, dst):
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return "skipped"
        raise RuntimeError(f"target exists and differs from source: {dst}")
    os.link(src, dst)
    return "linked"


def build_split(split_key, paths, inventory, all_attrs):
    cn = load_json(os.path.join(BASE, paths["cn"]))
    latin = load_json(os.path.join(BASE, paths["latin"]))

    cn_names = list(cn.keys())
    latin_names_all = list(latin.keys())

    sample_cn_attrs = next(iter(cn.values()))
    cn_missing = [a for a in all_attrs if a not in sample_cn_attrs]
    assert set(cn_missing) == set(EXCLUSIVE) - {"monospace"}, cn_missing

    # 每个 split 用独立派生种子，保证可复现且互不干扰
    rng = random.Random(SEED + (1 if split_key == "val" else 2))
    latin_subset = rng.sample(sorted(latin_names_all), LATIN_EVAL)
    assert not (set(cn_names) & set(latin_subset)), "font name collision"

    merged = {}
    for name in cn_names:
        merged[name] = {a: cn[name].get(a, PLACEHOLDER) for a in all_attrs}
    for name in latin_subset:
        merged[name] = {a: latin[name][a] for a in all_attrs}

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

    images_dir = os.path.join(OUT_DIR, f"mixed_{split_key}_images")
    os.makedirs(images_dir, exist_ok=True)
    link_log = {}
    for name in cn_names:
        src = os.path.join(CN_IMAGES_DIR, name + ".png")
        assert os.path.exists(src), src
        link_log[name] = link_or_verify(src, os.path.join(images_dir, name + ".png"))
    for name in latin_subset:
        src = os.path.join(GW_IMAGES_DIR, name + ".png")
        assert os.path.exists(src), src
        link_log[name] = link_or_verify(src, os.path.join(images_dir, name + ".png"))

    total = len(cn_names) + len(latin_subset)
    for name, row in merged.items():
        assert len(row) == 37, (name, len(row))
    assert len(font_paths) == total
    pngs = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    assert len(pngs) == total, (len(pngs), total)

    out_json = os.path.join(OUT_DIR, f"mixed_{split_key}_font_to_attribute_values.json")
    out_paths = os.path.join(OUT_DIR, f"mixed_{split_key}_font_paths.json")
    out_meta = os.path.join(OUT_DIR, f"mixed_{split_key}_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    with open(out_paths, "w", encoding="utf-8") as f:
        json.dump(font_paths, f, ensure_ascii=False, indent=2)

    meta = {
        "split": split_key,
        "seed": SEED + (1 if split_key == "val" else 2),
        "counts": {"cn": len(cn_names), "latin": LATIN_EVAL, "total": total},
        "cn_share": round(len(cn_names) / total, 4),
        "cn_names": cn_names,
        "latin_names": latin_subset,
        "placeholder_attributes": cn_missing,
        "placeholder_value": PLACEHOLDER,
        "exclusive_attributes": EXCLUSIVE,
        "active_attributes": [a for a in all_attrs if a not in EXCLUSIVE],
        "n_active_attributes": len([a for a in all_attrs if a not in EXCLUSIVE]),
        "images_dir": images_dir,
        "link_log": link_log,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{split_key}] total={total} cn={cn_names} latin={latin_subset}")
    print(f"[{split_key}] active attrs={meta['n_active_attributes']} hardlinked pngs={len(pngs)}")
    return meta


def main():
    inventory = load_json(INVENTORY_JSON)
    # 37 属性顺序以拉丁训练 JSON 为准
    latin_train = load_json(os.path.join(BASE, "attributeData", "train_font_to_attribute_values.json"))
    all_attrs = list(next(iter(latin_train.values())).keys())
    assert len(all_attrs) == 37

    os.makedirs(OUT_DIR, exist_ok=True)
    for split_key, paths in EVAL_SPLITS.items():
        build_split(split_key, paths, inventory, all_attrs)
    print("\nALL EVAL SETS BUILT AND VERIFIED")


if __name__ == "__main__":
    main()

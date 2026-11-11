"""
三模型在混合验证集上的属性预测相关性对比：
旧拉丁 LoRA（OLD）、S0 混合 LoRA（S0）、全模块 A+B+C+D（FULL）。

在三个子集（整体 13 / 仅中文 5 / 仅拉丁 8）上分别计算 CLIP 余弦与真实
属性分数的 Pearson 相关均值，判定全模块是否在降低纠缠的同时不回落：
中文是否提升/保持、拉丁是否退化、整体是否稳定。
"""
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

from dataset.dataset import MyDataset
from models.init_model import device, load_model, preprocess
from models.lora import LoRAConfig
from utils.tokenizer import tokenize

BASE = os.path.dirname(os.path.abspath(__file__))
MIXED = os.path.join(BASE, "data_cn", "mixed")

_BASE_NAME = (
    "ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch48_aug50_lbound_of_scale0.01_"
    "max_attr_num_3_random_p_num_1000_geta0.2_use_negative_lr1e-05-0.1_image_file_dir"
)
CKPTS = {
    "OLD": os.path.join(BASE, "model_checkpoints", "fontclip_lora_best.pt"),
    "S0": os.path.join(BASE, _BASE_NAME + ".pt"),
    "FULL": os.path.join(BASE, _BASE_NAME + "_orth0.1-b4.0_attrh1.0_cs1.0-o0.1-r0.5_anchor1.0.pt"),
}


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(ckpt):
    lora = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    model = load_model(
        ckpt,
        model_name="ViT-B/32",
        use_oft_vision=False,
        use_oft_text=False,
        use_lora_text=True,
        use_lora_vision=False,
        lora_config_text=lora,
        use_coop_text=False,
        use_coop_vision=False,
        precontext_dropout_rate=0,
        pt_applied_layers=None,
    )
    model.float()
    model.eval()
    return model


def encode_images(model, names, img_dir):
    tensors = []
    for name in names:
        im = Image.open(os.path.join(img_dir, name + ".png")).convert("RGB")
        tensors.append(preprocess(im))
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    return feat / torch.norm(feat, dim=1, keepdim=True)


def encode_texts(model, attrs):
    prompts = [MyDataset.generate_prompt(a) for a in attrs]
    tok = tokenize(prompts).to(device)
    with torch.no_grad():
        feat = model.encode_text(tok)
    return feat / torch.norm(feat, dim=1, keepdim=True)


def eval_subset(model, names, attrs, labels, img_dir):
    img_feat = encode_images(model, names, img_dir)
    text_feat = encode_texts(model, attrs)
    cos = (text_feat @ img_feat.T).cpu().numpy()
    gt = np.array([[float(labels[name][a]) for name in names] for a in attrs])

    per_attr = []
    for i, a in enumerate(attrs):
        if np.std(gt[i]) == 0:
            per_attr.append((a, np.nan))
            continue
        r = np.corrcoef(cos[i], gt[i])[0, 1]
        per_attr.append((a, float(r)))
    valid = [r for _, r in per_attr if not np.isnan(r)]
    return float(np.mean(valid)), per_attr


def main():
    val_meta = load_json(os.path.join(MIXED, "mixed_val_meta.json"))
    labels = load_json(os.path.join(MIXED, "mixed_val_font_to_attribute_values.json"))
    attrs = val_meta["active_attributes"]
    img_dir = val_meta["images_dir"].replace("\\", os.sep)
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(MIXED, "mixed_val_images")

    cn_names = val_meta["cn_names"]
    latin_names = val_meta["latin_names"]
    all_names = cn_names + latin_names
    subsets = {
        "all_13": all_names,
        "chinese_5": cn_names,
        "latin_8": latin_names,
    }

    models = {}
    for tag, ckpt in CKPTS.items():
        assert os.path.exists(ckpt), ckpt
        print("loading %s ..." % tag)
        models[tag] = build_model(ckpt)

    # 每子集每模型的均值与逐属性
    means = {tag: {} for tag in CKPTS}
    details = {}
    for subset, names in subsets.items():
        details[subset] = {}
        for tag, model in models.items():
            mean, per_attr = eval_subset(model, names, attrs, labels, img_dir)
            means[tag][subset] = mean
            details[subset][tag] = dict(per_attr)

    summary_rows = []
    for subset in subsets:
        old = means["OLD"][subset]
        s0 = means["S0"][subset]
        full = means["FULL"][subset]
        summary_rows.append([subset, round(old, 4), round(s0, 4), round(full, 4),
                             round(full - s0, 4), round(full - old, 4)])
        print("%s: OLD=%.4f S0=%.4f FULL=%.4f (FULL-S0=%+.4f FULL-OLD=%+.4f)"
              % (subset, old, s0, full, full - s0, full - old))

    with open(os.path.join(BASE, "three_model_summary.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "OLD", "S0", "FULL", "full_minus_s0", "full_minus_old"])
        w.writerows(summary_rows)

    with open(os.path.join(BASE, "three_model_per_attr.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "attribute", "OLD", "S0", "FULL"])
        for subset in subsets:
            for a in attrs:
                row = [subset, a]
                for tag in ("OLD", "S0", "FULL"):
                    r = details[subset][tag][a]
                    row.append("" if np.isnan(r) else round(r, 4))
                w.writerow(row)

    print("THREE MODEL COMPARE DONE")


if __name__ == "__main__":
    main()

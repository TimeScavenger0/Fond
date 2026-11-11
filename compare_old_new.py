"""
对比旧拉丁 LoRA 与新混合 LoRA 在中文验证集上的属性预测相关性。

两个模型用相同 LoRAConfig 加载、fp16->fp32，在混合验证集的三个子集
（整体 13 字体 / 仅中文 5 / 仅拉丁 8）上分别计算：
CLIP 余弦相似度与真实属性分数的 Pearson 相关均值，回答两个问题：
中文属性预测是否提升（核心目标）、拉丁是否退化（防退化）。
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
OLD_CKPT = os.path.join(BASE, "model_checkpoints", "fontclip_lora_best.pt")
NEW_CKPT = os.path.join(
    BASE,
    "ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch48_aug50_lbound_of_scale0.01_"
    "max_attr_num_3_random_p_num_1000_geta0.2_use_negative_lr1e-05-0.1_image_file_dir.pt",
)


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

    print("loading OLD model ...")
    old_model = build_model(OLD_CKPT)
    print("loading NEW model ...")
    new_model = build_model(NEW_CKPT)

    summary_rows = []
    detail_rows = []
    for subset, names in subsets.items():
        old_mean, old_detail = eval_subset(old_model, names, attrs, labels, img_dir)
        new_mean, new_detail = eval_subset(new_model, names, attrs, labels, img_dir)
        summary_rows.append([subset, round(old_mean, 4), round(new_mean, 4),
                             round(new_mean - old_mean, 4)])
        old_map = dict(old_detail)
        for a, new_r in new_detail:
            old_r = old_map[a]
            detail_rows.append([subset, a,
                                "" if np.isnan(old_r) else round(old_r, 4),
                                "" if np.isnan(new_r) else round(new_r, 4)])
        print("%s: old=%.4f new=%.4f delta=%+.4f"
              % (subset, old_mean, new_mean, new_mean - old_mean))

    with open(os.path.join(BASE, "cn_val_compare_summary.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "old_corr", "new_corr", "delta"])
        w.writerows(summary_rows)

    with open(os.path.join(BASE, "cn_val_compare_per_attr.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "attribute", "old_corr", "new_corr"])
        w.writerows(detail_rows)

    print("COMPARE DONE")


if __name__ == "__main__":
    main()

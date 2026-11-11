"""
属性方向与纠缠分析（S0 基线，为 S1 模块 A 提供对照）。

按训练方案定义，属性方向 d_a = encode_text("a font") - encode_text("not a font")，
归一化后计算属性方向两两余弦矩阵，量化纠缠：
整体平均绝对余弦、方案点名的易纠缠对（serif/formal、thin/delicate、
strong/angular）以及若干对立属性对。对比新旧模型，输出 CSV 与热力图。
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from dataset.dataset import MyDataset
from models.init_model import device, load_model
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

# 方案点名的易纠缠（共现）对，期望方向余弦下降
ENTANGLED_PAIRS = [
    ("serif", "formal"),
    ("thin", "delicate"),
    ("strong", "angular"),
]
# 语义对立对，期望方向余弦更负
OPPOSITE_PAIRS = [
    ("thin", "strong"),
    ("calm", "dramatic"),
    ("boring", "playful"),
    ("soft", "sharp"),
    ("legible", "sloppy"),
]


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


def direction_matrix(model, attrs):
    pos_prompts = [MyDataset.generate_prompt(a) for a in attrs]
    neg_prompts = [MyDataset.generate_prompt(a, negative=True) for a in attrs]
    pos_tok = tokenize(pos_prompts).to(device)
    neg_tok = tokenize(neg_prompts).to(device)
    with torch.no_grad():
        f_pos = model.encode_text(pos_tok)
        f_neg = model.encode_text(neg_tok)
    d = f_pos - f_neg
    d = d / torch.norm(d, dim=1, keepdim=True)
    return (d @ d.T).cpu().numpy()


def pair_cos(mat, attrs, a, b):
    return mat[attrs.index(a), attrs.index(b)]


def main():
    simhei = "C:/Windows/Fonts/simhei.ttf"
    if os.path.exists(simhei):
        font_manager.fontManager.addfont(simhei)
        plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    attrs = load_json(os.path.join(MIXED, "mixed_val_meta.json"))["active_attributes"]

    print("loading OLD ...")
    old_model = build_model(OLD_CKPT)
    print("loading NEW ...")
    new_model = build_model(NEW_CKPT)

    old_mat = direction_matrix(old_model, attrs)
    new_mat = direction_matrix(new_model, attrs)

    n = len(attrs)
    off_mask = ~np.eye(n, dtype=bool)
    old_mean_abs = float(np.mean(np.abs(old_mat[off_mask])))
    new_mean_abs = float(np.mean(np.abs(new_mat[off_mask])))
    print("mean |off-diagonal cosine| old=%.4f new=%.4f delta=%+.4f"
          % (old_mean_abs, new_mean_abs, new_mean_abs - old_mean_abs))

    rows = [["mean_abs_offdiag", old_mean_abs, new_mean_abs]]
    for a, b in ENTANGLED_PAIRS + OPPOSITE_PAIRS:
        ov = pair_cos(old_mat, attrs, a, b)
        nv = pair_cos(new_mat, attrs, a, b)
        kind = "entangled" if (a, b) in ENTANGLED_PAIRS else "opposite"
        rows.append(["%s_%s_%s" % (kind, a, b), float(ov), float(nv)])
        print("%s/%s: old=%+.4f new=%+.4f" % (a, b, ov, nv))

    with open(os.path.join(BASE, "attribute_direction_cosine.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "old_cos", "new_cos"])
        w.writerows([[r[0], round(r[1], 4), round(r[2], 4)] for r in rows])

    # 热力图对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    vmin, vmax = -1, 1
    for ax, mat, title in ((axes[0], old_mat, "旧：拉丁 LoRA"),
                           (axes[1], new_mat, "新：混合 LoRA（S0）")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_xticklabels(attrs, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(attrs, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    fig.suptitle("属性方向余弦矩阵（属性纠缠）对比", fontsize=13)
    fig.savefig(os.path.join(BASE, "attribute_direction_heatmap.png"),
                dpi=140, bbox_inches="tight")
    print("DIRECTION ANALYSIS DONE")


if __name__ == "__main__":
    main()

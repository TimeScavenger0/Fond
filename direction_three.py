"""
三模型属性方向与纠缠对比：OLD（拉丁 LoRA）、S0（混合）、FULL（全模块）。

属性方向 d_a = encode_text("a font") - encode_text("not a font")，归一化后
算两两余弦矩阵，量化整体平均绝对余弦、易纠缠对与对立对，判定模块 A 的
正交约束是否真正把纠缠降下来，同时画三张热力图。
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

_BASE_NAME = (
    "ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch48_aug50_lbound_of_scale0.01_"
    "max_attr_num_3_random_p_num_1000_geta0.2_use_negative_lr1e-05-0.1_image_file_dir"
)
CKPTS = {
    "OLD": os.path.join(BASE, "model_checkpoints", "fontclip_lora_best.pt"),
    "S0": os.path.join(BASE, _BASE_NAME + ".pt"),
    "FULL": os.path.join(BASE, _BASE_NAME + "_orth0.1-b4.0_attrh1.0_cs1.0-o0.1-r0.5_anchor1.0.pt"),
}

ENTANGLED_PAIRS = [
    ("serif", "formal"),
    ("thin", "delicate"),
    ("strong", "angular"),
]
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
    n = len(attrs)
    off_mask = ~np.eye(n, dtype=bool)

    mats = {}
    for tag, ckpt in CKPTS.items():
        assert os.path.exists(ckpt), ckpt
        print("loading %s ..." % tag)
        mats[tag] = direction_matrix(build_model(ckpt), attrs)

    mean_abs = {tag: float(np.mean(np.abs(m[off_mask]))) for tag, m in mats.items()}
    print("mean |off-diag|: OLD=%.4f S0=%.4f FULL=%.4f"
          % (mean_abs["OLD"], mean_abs["S0"], mean_abs["FULL"]))
    print("FULL-S0=%+.4f FULL-OLD=%+.4f"
          % (mean_abs["FULL"] - mean_abs["S0"], mean_abs["FULL"] - mean_abs["OLD"]))

    rows = [["mean_abs_offdiag", mean_abs["OLD"], mean_abs["S0"], mean_abs["FULL"]]]
    for a, b in ENTANGLED_PAIRS + OPPOSITE_PAIRS:
        kind = "entangled" if (a, b) in ENTANGLED_PAIRS else "opposite"
        vals = {tag: pair_cos(mats[tag], attrs, a, b) for tag in CKPTS}
        rows.append(["%s_%s_%s" % (kind, a, b), vals["OLD"], vals["S0"], vals["FULL"]])
        print("%s/%s: OLD=%+.4f S0=%+.4f FULL=%+.4f"
              % (a, b, vals["OLD"], vals["S0"], vals["FULL"]))

    with open(os.path.join(BASE, "direction_three_cosine.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "OLD", "S0", "FULL"])
        w.writerows([[r[0], round(r[1], 4), round(r[2], 4), round(r[3], 4)] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    titles = {"OLD": "OLD 拉丁 LoRA", "S0": "S0 混合 LoRA", "FULL": "FULL 全模块 A+B+C+D"}
    im = None
    for ax, tag in zip(axes, ("OLD", "S0", "FULL")):
        im = ax.imshow(mats[tag], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(titles[tag])
        ax.set_xticks(range(n))
        ax.set_xticklabels(attrs, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(attrs, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    fig.suptitle("属性方向余弦矩阵（属性纠缠）三模型对比", fontsize=14)
    fig.savefig(os.path.join(BASE, "direction_three_heatmap.png"),
                dpi=130, bbox_inches="tight")
    print("DIRECTION THREE DONE")


if __name__ == "__main__":
    main()

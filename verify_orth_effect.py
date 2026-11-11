"""
模块 A 有效性验证：从 S0 best 起点，仅用属性方向正交损失优化若干步，
测量训练前后属性方向余弦（纠缠）是否下降，验证模块 A 机理而非仅验证管线。

为聚焦因果，本脚本不加载图像数据、不跑 BCE，只对文本编码器的 LoRA 与 ln_final
施加正交梯度。输出整体平均绝对非对角余弦与三个易纠缠对的前后对比。
"""
import json
import os

import numpy as np
import torch

from dataset.dataset import MyDataset
from models.init_model import device, load_model
from models.lora import LoRAConfig
from utils.tokenizer import tokenize

BASE = os.path.dirname(os.path.abspath(__file__))
MIXED = os.path.join(BASE, "data_cn", "mixed")
S0_CKPT = os.path.join(
    BASE,
    "ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch48_aug50_lbound_of_scale0.01_"
    "max_attr_num_3_random_p_num_1000_geta0.2_use_negative_lr1e-05-0.1_image_file_dir.pt",
)
ENTANGLED = [("serif", "formal"), ("thin", "delicate"), ("strong", "angular")]
STEPS = 150
ORTH_W = 0.1
BOOST = 4.0


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build():
    lora = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    model = load_model(
        S0_CKPT,
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
    # 只训练文本 LoRA 与 ln_final，其余冻结，聚焦正交机理
    for name, p in model.named_parameters():
        p.requires_grad = ("lora" in name) or name.startswith("ln_final")
    return model


def direction_cosine(model, attrs):
    pos = tokenize([MyDataset.generate_prompt(a) for a in attrs]).to(device)
    neg = tokenize([MyDataset.generate_prompt(a, negative=True) for a in attrs]).to(device)
    with torch.no_grad():
        d = model.encode_text(pos) - model.encode_text(neg)
    d = d / torch.norm(d, dim=1, keepdim=True)
    return (d @ d.T).cpu().numpy()


def orth_loss(model, attrs):
    pos = tokenize([MyDataset.generate_prompt(a) for a in attrs]).to(device)
    neg = tokenize([MyDataset.generate_prompt(a, negative=True) for a in attrs]).to(device)
    d = model.encode_text(pos) - model.encode_text(neg)
    d = d / torch.norm(d, dim=1, keepdim=True)
    cosine = d @ d.T
    n = len(attrs)
    w = torch.full((n, n), ORTH_W, device=device, dtype=cosine.dtype)
    for a, b in ENTANGLED:
        i, j = attrs.index(a), attrs.index(b)
        w[i, j] = w[j, i] = ORTH_W * BOOST
    upper = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    return torch.sum(w[upper] * cosine[upper].pow(2))


def summarize(mat, attrs):
    n = len(attrs)
    off = ~np.eye(n, dtype=bool)
    out = {"mean_abs": float(np.mean(np.abs(mat[off])))}
    for a, b in ENTANGLED:
        out["%s_%s" % (a, b)] = float(mat[attrs.index(a), attrs.index(b)])
    return out


def main():
    attrs = load_json(os.path.join(MIXED, "mixed_val_meta.json"))["active_attributes"]
    model = build()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable))
    optimizer = torch.optim.Adam(trainable, lr=1e-4, betas=(0.9, 0.98), eps=1e-6)

    before = summarize(direction_cosine(model, attrs), attrs)
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        loss = orth_loss(model, attrs)
        loss.backward()
        optimizer.step()
        if (step + 1) % 30 == 0:
            print("step %d orth_loss=%.4f" % (step + 1, loss.item()))

    model.eval()
    after = summarize(direction_cosine(model, attrs), attrs)

    print("\nmetric                  before      after       delta")
    for k in before:
        print("%-22s  %+.4f    %+.4f    %+.4f"
              % (k, before[k], after[k], after[k] - before[k]))
    print("ORTH VERIFY DONE")


if __name__ == "__main__":
    main()

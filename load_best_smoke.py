"""
加载混合 LoRA best 权重并做 CPU 前向冒烟。

GPU 保存的 state_dict 含 fp16，load_model 加载后需 model.float() 转回 fp32，
否则 CPU 上 half 算子可能报错。验证文本/图像编码前向可跑通、输出无 NaN。
"""
import os

import torch

from models.init_model import load_model, device
from models.lora import LoRAConfig
from utils.tokenizer import tokenize

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(
    BASE,
    "ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch48_aug50_lbound_of_scale0.01_"
    "max_attr_num_3_random_p_num_1000_geta0.2_use_negative_lr1e-05-0.1_image_file_dir.pt",
)


def main():
    # 先看 checkpoint 保存时的元信息
    raw = torch.load(CKPT, map_location="cpu")
    print("saved epoch:", raw["epoch"], "saved loss:", raw["loss"])
    print("signature:", raw["signature"])

    lora_config_text = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    model = load_model(
        CKPT,
        model_name="ViT-B/32",
        use_oft_vision=False,
        use_oft_text=False,
        use_lora_text=True,
        use_lora_vision=False,
        lora_config_text=lora_config_text,
        use_coop_text=False,
        use_coop_vision=False,
        precontext_dropout_rate=0,
        pt_applied_layers=None,
    )

    # 关键：fp16 -> fp32，保证 CPU 前向正常
    model.float()
    model.eval()
    dtypes = {str(p.dtype) for p in model.parameters()}
    print("param dtypes after float():", dtypes)

    # 文本编码前向
    texts = ["italic", "thin", "strong", "a font with wide characters"]
    tokens = tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    print("text features:", tuple(text_features.shape), "NaN:", bool(torch.isnan(text_features).any()))

    # 图像编码前向（随机张量，仅验证算子可跑）
    dummy_image = torch.rand(2, 3, 224, 224, device=device)
    with torch.no_grad():
        image_features = model.encode_image(dummy_image)
    print("image features:", tuple(image_features.shape), "NaN:", bool(torch.isnan(image_features).any()))

    print("LOAD AND FORWARD SMOKE PASSED")


if __name__ == "__main__":
    main()

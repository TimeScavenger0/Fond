"""
检查 Featurize 混合 LoRA best 权重的内部结构。

GPU 上 convert_weights 把模型转 fp16 后保存，故需先弄清：
顶层字段、参数总数、LoRA 参数数、dtype 分布、ln_final/visual.proj 是否存在，
为后续 fp16->fp32 加载与中文效果验证做准备。
"""
import os
from collections import Counter

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(
    BASE,
    "ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch48_aug50_lbound_of_scale0.01_"
    "max_attr_num_3_random_p_num_1000_geta0.2_use_negative_lr1e-05-0.1_image_file_dir.pt",
)


def main():
    ckpt = torch.load(CKPT, map_location="cpu")
    print("top-level keys:", list(ckpt.keys()))

    sd = ckpt["model_state_dict"]
    print("total tensors:", len(sd))

    dtype_counter = Counter(str(t.dtype) for t in sd.values())
    print("dtype distribution:", dict(dtype_counter))

    lora_keys = [k for k in sd if "lora" in k.lower()]
    print("lora tensors:", len(lora_keys))

    ln_final_keys = [k for k in sd if "ln_final" in k]
    print("ln_final tensors:", ln_final_keys)

    visual_proj_keys = [k for k in sd if "visual.proj" in k or k.endswith("proj")]
    print("proj-like tensors:", visual_proj_keys[:10])

    # 抽样看几个 LoRA 参数的 dtype 与形状
    print("sample lora entries:")
    for k in lora_keys[:4]:
        t = sd[k]
        print("  ", k, tuple(t.shape), t.dtype)

    # 统计可训练参数总量
    total_params = sum(t.numel() for t in sd.values())
    print("total params:", total_params)


if __name__ == "__main__":
    main()

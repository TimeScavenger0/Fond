"""
中文 + 拉丁混合 LoRA 续训入口（训练方案第五节 S0）。

从 model_checkpoints/fontclip_lora_best.pt 加载现有拉丁 LoRA（文本编码器，
r=256、alpha=1024、q/k/v/out），在混合数据上续训，仅主 BCE 对比损失（S0）。
默认 --smoke 在本地 CPU 用极小配置跑通管线；正式训练用命令行参数覆盖。

跨平台：项目根目录基于本文件位置自动探测，不硬编码盘符。本地 Windows 直接使用
原始字体/图片路径；云端 Linux 上原始路径不存在时，自动把字体重映射到 --cloud-fonts
目录、把图片目录重映射到项目内 data_cn/mixed，因此同一份代码两端可跑。

用法：
    python train_cn_mixed.py --smoke
    python train_cn_mixed.py --epoch 100 --batch_size 48 --lr 1e-5
"""
import argparse
import json
import os

from models.lora import LoRAConfig
from trains.config_fine_tune import Config
from trains.trainer import Trainer

# 项目根：本文件位于项目根，跨平台自动探测
BASE = os.path.dirname(os.path.abspath(__file__))
MIXED_DIR = os.path.join(BASE, "data_cn", "mixed")
DEFAULT_CLOUD_FONTS = os.path.join(BASE, "cloud_fonts")
CHECKPOINT = os.path.join(BASE, "model_checkpoints", "fontclip_lora_best.pt")


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _basename_any(path):
    """跨平台提取文件名：同时识别 Windows 反斜杠与 Unix 正斜杠。

    Linux 上 os.path.basename 只认正斜杠，对 JSON 里的 Windows 路径
    (C:\\Windows\\Fonts\\x.TTF) 会原样返回整串，故统一替换后再切分。
    """
    return path.replace("\\", "/").rstrip("/").split("/")[-1]


def resolve_font_path(original, cloud_font_dir):
    """原始路径存在（本地）则直接用；否则按 basename 重映射到云端字体目录。"""
    if original and os.path.exists(original):
        return original
    candidate = os.path.join(cloud_font_dir, _basename_any(original))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"font not found: {original} (also tried {candidate})")


def resolve_image_dir(original, root):
    """原始图片目录存在（本地）则用；否则映射到项目内 data_cn/mixed/<basename>。"""
    if original and os.path.exists(original):
        return original
    candidate = os.path.join(root, "data_cn", "mixed", _basename_any(original))
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"image dir not found: {original} (also tried {candidate})")


def main():
    parser = argparse.ArgumentParser(description="Mixed CN+Latin LoRA continuation training")
    parser.add_argument("--smoke", action="store_true", help="本地 CPU 极小配置跑通管线")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sample_num_each_epoch", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--max_sample_num", type=int, default=3)
    parser.add_argument("--rotation_degrees", type=float, default=10)
    parser.add_argument("--target_cn_share", type=float, default=0.4)
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT)
    parser.add_argument("--cloud-fonts", dest="cloud_font_dir", default=DEFAULT_CLOUD_FONTS,
                        help="云端字体目录（原始路径缺失时按 basename 在此查找）")
    # S1 模块 A：属性方向软正交
    parser.add_argument("--use_orth", action="store_true", help="S1：启用属性方向正交损失")
    parser.add_argument("--orth_weight", type=float, default=0.1, help="正交基础权重")
    parser.add_argument("--orth_boost", type=float, default=4.0, help="易纠缠对权重倍数")
    # S2 模块 B：独立属性判别头
    parser.add_argument("--use_attr_heads", action="store_true", help="S2：启用独立属性判别头")
    parser.add_argument("--attr_head_weight", type=float, default=1.0)
    # S3 模块 C/D：内容-风格分离与结构锚定
    parser.add_argument("--use_content_style", action="store_true", help="S3：启用内容-风格子空间分离")
    parser.add_argument("--cs_style_weight", type=float, default=1.0)
    parser.add_argument("--cs_orth_weight", type=float, default=0.1)
    parser.add_argument("--cs_recon_weight", type=float, default=0.5)
    parser.add_argument("--use_structure_anchor", action="store_true", help="S3：启用结构锚定（模块D）")
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    args = parser.parse_args()

    # ---- 读取混合训练/验证数据 ----
    train_meta = load_json(os.path.join(MIXED_DIR, "mixed_dataset_meta.json"))
    val_meta = load_json(os.path.join(MIXED_DIR, "mixed_val_meta.json"))
    train_font_paths_raw = load_json(os.path.join(MIXED_DIR, "mixed_font_paths.json"))
    val_font_paths_raw = load_json(os.path.join(MIXED_DIR, "mixed_val_font_paths.json"))

    # ---- 跨平台路径重映射（字体）----
    train_font_paths = [resolve_font_path(p, args.cloud_font_dir) for p in train_font_paths_raw]
    val_font_paths = [resolve_font_path(p, args.cloud_font_dir) for p in val_font_paths_raw]

    train_json = os.path.join(MIXED_DIR, "mixed_font_to_attribute_values.json")
    val_json = os.path.join(MIXED_DIR, "mixed_val_font_to_attribute_values.json")

    # ---- 跨平台路径重映射（图片目录）----
    train_images = resolve_image_dir(train_meta["mixed_images_dir"], BASE)
    val_images = resolve_image_dir(val_meta["images_dir"], BASE)

    exclusive = train_meta["exclusive_attributes"]
    cn_names = train_meta["cn_names"]

    # ---- LoRA 配置：与 checkpoint 完全一致（文本编码器 q/k/v/out）----
    lora_config_text = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    # 视觉侧不启用 LoRA（checkpoint 无视觉 LoRA），config 仅占位
    lora_config_vision = LoRAConfig()

    # ---- smoke 极小配置 ----
    if args.smoke:
        epoch = 1
        batch_size = 4
        sample_num_each_epoch = 2
        sample_num = 2
        max_sample_num = 2
        print("SMOKE MODE: 极小配置，仅验证管线可跑通")
    else:
        epoch = args.epoch
        batch_size = args.batch_size
        sample_num_each_epoch = args.sample_num_each_epoch
        sample_num = args.sample_num
        max_sample_num = args.max_sample_num

    print("BASE:", BASE)
    print("train images:", train_images)
    print("val images:", val_images)
    print("checkpoint:", args.checkpoint_path)

    config = Config(
        # 数据路径
        font_dir=train_font_paths,
        val_font_dir=val_font_paths,
        json_path=train_json,
        validation_json_path_override=val_json,
        image_file_dir=train_images,
        image_file_dir_for_validation=val_images,
        # 续训
        checkpoint_path=args.checkpoint_path,
        use_lora_text=True,
        use_lora_vision=False,
        lora_config_text=lora_config_text,
        lora_config_vision=lora_config_vision,
        # 混合训练扩展
        exclusive_attributes_override=exclusive,
        rotation_degrees=args.rotation_degrees,
        target_sampled_cn_share=args.target_cn_share,
        cn_font_names=cn_names,
        train_text_ln_final=True,
        # S1 模块 A
        use_orth_loss=args.use_orth,
        orth_loss_weight=args.orth_weight,
        orth_entangled_boost=args.orth_boost,
        # S2 模块 B
        use_attr_heads=args.use_attr_heads,
        attr_head_weight=args.attr_head_weight,
        # S3 模块 C/D
        use_content_style=args.use_content_style,
        cs_style_weight=args.cs_style_weight,
        cs_orth_weight=args.cs_orth_weight,
        cs_recon_weight=args.cs_recon_weight,
        use_structure_anchor=args.use_structure_anchor,
        anchor_weight=args.anchor_weight,
        # 训练超参
        EPOCH=epoch,
        BATCH_SIZE=batch_size,
        sample_num_each_epoch=sample_num_each_epoch,
        sample_num=sample_num,
        max_sample_num=max_sample_num,
        lr=args.lr,
        # 与拉丁主配置一致的口径
        use_negative=True,
        use_multiple_attributes=True,
        use_color_jitter=False,
        geta=0.2,
    )

    trainer = Trainer(config)
    result = trainer.train()
    print("training finished:", result)


if __name__ == "__main__":
    main()

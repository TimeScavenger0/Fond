"""
FontCLIP Stroke-Level Local Optimization
Deform only ONE stroke/region of a letter while keeping the rest fixed.

Works by zeroing the gradient of the frozen control points each step
(gradient masking), so the selected region moves and everything else stays put.

USAGE: edit `region` below to one of: "left" | "right" | "cross" | "counter" | "outer"
then run:  python run_stroke_opt.py
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from easydict import EasyDict as edict
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

from optimizer.optimize import (
    Config,
    draw_image_through_svg_from_font_path,
    get_signature,
    train,
    change_size_transform,
    device,
)
from optimizer.util_tools import update


def get_config(
    optimized_letter,
    font_path,
    visual_font_path=None,
    num_iter=501,
    use_lr_scheduler=True,
    target_image_path=None,
    semantic_concept=None,
    do_preprocess=True,
    lr=1.0,
    use_aug=False,
    use_single_character_image=False,
    image_file_path=None,
    use_L2_loss=False,
    L2_loss_w=1.0,
    use_visual_encoder=False,
    use_fclip=False,
    fclip_loss_w=1.0,
    use_fclip_direction_loss=False,
    use_fclip_direction_loss_vision=False,
    use_fclip_direction_loss_only=False,
    ref_semantic_concept=None,
    ref_image_file_path=None,
    fclip_direction_loss_w=1.0,
    checkpoint_path=None,
    use_conformal_loss=False,
    conformal_loss_w=1.0,
    use_tone_loss=False,
    tone_loss_w=1.0,
    use_tone_loss_schedular=False,
    use_G1_loss=False,
    G1_loss_w=1.0,
    skip_corner_threshold=-0.95,
    skip_corners=True,
    use_laplacian_loss=False,
    laplacian_loss_w=0.2,
    is_counter=False,
    skip_edge_laplacian=False,
    only_edge_laplacian=False,
    use_laplacian_between_beziers_loss=False,
    laplacian_between_beziers_loss_w=1.0,
    laplacian_between_beziers_loss_threshold=-0.8,
    use_Xing_loss=False,
    Xing_loss_w=1.0,
    use_direction_loss=False,
    direction_loss_w=1.0,
):
    visual_optimize = True
    if semantic_concept is not None:
        visual_optimize = False

    char_size = 150
    size = 200

    multiple_attributes = False
    multiple_attributes_preserve_init = False
    target_attributes = ["thin", "formal", "legible"]
    reduce_cp = False
    epsilon = 50
    target_attributes_weights = [fclip_loss_w] + [
        fclip_loss_w / 10 for _ in range(len(target_attributes) - 1)
    ]
    use_cos_loss = False
    cos_loss_w = 0.05
    is_counter = is_counter
    num_per_curve = 8
    skip_control_points = True
    skip_corners = skip_corners
    skip_corner_threshold = skip_corner_threshold
    skip_edge_laplacian = skip_edge_laplacian
    only_edge_laplacian = only_edge_laplacian
    skip_edge_cos = True
    use_lr_schedular = use_lr_scheduler

    cfg = Config(
        font_path,
        optimized_letter,
        None,
        optimized_letter,
        use_wandb=USE_WANDB,
        wandb_user=WANDB_USER,
        num_iter=num_iter,
        char_size=char_size,
        size=size,
        do_preprocess=do_preprocess,
        lr=lr,
        use_aug=use_aug,
        use_visual_encoder=use_visual_encoder,
        fclip_loss=use_fclip,
        fclip_loss_w=fclip_loss_w,
        use_fclip_direction_loss=use_fclip_direction_loss,
        use_fclip_direction_loss_vision=use_fclip_direction_loss_vision,
        use_fclip_direction_loss_only=use_fclip_direction_loss_only,
        ref_semantic_concept=ref_semantic_concept,
        ref_image_file_path=ref_image_file_path,
        fclip_direction_loss_w=fclip_direction_loss_w,
        laplacian_loss_w=laplacian_loss_w,
        cos_loss_w=cos_loss_w,
        G1_loss_w=G1_loss_w,
        L2_loss_w=L2_loss_w,
        conformal_loss_w=conformal_loss_w,
        tone_loss_w=tone_loss_w,
        multiple_attributes=multiple_attributes,
        multiple_text_encoders=multiple_attributes_preserve_init,
        target_attributes=target_attributes,
        target_attributes_weights=target_attributes_weights,
        use_tone_loss=use_tone_loss,
        use_conformal_loss=use_conformal_loss,
        use_laplacian_loss=use_laplacian_loss,
        use_cos_loss=use_cos_loss,
        use_G1_loss=use_G1_loss,
        reduce_cp=reduce_cp,
        epsilon=epsilon,
        checkpoint_path=checkpoint_path,
        visual_optimize=visual_optimize,
        image_file_path=image_file_path,
        use_tone_loss_schedular=use_tone_loss_schedular,
        use_lr_scheduler=use_lr_schedular,
        use_L2_loss=use_L2_loss,
        is_counter=is_counter,
        num_per_curve=num_per_curve,
        skip_control_points=skip_control_points,
        skip_corners=skip_corners,
        skip_corner_threshold=skip_corner_threshold,
        skip_edge_laplacian=skip_edge_laplacian,
        skip_edge_cos=skip_edge_cos,
        only_edge_laplacian=only_edge_laplacian,
        use_laplacian_between_beziers_loss=use_laplacian_between_beziers_loss,
        laplacian_between_beziers_loss_w=laplacian_between_beziers_loss_w,
        laplacian_between_beziers_loss_threshold=laplacian_between_beziers_loss_threshold,
        use_Xing_loss=use_Xing_loss,
        Xing_loss_w=Xing_loss_w,
        use_direction_loss=use_direction_loss,
        direction_loss_w=direction_loss_w,
    )

    with open(cfg.config, "r") as f:
        cfg_full = yaml.load(f, Loader=yaml.Loader)

    cfg_key = cfg.experiment
    cfgs = [vars(cfg)]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get("parent_config", "baseline")

    tmp_cfg = edict()
    for options in reversed(cfgs):
        update(tmp_cfg, options)
    cfg = tmp_cfg
    del cfgs

    if target_image_path is not None:
        target_img = to_tensor(
            Image.open(target_image_path).convert("L").convert("RGB")
        )
        target_img = change_size_transform(size)(target_img).to(device)
    else:
        if image_file_path is not None:
            target_img = Image.open(image_file_path)
            target_img = to_tensor(target_img)
        else:
            target_img = draw_image_through_svg_from_font_path(
                visual_font_path,
                optimized_letter,
                svg_size=size,
                char_size=char_size,
                image_size=size,
            )
    cfg.target_img = target_img
    cfg.semantic_concept = semantic_concept

    return cfg


USE_WANDB = 0
WANDB_USER = ""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="FontCLIP stroke-level local optimization (deform only one stroke)"
    )
    parser.add_argument(
        "--region",
        default="left",
        choices=["left", "right", "cross", "counter", "outer"],
        help="which stroke to move; the rest stay fixed",
    )
    parser.add_argument("--num_iter", type=int, default=100, help="optimization steps")
    parser.add_argument("--letter", default="A", help="character to optimize")
    parser.add_argument("--font_path", default="gwfonts/Arimo-Regular.ttf")
    parser.add_argument("--semantic_concept", default="italic, slanted font")
    parser.add_argument("--lr", type=float, default=1.5)
    args = parser.parse_args()

    # ====== Configuration ======
    font_path = args.font_path
    optimized_letter = args.letter
    semantic_concept = args.semantic_concept

    # WHICH STROKE to move. Others stay fixed.
    region = args.region

    checkpoint_path = "model_checkpoints/fontclip_lora_best.pt"
    num_iter = args.num_iter  # CPU mode; keep moderate for local tweaks
    lr = args.lr

    fclip_loss_w = 3.5
    conformal_loss_w = 5.0
    tone_loss_w = 5.0
    # For local deformation we keep L2 OFF so the moving stroke isn't pulled
    # back toward the original shape. Turn it on (use_L2_loss=True, L2_loss_w~0.5)
    # if the moving stroke drifts too far / collapses.

    print("=" * 60)
    print("FontCLIP Stroke-Level Local Optimization")
    print("=" * 60)
    print(f"  Letter:          {optimized_letter}")
    print(f"  Font:            {font_path}")
    print(f"  Region to move:  {region}")
    print(f"  Semantic target: {semantic_concept}")
    print(f"  Checkpoint:      {checkpoint_path}")
    print(f"  Iterations:      {num_iter}")
    print(f"  Device:          {device}")
    print("=" * 60)

    cfg = get_config(
        optimized_letter,
        font_path,
        visual_font_path=font_path,
        num_iter=num_iter,
        use_lr_scheduler=True,
        semantic_concept=semantic_concept,
        do_preprocess=True,
        lr=lr,
        use_aug=True,
        use_fclip=True,
        fclip_loss_w=fclip_loss_w,
        checkpoint_path=checkpoint_path,
        use_conformal_loss=True,
        conformal_loss_w=conformal_loss_w,
        use_tone_loss=True,
        tone_loss_w=tone_loss_w,
        use_L2_loss=False,
    )

    # Inject the stroke-region flag read by optimizer/optimize.py train()
    cfg.optimize_region = region

    signature = get_signature(cfg, font_path, None, None)
    print(f"\nExperiment signature: {signature}")
    print(f"Output directory: {cfg.experiment_dir}")
    print("\nStarting optimization...\n")

    train(cfg, signature)

    print("\n" + "=" * 60)
    print("Stroke-local optimization complete!")
    print(f"Output SVG: {os.path.join(cfg.experiment_dir, 'output-svg', 'output.svg')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

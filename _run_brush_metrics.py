# -*- coding: utf-8 -*-
"""Run brush local editing experiment and compute new metrics."""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image, ImageDraw
from optimizer.optimize import Config, train
import yaml
from easydict import EasyDict as edict
from optimizer.util_tools import update

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'S', 'T']

def create_brush_mask(size=200, cx=100, cy=120, radius=40):
    """Create a circular brush mask in the lower half of the glyph."""
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=255)
    return np.array(mask) > 127

def run_brush_experiment():
    results = []
    brush_mask = create_brush_mask()
    
    for letter in letters:
        print(f"\n=== Brush experiment | {letter} ===")
        cfg = Config(
            "gwfonts/Arimo-Regular.ttf", letter, None, letter,
            num_iter=300, char_size=150, size=200,
            do_preprocess=True, lr=1.5, use_aug=True,
            use_visual_encoder=False, fclip_loss=True, fclip_loss_w=2.0,
            use_fclip_direction_loss=False, use_fclip_direction_loss_vision=False,
            use_fclip_direction_loss_only=False,
            ref_semantic_concept=None, ref_image_file_path=None, fclip_direction_loss_w=1.0,
            laplacian_loss_w=0.2, cos_loss_w=0.05, G1_loss_w=1.0, L2_loss_w=1.0,
            conformal_loss_w=8.0, tone_loss_w=5.0,
            multiple_attributes=False, multiple_text_encoders=False,
            target_attributes=["thin", "formal", "legible"],
            target_attributes_weights=[2.0, 0.2, 0.2],
            use_tone_loss=True, use_conformal_loss=True, use_laplacian_loss=False,
            use_cos_loss=False, use_G1_loss=False,
            reduce_cp=False, epsilon=50, checkpoint_path="model_checkpoints/fontclip_lora_best.pt",
            visual_optimize=False, image_file_path=None,
            use_tone_loss_schedular=False, use_lr_scheduler=True,
            use_L2_loss=False, is_counter=False, num_per_curve=8,
            skip_control_points=True, skip_corners=True, skip_corner_threshold=-0.95,
            skip_edge_laplacian=False, skip_edge_cos=True, only_edge_laplacian=False,
            use_laplacian_between_beziers_loss=False,
            laplacian_between_beziers_loss_w=1.0,
            laplacian_between_beziers_loss_threshold=-0.8,
            use_Xing_loss=False, Xing_loss_w=1.0,
            use_direction_loss=False, direction_loss_w=1.0,
        )
        with open(cfg.config, "r") as f:
            full = yaml.load(f, Loader=yaml.Loader)
        key = cfg.experiment
        stack = [vars(cfg)]
        while key:
            stack.append(full[key])
            key = stack[-1].get("parent_config", "baseline")
        merged = edict()
        for opt in reversed(stack):
            update(merged, opt)
        merged.semantic_concept = "italic, slanted font"
        merged.use_structure_reg = True
        merged.use_vgg_loss = False
        merged.ref_image_path = None
        merged.two_stage_lr = True
        merged.num_samples = 3
        merged.anchor_gradient_scale = 0.3
        merged.brush_mask_image = brush_mask
        
        merged.experiment_dir = f"output/brush_test_{letter}"
        os.makedirs(merged.experiment_dir, exist_ok=True)
        for sub in ["", "output-svg", "output-png", "video-png"]:
            os.makedirs(os.path.join(merged.experiment_dir, sub), exist_ok=True)
        
        t0 = time.time()
        final_loss = train(merged, f"brush_test_{letter}")
        elapsed = time.time() - t0
        
        results.append({"letter": letter, "final_loss": final_loss, "time": elapsed})
        print(f"  {letter}: loss={final_loss:.4f}, time={elapsed:.1f}s")
    
    out_path = "output/brush_experiment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    avg_loss = np.mean([r["final_loss"] for r in results])
    print(f"\n=== Brush experiment avg loss: {avg_loss:.4f} ===")
    return results

if __name__ == "__main__":
    run_brush_experiment()

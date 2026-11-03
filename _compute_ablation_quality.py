# -*- coding: utf-8 -*-
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import pydiffvg

def render_svg(svg_path, size=200):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=img.device
    ) * (1 - img[:, :, 3:4])
    img = img[:, :, :3].detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("L")

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'S', 'T']
configs = ["no_two_stage_lr", "no_best_of_3", "no_structure_reg", "no_vgg_loss", "no_anchor_scale"]

all_results = {}
for config in configs:
    print(f"\n=== Computing quality metrics for {config} ===")
    config_results = []
    for letter in letters:
        init_svg = f"output/ablation_{config}_{letter}/svg-init/init.svg"
        out_svg = f"output/ablation_{config}_{letter}/output-svg/output.svg"
        
        if not os.path.exists(init_svg) or not os.path.exists(out_svg):
            print(f"  {letter}: missing SVG files")
            continue
        
        init_img = np.array(render_svg(init_svg))
        out_img = np.array(render_svg(out_svg))
        
        rmse = np.sqrt(mean_squared_error(init_img, out_img))
        psnr = peak_signal_noise_ratio(init_img, out_img, data_range=255)
        ssim = structural_similarity(init_img, out_img, data_range=255)
        
        config_results.append({"letter": letter, "rmse": round(rmse, 2), "psnr": round(psnr, 2), "ssim": round(ssim, 4)})
        print(f"  {letter}: RMSE={rmse:.2f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    
    if config_results:
        avg_rmse = np.mean([r["rmse"] for r in config_results])
        avg_psnr = np.mean([r["psnr"] for r in config_results])
        avg_ssim = np.mean([r["ssim"] for r in config_results])
        all_results[config] = {
            "per_letter": config_results,
            "avg_rmse": round(avg_rmse, 2),
            "avg_psnr": round(avg_psnr, 2),
            "avg_ssim": round(avg_ssim, 4)
        }
        print(f"  Average: RMSE={avg_rmse:.2f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

with open("output/ablation_quality_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n=== Summary ===")
for config, data in all_results.items():
    print(f"{config}: RMSE={data['avg_rmse']:.2f}, PSNR={data['avg_psnr']:.2f}, SSIM={data['avg_ssim']:.4f}")

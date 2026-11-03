# -*- coding: utf-8 -*-
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

results = []
for letter in letters:
    our_init_svg = f"output/eval_{letter}/svg-init/init.svg"
    our_out_svg = f"output/eval_{letter}/output-svg/output.svg"
    base_init_svg = f"output/baseline_{letter}/svg-init/init.svg"
    base_out_svg = f"output/baseline_{letter}/output-svg/output.svg"
    
    row = {"letter": letter}
    
    if os.path.exists(our_init_svg) and os.path.exists(our_out_svg):
        init_img = np.array(render_svg(our_init_svg))
        out_img = np.array(render_svg(our_out_svg))
        our_rmse = np.sqrt(mean_squared_error(init_img, out_img))
        our_psnr = peak_signal_noise_ratio(init_img, out_img, data_range=255)
        our_ssim = structural_similarity(init_img, out_img, data_range=255)
        row["our_rmse"] = round(our_rmse, 2)
        row["our_psnr"] = round(our_psnr, 2)
        row["our_ssim"] = round(our_ssim, 4)
    
    if os.path.exists(base_init_svg) and os.path.exists(base_out_svg):
        init_img = np.array(render_svg(base_init_svg))
        out_img = np.array(render_svg(base_out_svg))
        base_rmse = np.sqrt(mean_squared_error(init_img, out_img))
        base_psnr = peak_signal_noise_ratio(init_img, out_img, data_range=255)
        base_ssim = structural_similarity(init_img, out_img, data_range=255)
        row["base_rmse"] = round(base_rmse, 2)
        row["base_psnr"] = round(base_psnr, 2)
        row["base_ssim"] = round(base_ssim, 4)
    
    results.append(row)
    print(f"{letter}: our_rmse={row.get('our_rmse', 'N/A')}, base_rmse={row.get('base_rmse', 'N/A')}")

our_rmse = [r["our_rmse"] for r in results if "our_rmse" in r]
our_psnr = [r["our_psnr"] for r in results if "our_psnr" in r]
our_ssim = [r["our_ssim"] for r in results if "our_ssim" in r]
base_rmse = [r["base_rmse"] for r in results if "base_rmse" in r]
base_psnr = [r["base_psnr"] for r in results if "base_psnr" in r]
base_ssim = [r["base_ssim"] for r in results if "base_ssim" in r]

print(f"\nOur Method: RMSE={np.mean(our_rmse):.2f}, PSNR={np.mean(our_psnr):.2f}, SSIM={np.mean(our_ssim):.4f}")
print(f"Baseline:   RMSE={np.mean(base_rmse):.2f}, PSNR={np.mean(base_psnr):.2f}, SSIM={np.mean(base_ssim):.4f}")

with open("output/quality_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

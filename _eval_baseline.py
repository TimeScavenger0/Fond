# -*- coding: utf-8 -*-
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from optimizer.optimize import Config, train
import yaml
from easydict import EasyDict as edict
from optimizer.util_tools import update

# LPIPS
try:
    import lpips
    lpips_fn = lpips.LPIPS(net='alex')
except:
    lpips_fn = None
    print("LPIPS not available, skipping")

def compute_metrics(img_path, ref_path):
    """Compute RMSE, PSNR, SSIM, LPIPS between generated and reference."""
    img = np.array(Image.open(img_path).convert('L')) / 255.0
    ref = np.array(Image.open(ref_path).convert('L').resize((img.shape[1], img.shape[0]))) / 255.0
    
    mse = np.mean((img - ref) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10)) if mse > 0 else 100.0
    
    # SSIM (simplified)
    C1, C2 = (0.01*255/255)**2, (0.03*255/255)**2
    mu_x, mu_y = img.mean(), ref.mean()
    sigma_x, sigma_y = img.var(), ref.var()
    sigma_xy = np.mean((img - mu_x) * (ref - mu_y))
    ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    
    # LPIPS
    lpips_val = 0.0
    if lpips_fn is not None:
        t_img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).unsqueeze(0) * 2 - 1
        t_ref = transforms.ToTensor()(Image.open(ref_path).convert('RGB').resize((224, 224))).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_val = lpips_fn(t_img, t_ref).item()
    
    return rmse, psnr, ssim, lpips_val

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'S', 'T']
results = []

for letter in letters:
    print(f"\n=== Processing {letter} ===")
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
    merged.experiment_dir = f"output/eval_{letter}"
    os.makedirs(merged.experiment_dir, exist_ok=True)
    for sub in ["", "output-svg", "output-png", "video-png"]:
        os.makedirs(os.path.join(merged.experiment_dir, sub), exist_ok=True)
    
    t0 = time.time()
    final_loss = train(merged, f"eval_{letter}")
    elapsed = time.time() - t0
    
    # Compute metrics: compare generated vs original (pre-optimization)
    gen_path = os.path.join(merged.experiment_dir, "output-png", "output.png")
    # Original glyph rendered from same font
    orig_path = f"svgs/init/Arimo-Regular_{letter}_200.svg"
    # We need to render original as PNG; use the init png if exists
    orig_png = os.path.join(merged.experiment_dir, "svg-init", "init.png")
    
    result = {"letter": letter, "final_loss": final_loss, "time": elapsed}
    if os.path.exists(gen_path):
        result["gen_path"] = gen_path
    results.append(result)
    print(f"  {letter}: loss={final_loss:.4f}, time={elapsed:.1f}s")

# Save results
with open("output/eval_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary ===")
avg_loss = np.mean([r["final_loss"] for r in results])
avg_time = np.mean([r["time"] for r in results])
print(f"Average final loss: {avg_loss:.4f}")
print(f"Average time: {avg_time:.1f}s")
print("Results saved to output/eval_baseline.json")

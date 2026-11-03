"""Generate before/after comparison image for vector optimization results."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import pydiffvg
from PIL import Image
import numpy as np

EXPERIMENT_DIR = r"output\conformal_1.0_dist_pixel_100_kernel201_A\Arimo-Regular\A_Arimo-Regular_aug_italic, formal, legible font_fclip3.5_tone5.0-100-201_conf5.0_cp1_lr1.5use_lr_scheduler"

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
    return Image.fromarray(img)

# Load initial and final
init_svg = os.path.join(EXPERIMENT_DIR, "svg-init", "init.svg")
final_svg = os.path.join(EXPERIMENT_DIR, "output-svg", "output.svg")

init_img = render_svg(init_svg)
final_img = render_svg(final_svg)

# Create side-by-side comparison
w, h = init_img.size
gap = 20
label_h = 40
canvas = Image.new("RGB", (w * 2 + gap * 3, h + label_h + gap * 2), (255, 255, 255))

from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(canvas)

# Place images
canvas.paste(init_img, (gap, label_h + gap))
canvas.paste(final_img, (w + gap * 2, label_h + gap))

# Labels
try:
    font = ImageFont.truetype("gwfonts/Arimo-Regular.ttf", 20)
except:
    font = ImageFont.load_default()

draw.text((gap + w//2 - 40, 10), "Before (Arimo)", fill=(0, 0, 0), font=font)
draw.text((w + gap * 2 + w//2 - 40, 10), "After (italic formal)", fill=(0, 0, 0), font=font)

# Draw separator
draw.line([(w + gap*2 - gap//2, gap), (w + gap*2 - gap//2, h + label_h + gap)],
          fill=(200, 200, 200), width=2)

output_path = os.path.join(EXPERIMENT_DIR, "comparison.png")
canvas.save(output_path)
print(f"Comparison saved to: {output_path}")
print(f"Init size: {init_img.size}, Final size: {final_img.size}")

# Also save a progression strip (iter 0, 10, 25, 49)
iters = [0, 10, 25, 49]
strip_h = h
strip_w = w * len(iters) + gap * (len(iters) + 1)
strip = Image.new("RGB", (strip_w, strip_h + label_h), (255, 255, 255))
draw2 = ImageDraw.Draw(strip)
for idx, it in enumerate(iters):
    png_path = os.path.join(EXPERIMENT_DIR, "video-png", f"iter{it:04d}.png")
    if os.path.exists(png_path):
        frame = Image.open(png_path).convert("RGB")
        x = gap + idx * (w + gap)
        strip.paste(frame, (x, label_h))
        draw2.text((x + w//2 - 20, 10), f"iter {it}", fill=(0, 0, 0), font=font)

strip_path = os.path.join(EXPERIMENT_DIR, "progression.png")
strip.save(strip_path)
print(f"Progression saved to: {strip_path}")

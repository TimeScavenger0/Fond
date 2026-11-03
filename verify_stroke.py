"""Verify stroke-local optimization: frozen points must not move, moving region must change."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import pydiffvg

DIR = r"output\conformal_1.0_dist_pixel_100_kernel201_A\Arimo-Regular\A_Arimo-Regular_aug_italic, slanted font_fclip3.5_tone5.0-100-201_conf5.0_cp1_lr1.5use_lr_scheduler"
INIT = os.path.join(DIR, "svg-init", "init.svg")
OUT = os.path.join(DIR, "output-svg", "output.svg")


def load(svg):
    w, h, shapes, groups = pydiffvg.svg_to_scene(svg)
    return w, h, shapes


wi, hi, si = load(INIT)
wo, ho, so = load(OUT)

p0i, p0o = si[0].points, so[0].points
mid = (p0i[:, 0].min() + p0i[:, 0].max()) / 2
right = p0i[:, 0] >= mid
left = p0i[:, 0] < mid

d_right = (p0o[right] - p0i[right]).abs().max().item()
d_left = (p0o[left] - p0i[left]).abs().max().item()
p1i, p1o = si[1].points, so[1].points
d_counter = (p1o - p1i).abs().max().item()

print(f"shape0 points: {p0i.shape[0]}")
print(f"  frozen RIGHT stroke ({right.sum().item()} pts)  max displacement = {d_right:.4f}")
print(f"  moving  LEFT  stroke ({left.sum().item()} pts)  max displacement = {d_left:.4f}")
print(f"shape1 counter ({p1i.shape[0]} pts, frozen) max displacement = {d_counter:.4f}")

print("\n=> VERDICT")
print("  RIGHT stroke frozen:", "PASS (no movement)" if d_right < 1e-3 else "FAIL (moved!)")
print("  counter    frozen:", "PASS (no movement)" if d_counter < 1e-3 else "FAIL (moved!)")
print("  LEFT  stroke moved:", "PASS (deformed)" if d_left > 0.5 else "WEAK (barely moved)")

# Render before/after comparison
render = pydiffvg.RenderFunction.apply


def render_shapes(shapes, w, h):
    g = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor(list(range(len(shapes))), dtype=torch.int32),
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    scene = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, [g])
    img = render(w, h, 2, 2, 0, None, *scene)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3
    ) * (1 - img[:, :, 3:4])
    return img.detach().cpu()


from PIL import Image
import numpy as np

im0 = render_shapes(si, wi, hi)
im1 = render_shapes(so, wo, ho)
a = (im0.numpy() * 255).astype(np.uint8)
b = (im1.numpy() * 255).astype(np.uint8)
gap = 30
canvas = Image.new("RGB", (a.shape[1] * 2 + gap * 3, a.shape[0] + gap * 2), (255, 255, 255))
canvas.paste(Image.fromarray(a), (gap, gap))
canvas.paste(Image.fromarray(b), (a.shape[1] + gap * 2, gap))
canvas.save(os.path.join(DIR, "stroke_compare.png"))
print("\nSaved stroke_compare.png")

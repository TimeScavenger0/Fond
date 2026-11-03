# -*- coding: utf-8 -*-
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, yaml
from easydict import EasyDict as edict
from optimizer.optimize import Config, my_preprocess
from optimizer.util_tools import update
from models.init_model import load_model, device, preprocess as clip_preprocess
from models.lora import LoRAConfig
import pydiffvg
from optimizer.optimize import render_init, draw_image_through_svg_from_font_path
from optimizer.losses import FCLIPLoss

cfg = Config("gwfonts/Arimo-Regular.ttf", "M", None, "M",
             num_iter=10, char_size=150, size=200,
             do_preprocess=True, lr=1.5, use_aug=False, fclip_loss=True,
             checkpoint_path="model_checkpoints/fontclip_lora_best.pt",
             fclip_loss_w=3.5, conformal_loss_w=0.0, tone_loss_w=0.0)

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
merged.experiment_dir = "output/diag"
os.makedirs(merged.experiment_dir, exist_ok=True)

# Load model
model = load_model(
    checkpoint_path=merged.checkpoint_path,
    model_name="ViT-B/32",
    use_lora_text=True,
    lora_config_text=LoRAConfig(r=256, alpha=1024.0, bias=False,
                                 learnable_alpha=False, apply_q=True,
                                 apply_k=True, apply_v=True, apply_out=True),
)
model = model.to(device)

fclip_loss = FCLIPLoss(merged, clip_model=model, device=device,
                       preprocess=my_preprocess, clip_preprocess=clip_preprocess,
                       image_init=None)

print(f"Loss weight = {fclip_loss.loss_weight}")

# Render initial M
canvas, shapes, groups, ind = render_init(merged, "M")
render = pydiffvg.RenderFunction(sample_width=2, sample_height=2)
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas, shapes, groups)
img = render(200, 200, 2, 2, 0, None, *scene_args)

# Check cosine similarity before optimization
with torch.no_grad():
    x0 = img.unsqueeze(0).permute(0, 3, 1, 2)
    emb_x = model.encode_image(my_preprocess(x0))
    emb_t = fclip_loss.embedded_text
    cs = torch.cosine_similarity(emb_x, emb_t, dim=-1)
    print(f"Initial cos_sim = {cs.item():.6f}")

# One optimization step
x = img.unsqueeze(0).permute(0, 3, 1, 2)
loss = fclip_loss(x, step=None)
print(f"Initial fclip loss = {loss.item():.6f}")
loss.backward()

# Check gradients
total_grad_norm = 0.0
for si, s in enumerate(shapes):
    if hasattr(s, 'points') and s.points.grad is not None:
        gn = s.points.grad.norm().item()
        total_grad_norm += gn
        print(f"  Shape {si} points grad norm = {gn:.6f}, max abs = {s.points.grad.abs().max().item():.6f}")
    elif hasattr(s, 'points'):
        print(f"  Shape {si} points grad = None!")

print(f"Total control point grad norm = {total_grad_norm:.6f}")

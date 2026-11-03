# -*- coding: utf-8 -*-
"""Baseline: run global style optimization on representative letters and record CLIP similarity."""
import os, sys, time, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
import pydiffvg

from optimizer.optimize import Config, train, draw_image_through_svg_from_font_path, device
from models.init_model import load_model
from models.lora import LoRAConfig
from utils.tokenizer import tokenize as clip_tokenize

FONT = "gwfonts/Arimo-Regular.ttf"
STYLE = "italic, slanted font"
LETTERS = ["A", "B", "C", "E", "H", "K", "M", "O", "S", "T"]
NUM_ITER = 100
LR = 1.5
SIZE = 200
OUT_ROOT = "output/baseline"
CHECKPOINT = "model_checkpoints/fontclip_lora_best.pt"

# Load scoring model (same as train())
lora_cfg = LoRAConfig(r=256, alpha=1024.0, bias=False, learnable_alpha=False,
                      apply_q=True, apply_k=True, apply_v=True, apply_out=True)
score_model = load_model(checkpoint_path=CHECKPOINT, model_name="ViT-B/32",
                         use_lora_text=True, lora_config_text=lora_cfg)
score_model.eval()
print("Scoring model loaded.")

# Get the transform used by train()
from models.init_model import my_preprocess
transform = my_preprocess


def render_svg(svg_path):
    w, h, shapes, groups = pydiffvg.svg_to_scene(svg_path)
    scene = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, groups)
    img = pydiffvg.RenderFunction.apply(w, h, 2, 2, 0, None, *scene)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
    return img[:, :, :3]


@torch.no_grad()
def clip_sim(img_tensor, text):
    arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[0] in (1, 3) and arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.concatenate([arr]*3, axis=-1)
    pil = Image.fromarray(arr)
    x = transform(pil).unsqueeze(0).to(device)
    t = clip_tokenize([text]).to(device)
    i_emb = score_model.encode_image(x)
    t_emb = score_model.encode_text(t)
    i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    return float((i_emb * t_emb).sum().item())


def build_cfg_for(letter):
    cfg = Config(FONT, letter, None, letter,
                 num_iter=NUM_ITER, char_size=150, size=SIZE,
                 do_preprocess=True, lr=LR, use_aug=True, fclip_loss=True,
                 checkpoint_path=CHECKPOINT)
    import yaml
    from easydict import EasyDict as edict
    from optimizer.util_tools import update
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
    merged.semantic_concept = STYLE
    merged.target_img = draw_image_through_svg_from_font_path(
        FONT, letter, svg_size=SIZE, char_size=150, image_size=SIZE)
    return merged


results = []
os.makedirs(OUT_ROOT, exist_ok=True)

for letter in LETTERS:
    print(f"\n===== {letter} =====")
    ldir = os.path.join(OUT_ROOT, letter)
    os.makedirs(ldir, exist_ok=True)

    cfg = build_cfg_for(letter)
    cfg.experiment_dir = ldir
    sig = f"base_{letter}"

    pre = clip_sim(cfg.target_img, STYLE)
    print(f"  pre: {pre:.4f}")

    t0 = time.time()
    train(cfg, sig)
    elapsed = time.time() - t0

    out_svg = os.path.join(ldir, "output-svg", "output.svg")
    if os.path.exists(out_svg):
        post_t = render_svg(out_svg)
        post = clip_sim(post_t, STYLE)
        Image.fromarray((post_t.cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(ldir, "result.png"))
    else:
        post = -1.0
        print(f"  WARN: no output.svg")

    print(f"  post: {post:.4f}  delta: {post-pre:+.4f}  time: {elapsed:.1f}s")
    results.append({"letter": letter, "pre": round(pre, 4), "post": round(post, 4),
                    "delta": round(post - pre, 4), "time": round(elapsed, 1)})

print("\n===== Summary =====")
print(f"{'L':<4}{'pre':>8}{'post':>8}{'delta':>8}{'t(s)':>8}")
for r in results:
    print(f"{r['letter']:<4}{r['pre']:>8.4f}{r['post']:>8.4f}{r['delta']:>8.4f}{r['time']:>8.1f}")
avg_pre = np.mean([r["pre"] for r in results])
avg_post = np.mean([r["post"] for r in results if r["post"] > 0])
avg_t = np.mean([r["time"] for r in results])
print(f"{'AVG':<4}{avg_pre:>8.4f}{avg_post:>8.4f}{avg_post-avg_pre:>8.4f}{avg_t:>8.1f}")

with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump({"config": {"font": FONT, "style": STYLE, "iter": NUM_ITER, "lr": LR},
               "results": results, "avg_pre": round(avg_pre, 4),
               "avg_post": round(avg_post, 4), "avg_delta": round(avg_post-avg_pre, 4),
               "avg_time": round(avg_t, 1)}, f, indent=2)

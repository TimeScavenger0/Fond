# -*- coding: utf-8 -*-
"""V2 baseline: two-stage LR + best-of-3 sampling."""
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
LETTERS = ["A", "B", "M"]
NUM_ITER = 100
LR = 1.5
SIZE = 200
OUT_ROOT = "output/baseline_v2"
CHECKPOINT = "model_checkpoints/fontclip_lora_best.pt"
N_TRIALS = 3

lora_cfg = LoRAConfig(r=256, alpha=1024.0, bias=False, learnable_alpha=False,
                      apply_q=True, apply_k=True, apply_v=True, apply_out=True)
score_model = load_model(checkpoint_path=CHECKPOINT, model_name="ViT-B/32",
                         use_lora_text=True, lora_config_text=lora_cfg)
score_model.eval()
from models.init_model import my_preprocess
transform = my_preprocess
print("Scoring model loaded.")


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
                 checkpoint_path=CHECKPOINT,
                 two_stage_lr=True, two_stage_switch=0.6, two_stage_decay=0.2)
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
    print(f"\n===== {letter} (best of {N_TRIALS}) =====")
    ldir = os.path.join(OUT_ROOT, letter)
    os.makedirs(ldir, exist_ok=True)

    cfg = build_cfg_for(letter)
    pre = clip_sim(cfg.target_img, STYLE)
    print(f"  pre: {pre:.4f}")

    best_post = -999
    best_trial = -1
    for trial in range(N_TRIALS):
        print(f"  --- trial {trial+1}/{N_TRIALS} ---")
        # fresh cfg each trial (re-render target)
        cfg_t = build_cfg_for(letter)
        cfg_t.experiment_dir = os.path.join(ldir, f"trial{trial+1}")
        for sub in ["", "output-svg", "output-png", "video-png"]:
            os.makedirs(os.path.join(cfg_t.experiment_dir, sub), exist_ok=True)
        sig = f"v2_{letter}_t{trial+1}"
        t0 = time.time()
        train(cfg_t, sig)
        elapsed = time.time() - t0
        out_svg = os.path.join(cfg_t.experiment_dir, "output-svg", "output.svg")
        if os.path.exists(out_svg):
            post_t = render_svg(out_svg)
            post = clip_sim(post_t, STYLE)
            print(f"  trial {trial+1}: post={post:.4f}  delta={post-pre:+.4f}  time={elapsed:.1f}s")
            if post > best_post:
                best_post = post
                best_trial = trial + 1
                # save best result
                Image.fromarray((post_t.cpu().numpy()*255).astype(np.uint8)).save(
                    os.path.join(ldir, "result.png"))
        else:
            print(f"  trial {trial+1}: no output.svg")

    print(f"  BEST: trial {best_trial}, post={best_post:.4f}, delta={best_post-pre:+.4f}")
    results.append({"letter": letter, "pre": round(pre,4), "post": round(best_post,4),
                    "delta": round(best_post-pre,4), "best_trial": best_trial})

print("\n===== V2 Summary =====")
for r in results:
    print(f"  {r['letter']}: pre={r['pre']:.4f} post={r['post']:.4f} delta={r['delta']:+.4f} (trial {r['best_trial']})")
avg_pre = np.mean([r["pre"] for r in results])
avg_post = np.mean([r["post"] for r in results])
print(f"  AVG: pre={avg_pre:.4f} post={avg_post:.4f} delta={avg_post-avg_pre:+.4f}")

with open(os.path.join(OUT_ROOT, "results_v2.json"), "w") as f:
    json.dump({"config": {"font": FONT, "style": STYLE, "iter": NUM_ITER, "lr": LR,
                          "two_stage": True, "best_of": N_TRIALS},
               "results": results, "avg_delta": round(avg_post-avg_pre, 4)}, f, indent=2)

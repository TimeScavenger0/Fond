# -*- coding: utf-8 -*-
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from models.init_model import load_model, device, preprocess as clip_preprocess
from models.lora import LoRAConfig
from utils.tokenizer import tokenize

lora_config_text = LoRAConfig(
    r=256, alpha=1024.0, bias=False, learnable_alpha=False,
    apply_q=True, apply_k=True, apply_v=True, apply_out=True,
)
model = load_model(
    "model_checkpoints/fontclip_lora_best.pt",
    model_name="ViT-B/32",
    use_oft_vision=False, use_oft_text=False,
    oft_config_vision=None, oft_config_text=None,
    use_lora_text=True, use_lora_vision=False,
    lora_config_vision=None, lora_config_text=lora_config_text,
    use_coop_text=False, use_coop_vision=False,
    precontext_length_vision=10, precontext_length_text=77,
    precontext_dropout_rate=0, pt_applied_layers=None,
)
model.eval()

def render_init_char(letter, font_path="gwfonts/Arimo-Regular.ttf", size=200, char_size=150):
    """Render initial character with PIL."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, char_size)
    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), letter, fill=(0, 0, 0), font=font)
    return img

def compute_clip_score(img, text_prompt):
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    text_tokens = tokenize([text_prompt]).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        text_feat = model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        score = (img_feat @ text_feat.T).item()
    return score

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'S', 'T']
prompt = "italic, slanted font"

results = []
for letter in letters:
    # Our method output
    our_out = f"output/eval_{letter}/output-png/output.png"
    # Baseline output
    base_out = f"output/baseline_{letter}/output-png/output.png"
    
    # Init char (rendered with PIL)
    init_img = render_init_char(letter)
    
    row = {"letter": letter}
    
    # Init score (same for both methods, since both start from same font)
    init_score = compute_clip_score(init_img, prompt)
    row["init_score"] = round(init_score, 4)
    
    if os.path.exists(our_out):
        our_img = Image.open(our_out).convert("RGB")
        our_post = compute_clip_score(our_img, prompt)
        row["our_post"] = round(our_post, 4)
        row["our_delta"] = round(our_post - init_score, 4)
    
    if os.path.exists(base_out):
        base_img = Image.open(base_out).convert("RGB")
        base_post = compute_clip_score(base_img, prompt)
        row["base_post"] = round(base_post, 4)
        row["base_delta"] = round(base_post - init_score, 4)
    
    results.append(row)
    print(f"{letter}: our_delta={row.get('our_delta', 'N/A')}, base_delta={row.get('base_delta', 'N/A')}")

our_deltas = [r["our_delta"] for r in results if "our_delta" in r]
base_deltas = [r["base_delta"] for r in results if "base_delta" in r]
print(f"\nOur Method avg delta: {np.mean(our_deltas):.4f}")
print(f"Baseline avg delta: {np.mean(base_deltas):.4f}")

with open("output/clip_score_results.json", "w") as f:
    json.dump(results, f, indent=2)

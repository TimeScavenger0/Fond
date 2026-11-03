# -*- coding: utf-8 -*-
"""
FontCLIP Brush-Based Stroke Optimization
========================================
Render a character, let the user paint which stroke(s) to deform on a
Gradio canvas, then optimize only those control points toward a semantic
description while keeping everything else frozen.

USAGE:
    python run_brush_opt.py

Flow:
  1. Type letter / font / semantic description, click "Render glyph".
  2. Paint the stroke you want to deform (brush) on the canvas.
  3. Click "Start optimization". The painted region moves, everything else stays.
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import gradio as gr
from torchvision.transforms.functional import to_pil_image
from skimage.measure import find_contours, approximate_polygon

from optimizer.optimize import (
    get_signature,
    train,
    draw_image_through_svg_from_font_path,
    device,
)
from optimizer.util_tools import update
import yaml
from easydict import EasyDict as edict
from PIL import Image

SIZE = 200
CHAR_SIZE = 150
CHECKPOINT = "model_checkpoints/fontclip_lora_best.pt"
USE_WANDB = 0
WANDB_USER = ""


def raster_to_svg(img_path, out_svg_path, canvas_size=SIZE, approx_tolerance=0.8):
    """Convert a raster glyph image (black on white) to an SVG with path(s).

    Uses marching-squares contour extraction, then Douglas-Peucker polygon
    simplification (approx_tolerance in pixels) to keep the control-point count
    close to that of a real TTF glyph (~100 points) instead of hundreds of
    raw line segments. Fewer, well-spaced points make the diffvg optimization
    stable and prevent self-intersection / tangled strokes.
    """
    img = Image.open(img_path).convert("L").resize((canvas_size, canvas_size), Image.LANCZOS)
    arr = np.array(img)
    # binary: glyph = dark pixels
    binary = arr < 128

    contours = find_contours(binary.astype(float), level=0.5)
    # keep contours with enough points; drop tiny specks
    contours = [c for c in contours if len(c) > 10]
    contours = sorted(contours, key=len, reverse=True)
    if not contours:
        raise ValueError("No contour found in uploaded image.")

    paths = []
    for contour in contours:
        # contour shape (N,2): (row=y, col=x); close the loop
        pts = np.array([[float(x), float(y)] for y, x in contour])
        # Douglas-Peucker simplification: reduce N points to ~tolerance-scaled set
        pts = approximate_polygon(pts, tolerance=approx_tolerance)
        if len(pts) < 3:
            continue

        # Convert the polygon to a smooth closed quadratic-Bezier path.
        # Each original polygon vertex becomes a Q control point; the curve
        # passes through the midpoint of every edge. This replaces jagged
        # line segments (L commands) with smooth curves like a real font.
        n = len(pts)
        midpoints = (pts + np.roll(pts, -1, axis=0)) / 2.0
        d = "M %.2f %.2f " % (midpoints[0][0], midpoints[0][1])
        for i in range(n):
            cx, cy = pts[i][0], pts[i][1]
            nx, ny = midpoints[(i + 1) % n][0], midpoints[(i + 1) % n][1]
            d += "Q %.2f %.2f %.2f %.2f " % (cx, cy, nx, ny)
        d += "Z"
        paths.append(d)

    if not paths:
        raise ValueError("No usable contour after simplification.")

    svg_body = '\n'.join(
        '  <path d="%s" fill="black" fill-rule="nonzero"/>' % d for d in paths
    )
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="%d" height="%d" viewBox="0 0 %d %d">\n'
        '%s\n</svg>'
    ) % (canvas_size, canvas_size, canvas_size, canvas_size, svg_body)

    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return out_svg_path


# ---------- config builder (reused from run_stroke_opt.py) ----------
def build_cfg(letter, font_path, semantic_concept, num_iter, lr, init_svg_path=None):
    char_size = CHAR_SIZE
    size = SIZE
    fclip_loss_w = 3.5
    conformal_loss_w = 5.0
    tone_loss_w = 5.0

    # minimal config mirroring run_stroke_opt.py defaults
    from optimizer.optimize import Config

    do_preprocess = init_svg_path is None
    cfg = Config(
        font_path, letter, None, letter,
        use_wandb=USE_WANDB, wandb_user=WANDB_USER,
        num_iter=num_iter, char_size=char_size, size=size,
        do_preprocess=do_preprocess, lr=lr, use_aug=True,
        use_visual_encoder=False, fclip_loss=True, fclip_loss_w=fclip_loss_w,
        use_fclip_direction_loss=False, use_fclip_direction_loss_vision=False,
        use_fclip_direction_loss_only=False,
        ref_semantic_concept=None, ref_image_file_path=None, fclip_direction_loss_w=1.0,
        laplacian_loss_w=0.2, cos_loss_w=0.05, G1_loss_w=1.0, L2_loss_w=1.0,
        conformal_loss_w=conformal_loss_w, tone_loss_w=tone_loss_w,
        multiple_attributes=False, multiple_text_encoders=False,
        target_attributes=["thin", "formal", "legible"],
        target_attributes_weights=[fclip_loss_w, fclip_loss_w / 10, fclip_loss_w / 10],
        use_tone_loss=True, use_conformal_loss=True, use_laplacian_loss=False,
        use_cos_loss=False, use_G1_loss=False,
        reduce_cp=False, epsilon=50, checkpoint_path=CHECKPOINT,
        visual_optimize=False, image_file_path=None,
        use_tone_loss_schedular=False, use_lr_scheduler=True,
        use_L2_loss=False, is_counter=False, num_per_curve=8,
        skip_control_points=True, skip_corners=True, skip_corner_threshold=-0.95,
        skip_edge_laplacian=False, skip_edge_cos=True, only_edge_laplacian=False,
        use_laplacian_between_beziers_loss=False,
        laplacian_between_beziers_loss_w=1.0, laplacian_between_beziers_loss_threshold=-0.8,
        use_Xing_loss=False, Xing_loss_w=1.0,
        use_direction_loss=False, direction_loss_w=1.0,
    )

    with open(cfg.config, "r") as f:
        cfg_full = yaml.load(f, Loader=yaml.Loader)
    cfg_key = cfg.experiment
    cfgs = [vars(cfg)]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get("parent_config", "baseline")
    tmp_cfg = edict()
    for options in reversed(cfgs):
        update(tmp_cfg, options)
    cfg = tmp_cfg
    del cfgs

    if init_svg_path:
        # Start from an existing SVG (e.g. previous optimization output)
        cfg.target = init_svg_path
        import pydiffvg
        w, h, shapes, groups = pydiffvg.svg_to_scene(init_svg_path)
        scene = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, groups)
        render = pydiffvg.RenderFunction.apply
        img = render(w, h, 2, 2, 0, None, *scene)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
        cfg.target_img = img[:, :, :3].detach()
    else:
        target_img = draw_image_through_svg_from_font_path(
            font_path, letter, svg_size=size, char_size=char_size, image_size=size,
        )
        cfg.target_img = target_img
    cfg.semantic_concept = semantic_concept
    return cfg


# ---------- global state ----------
STATE = {"init_pil": None, "letter": None, "font_path": None, "init_svg_path": None}


def render_glyph(letter, font_path, uploaded_image=None):
    """Render the initial glyph to a numpy image for the brush canvas.
    If uploaded_image is given, raster-to-SVG it first and use that as base.
    Render glyph always starts fresh from TTF / uploaded image; chained
    refinement happens by painting on the result canvas and re-optimizing.
    """
    letter = letter.strip()
    font_path = font_path.strip()

    # Render glyph always resets chained state: user explicitly wants a fresh start.
    STATE["chained_ready"] = False

    init_svg_path = None
    if uploaded_image is not None:
        # Gradio passes a file path when type="filepath"
        try:
            tmp_svg = os.path.join("output", "_tmp_uploaded.svg")
            os.makedirs(os.path.dirname(tmp_svg), exist_ok=True)
            raster_to_svg(uploaded_image, tmp_svg)
            init_svg_path = tmp_svg
        except Exception as e:
            return None, f"ERROR: failed to vectorize uploaded image: {e}"

    if init_svg_path:
        # Show the ORIGINAL uploaded photo on the canvas (not the re-rendered
        # vector approximation). The SVG is only used internally for optimization;
        # both are 200x200 so mask pixel coordinates map directly to path coords.
        orig = Image.open(uploaded_image).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        STATE["init_pil"] = orig
        STATE["letter"] = letter
        STATE["font_path"] = font_path
        STATE["init_svg_path"] = init_svg_path
        arr = np.array(orig)
        return arr, f"Uploaded image loaded. Vectorized internally at {init_svg_path}. Paint the stroke to deform."

    if not letter or not os.path.exists(font_path):
        return None, "ERROR: invalid letter or font path."
    t = draw_image_through_svg_from_font_path(
        font_path, letter, svg_size=SIZE, char_size=CHAR_SIZE, image_size=SIZE,
    )
    pil = to_pil_image(t.detach().cpu())
    STATE["init_pil"] = pil
    STATE["letter"] = letter
    STATE["font_path"] = font_path
    STATE["init_svg_path"] = None
    arr = np.array(pil)
    return arr, f"Rendered '{letter}' from {font_path}. Now paint the stroke you want to deform."


ATTR_KEYS = ["serif", "italic", "bold", "thin", "elegant", "playful", "sharp", "soft"]


def _attrs_to_prompt(vals):
    """Build a prompt string where slider strength becomes intensifier words.
    0.5 = neutral. 0.6-0.75 = slightly, 0.75-0.9 = bare, 0.9+ = very.
    Below 0.5 uses non- prefix.
    """
    parts = []
    for k, v in zip(ATTR_KEYS, vals):
        dev = abs(v - 0.5) * 2.0
        if dev < 0.1:
            continue
        word = k if v > 0.5 else f"non-{k}"
        if dev < 0.3:
            parts.append(f"slightly {word}")
        elif dev < 0.7:
            parts.append(word)
        else:
            parts.append(f"very {word}")
    return ", ".join(parts)


def run_optimization(brush_input, semantic_concept, num_iter, lr,
                     serif, italic, bold, thin, elegant, playful, sharp, soft,
                     style_ref_path=None, ref_weight=1.0):
    """Called when user clicks 'Start optimization'.

    brush_input: dict {"image": np.ndarray(H,W,3), "mask": np.ndarray(H,W)}
                 when source=upload + tool=sketch in gradio 3.40.
    """
    if STATE["init_pil"] is None:
        return None, "ERROR: render a glyph first."

    slider_vals = [serif, italic, bold, thin, elegant, playful, sharp, soft]
    attr_str = _attrs_to_prompt(slider_vals)
    compound_concept = f"{semantic_concept}, {attr_str}" if attr_str else semantic_concept
    print(f"[compound prompt] {compound_concept}")
    if style_ref_path:
        print(f"[style reference image] {style_ref_path}")

    if brush_input is None:
        # No interaction at all: treat as global optimization.
        mask = None
    elif isinstance(brush_input, dict):
        mask = brush_input.get("mask")
    else:
        mask = brush_input  # fallback: treat whole input as mask

    if mask is None:
        # No mask painted -> global optimization on the whole glyph.
        binary_mask = np.zeros((SIZE, SIZE), dtype=bool)
        n_painted = 0
    else:
        mask = np.array(mask)
        print(f"[debug] raw mask shape: {mask.shape}, dtype: {mask.dtype}, "
              f"min={mask.min()}, max={mask.max()}")

        if mask.ndim == 3:
            # sketch mask may be RGBA: use alpha channel as the paint mask
            if mask.shape[2] == 4:
                mask = mask[..., 3]
            else:
                mask = mask[..., 0]

        # Resize mask to match the 200x200 render canvas so control-point
        # coordinates (which are in 0..200 pixel space) align with the painted area.
        from PIL import Image as _PILImage
        if mask.shape[0] != SIZE or mask.shape[1] != SIZE:
            print(f"[debug] resizing mask from {mask.shape} to ({SIZE}, {SIZE})")
            mask_pil = _PILImage.fromarray(mask)
            mask = np.array(mask_pil.resize((SIZE, SIZE), _PILImage.NEAREST))

        # normalize: painted pixels are bright (white) on dark/transparent background
        binary_mask = mask > 127
        n_painted = int(binary_mask.sum())

    letter = STATE["letter"]
    font_path = STATE["font_path"]

    # Sanitize concept for filesystem and keep it SHORT to avoid Windows MAX_PATH.
    # Use a fixed short output dir per letter; compound prompt still goes to CLIP.
    fs_concept = compound_concept.replace(", ", "-").replace(",", "-").replace(" ", "_")[:40]
    cfg = build_cfg(letter, font_path, fs_concept, int(num_iter), float(lr),
                    init_svg_path=STATE.get("init_svg_path"))

    # Wire up reference-image style migration (FontDiffuser-style visual loss)
    if style_ref_path and os.path.exists(style_ref_path):
        cfg.ref_image_path = style_ref_path
        cfg.ref_image_loss_w = float(ref_weight)
        print(f"[ref_image_loss] enabled, weight={ref_weight}, source={style_ref_path}")

    if n_painted > 0:
        cfg.brush_mask_image = binary_mask  # picked up by train()
        mode_msg = f"Painted pixels: {n_painted}. Local stroke optimization..."
    else:
        mode_msg = "No brush on canvas. Running GLOBAL style optimization on the whole glyph..."

    # Override experiment dir to a short fixed path to avoid WinError 206.
    import time
    best_loss = float('inf')
    best_svg = None
    n_samples = 3
    for trial in range(n_samples):
        short_dir = os.path.join("output", f"brush_trial{trial}")
        cfg.experiment_dir = short_dir
        signature = f"brush_{letter}_{int(time.time())}_{trial}"
        out_dir = cfg.experiment_dir
        for sub in ["", "output-svg", "output-png", "video-png"]:
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        print(f"\nExperiment trial {trial+1}/{n_samples}: {signature}")
        print(f"Output directory: {out_dir}")
        print(mode_msg, end="\n\n")
        final_loss = train(cfg, signature)
        print(f"  trial {trial+1} final loss = {final_loss:.6f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_svg = os.path.join(out_dir, "output-svg", "output.svg")
            # Copy best result to brush_latest for preview
            import shutil
            latest_dir = os.path.join("output", "brush_latest")
            os.makedirs(latest_dir, exist_ok=True)
            for sub in ["output-svg", "output-png"]:
                src = os.path.join(out_dir, sub)
                dst = os.path.join(latest_dir, sub)
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    print(f"\nBest trial loss = {best_loss:.6f}, SVG = {best_svg}")

    # load output svg and render it for preview
    out_svg = os.path.join(out_dir, "output-svg", "output.svg")
    import pydiffvg
    w, h, shapes, groups = pydiffvg.svg_to_scene(out_svg)
    scene = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(w, h, 2, 2, 0, None, *scene)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
    result_np = (img[:, :, :3].detach().cpu().numpy() * 255).astype(np.uint8)

    # Do NOT chain automatically: each new optimization starts fresh from the
    # original TTF. The user must click "Render glyph" to load a deformed result
    # back onto the canvas if they want chained refinement.
    STATE["init_svg_path"] = None
    STATE["chained_ready"] = False

    return result_np, (
        f"Done! Output SVG: {out_svg}\n"
        f"Movable points were limited to your painted region; everything else stayed frozen.\n"
        f"To refine further, adjust the prompt/sliders and click Start optimization again "
        f"(starts fresh from the original font), or click Render glyph to reload a specific letter."
    )


# ---------- Gradio UI ----------
import glob as _glob
_font_choices = sorted(
    "gwfonts/" + os.path.basename(p) for p in _glob.glob("gwfonts/*.ttf")
)
if not _font_choices:
    _font_choices = ["gwfonts/Arimo-Regular.ttf"]

with gr.Blocks(title="FontCLIP Brush Stroke Optimizer") as demo:
    gr.Markdown("# FontCLIP Brush Stroke Optimizer\nRender a glyph, paint the stroke you want to deform, and optimize it toward a semantic style.")

    with gr.Row():
        letter_in = gr.Textbox(label="Letter", value="A", scale=1)
        font_in = gr.Dropdown(label="Font", choices=_font_choices,
                              value="gwfonts/Arimo-Regular.ttf", scale=2)
        uploaded_in = gr.Image(label="Upload a glyph photo (optional, overrides TTF)",
                               type="filepath", image_mode="L", scale=2)
        render_btn = gr.Button("1. Render glyph", variant="primary")

    with gr.Row():
        canvas = gr.Image(
            label="Paint the stroke to deform (white = will move)",
            source="upload",
            tool="sketch",
            type="numpy",
            brush_radius=8,
            brush_color="#ffffff",
            mask_opacity=0.7,
            height=SIZE,
            width=SIZE,
        )

    with gr.Row():
        semantic_in = gr.Textbox(label="Semantic target (main)", value="italic, slanted font", scale=3)
        iter_in = gr.Number(label="Iterations", value=300, scale=1)
        lr_in = gr.Number(label="Learning rate", value=1.5, scale=1)

    gr.Markdown("**Attribute sliders (fine-tune style intensity; 0.5 = neutral)**")
    with gr.Row():
        serif_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Serif")
        italic_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Italic")
        bold_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Bold")
        thin_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Thin")
        elegant_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Elegant")
        playful_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Playful")
        sharp_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Sharp")
        soft_in = gr.Slider(0, 1, value=0.5, step=0.05, label="Soft")

    style_ref_in = gr.Image(label="Style reference image (optional: visual style migration)",
                            type="filepath", image_mode="RGB", scale=2)
    ref_weight_in = gr.Slider(0, 5.0, value=1.0, step=0.1,
                              label="Reference image strength (0 = text only, 1 = equal, 2 = visual dominates)")

    run_btn = gr.Button("2. Start optimization", variant="primary")

    status = gr.Textbox(label="Status", lines=3)
    result_img = gr.Image(label="Result", type="numpy")

    render_btn.click(
        fn=render_glyph,
        inputs=[letter_in, font_in, uploaded_in],
        outputs=[canvas, status],
    )
    run_btn.click(
        fn=run_optimization,
        inputs=[canvas, semantic_in, iter_in, lr_in,
                serif_in, italic_in, bold_in, thin_in, elegant_in, playful_in, sharp_in, soft_in,
                style_ref_in, ref_weight_in],
        outputs=[result_img, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, debug=False)

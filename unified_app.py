# -*- coding: utf-8 -*-
"""统一字体系统界面（fontclip 环境，纯 CPU，端口 7861）。

把三条分散的能力整合到同一 Gradio 界面：
  Tab 1 文本/属性驱动：FontCLIP 生成种子锚点 -> FontDiffuser 批量扩散 -> 位图转矢量
  Tab 2 参考图 one-shot：参考图直接驱动 FontDiffuser 批量扩散 -> 位图转矢量
  Tab 3 画笔局部精修：确定性距离场内法线收缩（local_geom），可控可解释

FontDiffuser 依赖在 fontdiffuser 环境，经 env_bridge 以 subprocess 调用。
"""
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import gradio as gr
import pydiffvg
from PIL import Image

from run_brush_opt import build_cfg, SIZE, CHAR_SIZE, _font_choices
from optimizer.optimize import train, Config
from env_bridge import run_batch
from b2v_pipeline import batch_bitmap_to_svg
from b2v_svg import save_svg_full
from local_geom import local_thin

# 默认选用含完整中文字形的黑体，避免默认落在拉丁字体上导致中文 KeyError
DEFAULT_FONT = next(
    (f for f in _font_choices if os.path.basename(f).lower() == "simhei.ttf"),
    _font_choices[0],
)


# ---------------- 基础渲染工具 ----------------
def render_shapes_pil(shapes, groups, size=SIZE):
    scene = pydiffvg.RenderFunction.serialize_scene(size, size, shapes, groups)
    img = pydiffvg.RenderFunction.apply(size, size, 2, 2, 0, None, *scene)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(size, size, 3) * (1 - img[:, :, 3:4])
    return Image.fromarray((img[:, :, :3].detach().clamp(0, 1).numpy() * 255).astype("uint8"))


def new_run_dir(tag):
    d = os.path.join("output", "unified", tag + "_" + str(int(time.time())))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------- 种子锚点 ----------------
def make_seed(seed_char, font_path, concept, n_iter, lr, out_dir):
    exp_dir = os.path.join(out_dir, "seed_run")
    for sub in ["", "output-svg", "output-png", "video-png", "svg-init"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    cfg = build_cfg(seed_char, font_path, concept, int(n_iter), float(lr))
    cfg.experiment_dir = exp_dir
    final_loss = train(cfg, "seed_%s_%d" % (seed_char, int(time.time())))

    out_svg = os.path.join(exp_dir, "output-svg", "output.svg")
    w, h, shapes, groups = pydiffvg.svg_to_scene(out_svg)
    pil200 = render_shapes_pil(shapes, groups, SIZE)
    anchor = os.path.join(out_dir, "seed_anchor.png")
    pil200.resize((96, 96), Image.LANCZOS).save(anchor)
    return anchor, float(final_loss)


# ---------------- 域路由：字符分类与单字符矢量优化 ----------------
def is_vector_native(ch):
    """拉丁字母/数字/常见拉丁标点属于 FontCLIP 矢量训练域，直接矢量优化。"""
    cp = ord(ch)
    if 0x20 <= cp <= 0x7F:      # 基本拉丁 ASCII
        return True
    if 0x80 <= cp <= 0x24F:      # 拉丁扩展、IPA、组合附加
        return True
    if 0x2000 <= cp <= 0x209F:   # 通用标点、上下标
        return True
    if 0x20A0 <= cp <= 0x20CF:   # 货币符号
        return True
    return False


def split_chars(chars):
    """去重、去空白后按训练域分成（矢量字符，扩散字符）。"""
    vec, diff, seen = [], [], set()
    for ch in chars:
        if ch.isspace() or ch in seen:
            continue
        seen.add(ch)
        (vec if is_vector_native(ch) else diff).append(ch)
    return vec, diff


def optimize_vector_char(char, font_path, concept, n_iter, work_dir):
    """单个拉丁/符号字符：FontCLIP 矢量优化，返回输出 SVG 与 final_loss。"""
    cp = ord(char)
    exp_dir = os.path.join(work_dir, "vec_run_u%x" % cp)
    for sub in ["", "output-svg", "output-png", "video-png", "svg-init"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    cfg = build_cfg(char, font_path, concept, int(n_iter), 1.5)
    cfg.experiment_dir = exp_dir
    final_loss = train(cfg, "vec_u%x_%d" % (cp, int(time.time())))
    return os.path.join(exp_dir, "output-svg", "output.svg"), float(final_loss)


def export_vector_result(src_svg, char, svg_dir, preview_dir):
    """把单字符优化结果导出为统一命名 SVG 与预览 PNG。"""
    cp = ord(char)
    w, h, shapes, groups = pydiffvg.svg_to_scene(src_svg)
    dst_svg = os.path.join(svg_dir, "u%x.svg" % cp)
    save_svg_full(dst_svg, w, h, shapes, groups)
    png = os.path.join(preview_dir, "u%x.png" % cp)
    render_shapes_pil(shapes, groups, w).save(png)
    return dst_svg, png


# ---------------- Tab 1：文本/属性驱动 ----------------
def pipeline_text(font_path, seed_char, concept, chars, seed_iter, diff_steps):
    font_path, seed_char, chars = font_path.strip(), seed_char.strip(), chars.strip()
    if not chars:
        yield None, "缺少目标字符集", None
        return
    run_dir = new_run_dir("text")
    svg_dir = os.path.join(run_dir, "svg")
    preview_dir = os.path.join(run_dir, "preview")
    bitmap_dir = os.path.join(run_dir, "bitmap")
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    vec_chars, diff_chars = split_chars(chars)
    total = len(vec_chars) + len(diff_chars)
    result_map = {}
    ok = 0

    # 拉丁/符号：FontCLIP 矢量域直接优化，不走扩散，避免扩散拉丁重建崩坏
    for i, ch in enumerate(vec_chars, 1):
        yield None, "矢量优化拉丁字符 '%s'（%d/%d，FontCLIP %d 步）..." % (
            ch, i, len(vec_chars), int(seed_iter)), None
        try:
            src_svg, loss = optimize_vector_char(ch, font_path, concept, seed_iter, run_dir)
            _, png = export_vector_result(src_svg, ch, svg_dir, preview_dir)
            result_map[ch] = (png, ch)
            ok += 1
        except Exception as e:
            yield None, "字符 '%s' 矢量优化失败：%s" % (ch, e), None

    # 中文等：种子锚点 -> FontDiffuser 扩散 -> 位图转矢量
    if diff_chars:
        if not seed_char:
            yield None, "缺少种子字符，无法生成中文字符", None
        else:
            yield None, "生成种子锚点（FontCLIP %d 步）..." % int(seed_iter), None
            anchor, loss = make_seed(seed_char, font_path, concept, seed_iter, 1.5, run_dir)
            yield None, "种子锚点完成，final_loss=%.4f；开始批量扩散..." % loss, None
            cfg = {"style_image_path": anchor, "content_font_path": font_path,
                   "chars": diff_chars, "out_dir": bitmap_dir,
                   "num_inference_steps": int(diff_steps)}
            manifest = None
            for ev in run_batch(cfg):
                if ev.get("done"):
                    manifest = ev.get("manifest")
                yield None, "扩散生成中：" + ev["status"], None
            if manifest is not None:
                yield None, "位图转矢量（可微拟合）...", None
                records = batch_bitmap_to_svg(bitmap_dir, svg_dir, manifest, refine_steps=200)
                for r in records:
                    if r.get("svg"):
                        png = os.path.join(bitmap_dir, r["file"])
                        result_map[r["char"]] = (png, r["char"])
                        ok += 1

    # gallery 按用户输入的原始字符顺序展示
    ordered = [ch for ch in dict.fromkeys(chars) if not ch.isspace()]
    gallery = [result_map[ch] for ch in ordered if ch in result_map]
    yield gallery, "完成：%d/%d 字符已转为可编辑 SVG（拉丁走矢量、中文走扩散）。" % (
        ok, total), svg_dir


# ---------------- Tab 2：参考图 one-shot ----------------
def pipeline_ref(ref_image, font_path, chars, diff_steps):
    if ref_image is None or not chars.strip():
        yield None, "缺少参考图或目标字符集", None
        return
    run_dir = new_run_dir("ref")
    bitmap_dir = os.path.join(run_dir, "bitmap")
    svg_dir = os.path.join(run_dir, "svg")

    cfg = {"style_image_path": ref_image, "content_font_path": font_path.strip(),
           "chars": list(chars.strip()), "out_dir": bitmap_dir,
           "num_inference_steps": int(diff_steps)}
    manifest = None
    for ev in run_batch(cfg):
        if ev.get("done"):
            manifest = ev.get("manifest")
        yield None, "扩散生成中：" + ev["status"], None

    if manifest is None:
        yield None, "扩散生成失败", None
        return
    yield None, "位图转矢量（可微拟合）...", None
    records = batch_bitmap_to_svg(bitmap_dir, svg_dir, manifest, refine_steps=200)
    gallery = [(os.path.join(bitmap_dir, r["file"]), r["char"])
               for r in records if r.get("file")]
    ok = sum(1 for r in records if r.get("svg"))
    yield gallery, "完成：%d/%d 字符已转为可编辑 SVG。" % (ok, len(records)), svg_dir


# ---------------- Tab 3：画笔局部精修 ----------------
BRUSH_STATE = {"shapes": None, "groups": None}


def brush_render(char, font_path):
    char, font_path = char.strip(), font_path.strip()
    from fontTools.ttLib import TTFont
    try:
        cmap = TTFont(font_path).getBestCmap()
    except Exception as e:
        return None, "无法读取字体：%s" % e
    if ord(char) not in cmap:
        return None, "字体 %s 不含字符“%s”，请换一个含该字符的中文字体。" % (
            os.path.basename(font_path), char)
    tmp_svg = os.path.join("output", "unified", "_brush_tmp.svg")
    os.makedirs(os.path.dirname(tmp_svg), exist_ok=True)
    Config.create_svg_from_font(font_path, char, output_path=tmp_svg,
                                char_size=CHAR_SIZE, size=SIZE)
    w, h, shapes, groups = pydiffvg.svg_to_scene(tmp_svg)
    BRUSH_STATE["shapes"], BRUSH_STATE["groups"] = shapes, groups
    pil = render_shapes_pil(shapes, groups, SIZE)
    return np.array(pil), "已渲染 '%s'，在需要修改的笔画上画白色画笔。" % char


def extract_mask(brush_input):
    if brush_input is None:
        return None
    mask = brush_input.get("mask") if isinstance(brush_input, dict) else brush_input
    if mask is None:
        return None
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 3] if mask.shape[2] == 4 else mask[..., 0]
    if mask.shape[0] != SIZE or mask.shape[1] != SIZE:
        mask = np.array(Image.fromarray(mask).resize((SIZE, SIZE), Image.NEAREST))
    return mask > 127


def brush_apply(brush_input, mode, amount):
    shapes = BRUSH_STATE["shapes"]
    if shapes is None:
        return None, "请先渲染字符。"
    mask = extract_mask(brush_input)
    if mask is None or int(mask.sum()) == 0:
        return None, "请先用画笔圈选要修改的区域。"
    m = "thin" if mode == "变细 (thin)" else "thick"
    new_shapes = local_thin(shapes, mask, float(amount), mode=m, size=SIZE)
    groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.arange(len(new_shapes), dtype=torch.int32),
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        use_even_odd_rule=torch.tensor(True))]
    pil = render_shapes_pil(new_shapes, groups, SIZE)
    run_dir = new_run_dir("brush")
    out_svg = os.path.join(run_dir, "brush_result.svg")
    save_svg_full(out_svg, SIZE, SIZE, new_shapes, groups)
    return np.array(pil), "局部%s完成（amount=%s），已导出：%s" % (mode, amount, out_svg)


# ---------------- UI ----------------
with gr.Blocks(title="统一字体系统") as demo:
    gr.Markdown("# 统一字体系统\n文本/属性驱动、参考图 one-shot、画笔局部精修，输出可编辑可解释矢量。")

    with gr.Tab("文本/属性驱动字库"):
        with gr.Row():
            t_font = gr.Dropdown(label="源字体", choices=_font_choices,
                                 value=DEFAULT_FONT, scale=2)
            t_seed = gr.Textbox(label="种子字符", value="永", scale=1)
        t_concept = gr.Textbox(label="语义描述", value="italic, slanted font")
        t_chars = gr.Textbox(label="目标字符集", value="人永鑫")
        with gr.Row():
            t_seed_iter = gr.Number(label="种子优化步数", value=40)
            t_diff_steps = gr.Number(label="扩散步数", value=20)
        t_btn = gr.Button("生成字库", variant="primary")
        t_status = gr.Textbox(label="状态")
        t_gallery = gr.Gallery(label="生成结果").style(grid=4)
        t_svgdir = gr.Textbox(label="SVG 目录")
        t_btn.click(fn=pipeline_text,
                    inputs=[t_font, t_seed, t_concept, t_chars, t_seed_iter, t_diff_steps],
                    outputs=[t_gallery, t_status, t_svgdir])

    with gr.Tab("参考图 one-shot"):
        with gr.Row():
            r_ref = gr.Image(label="风格参考图", type="filepath")
            r_font = gr.Dropdown(label="内容字体（提供字符集）", choices=_font_choices,
                                 value=DEFAULT_FONT)
        r_chars = gr.Textbox(label="目标字符集", value="人永鑫")
        r_steps = gr.Number(label="扩散步数", value=20)
        r_btn = gr.Button("生成字库", variant="primary")
        r_status = gr.Textbox(label="状态")
        r_gallery = gr.Gallery(label="生成结果").style(grid=4)
        r_svgdir = gr.Textbox(label="SVG 目录")
        r_btn.click(fn=pipeline_ref, inputs=[r_ref, r_font, r_chars, r_steps],
                    outputs=[r_gallery, r_status, r_svgdir])

    with gr.Tab("画笔局部精修"):
        with gr.Row():
            b_char = gr.Textbox(label="字符", value="鑫", scale=1)
            b_font = gr.Dropdown(label="字体", choices=_font_choices,
                                 value=DEFAULT_FONT, scale=2)
        b_render_btn = gr.Button("1. 渲染字符", variant="primary")
        b_canvas = gr.Image(label="画白色画笔选择要修改的区域", source="upload",
                            tool="sketch", type="numpy", brush_radius=8,
                            brush_color="#ffffff", mask_opacity=0.7,
                            height=SIZE, width=SIZE)
        with gr.Row():
            b_mode = gr.Radio(label="修改方式", choices=["变细 (thin)", "变粗 (thick)"],
                              value="变细 (thin)")
            b_amount = gr.Slider(0.5, 6.0, value=2.0, step=0.5, label="强度 amount")
        b_apply_btn = gr.Button("2. 局部精修", variant="primary")
        b_status = gr.Textbox(label="状态")
        b_result = gr.Image(label="结果", type="numpy")
        b_render_btn.click(fn=brush_render, inputs=[b_char, b_font],
                           outputs=[b_canvas, b_status])
        b_apply_btn.click(fn=brush_apply, inputs=[b_canvas, b_mode, b_amount],
                          outputs=[b_result, b_status])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, debug=False)

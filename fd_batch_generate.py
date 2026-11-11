# -*- coding: utf-8 -*-
"""FontDiffuser 批量字库生成（fontdiffuser 环境，纯 CPU）。

加载一次模型，对目标字符集逐个扩散生成 96x96 位图。由统一界面经
env_bridge 以 subprocess 调用，参数通过 JSON 文件传入。

用法：
    python fd_batch_generate.py <config_json>

JSON 字段：
    style_image_path    风格图（参考图或种子锚点，白底黑字）
    content_font_path   内容字体 TTF（用于渲染各目标字符）
    chars               目标字符列表
    out_dir             输出目录
    num_inference_steps 推理步数（默认 20）
    guidance_scale      CFG 强度（默认 7.5）
    seed                随机种子（默认 123）
"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(ROOT, "external", "FontDiffuser")
PYLIBS = os.path.join(ROOT, ".pylibs")
sys.path.insert(0, PYLIBS)
sys.path.insert(0, FD)

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from configs.fontdiffuser import get_parser
from src import (FontDiffuserDPMPipeline, FontDiffuserModelDPM, build_ddpm_scheduler,
                 build_unet, build_content_encoder, build_style_encoder)
from utils import ttf2im, load_ttf, is_char_in_font


def emit(line):
    print(line, flush=True)


def main():
    assert len(sys.argv) >= 2, "missing config json"
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfgj = json.load(f)

    style_path = cfgj["style_image_path"]
    content_font = cfgj["content_font_path"]
    chars = list(cfgj["chars"])
    out_dir = cfgj["out_dir"]
    steps = int(cfgj.get("num_inference_steps", 20))
    guidance = float(cfgj.get("guidance_scale", 7.5))
    seed = int(cfgj.get("seed", 123))
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    parser = get_parser()
    args = parser.parse_args([])
    args.ckpt_dir = os.path.join(FD, "ckpt")
    args.device = "cpu"
    args.num_inference_steps = steps
    args.guidance_scale = guidance
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)

    t0 = time.time()
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "unet.pth"), map_location="cpu"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "style_encoder.pth"), map_location="cpu"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "content_encoder.pth"), map_location="cpu"))
    model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)
    model.to("cpu")
    emit("MODEL_LOADED %.1f" % (time.time() - t0))

    scheduler = build_ddpm_scheduler(args=args)
    pipe = FontDiffuserDPMPipeline(model=model, ddpm_train_scheduler=scheduler,
                                 model_type=args.model_type, guidance_type=args.guidance_type,
                                 guidance_scale=args.guidance_scale)

    tf = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    style_pil = __import__("PIL").Image.open(style_path).convert("RGB")
    style_img = tf(style_pil)[None]

    pygame_font = load_ttf(content_font)
    manifest = {"chars": [], "out_dir": out_dir}
    done = 0
    for idx, ch in enumerate(chars):
        rec = {"char": ch, "codepoint": "u%x" % ord(ch)}
        try:
            if not is_char_in_font(content_font, ch):
                rec.update(status="not_in_font", file=None)
                emit("PROGRESS %d/%d %s NOT_IN_FONT" % (idx + 1, len(chars), ch))
            else:
                content_pil = ttf2im(pygame_font, ch)
                content_img = tf(content_pil)[None]
                with torch.no_grad():
                    images = pipe.generate(
                        content_images=content_img, style_images=style_img, batch_size=1,
                        order=args.order, num_inference_step=args.num_inference_steps,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                        t_start=args.t_start, t_end=args.t_end, dm_size=args.content_image_size,
                        algorithm_type=args.algorithm_type, skip_type=args.skip_type,
                        method=args.method, correcting_x0_fn=args.correcting_x0_fn)
                fname = "u%x.png" % ord(ch)
                images[0].save(os.path.join(out_dir, fname))
                rec.update(status="ok", file=fname)
                done += 1
                emit("PROGRESS %d/%d %s OK %s" % (idx + 1, len(chars), ch, fname))
        except Exception as e:
            rec.update(status="error", file=None, error=str(e))
            emit("PROGRESS %d/%d %s ERROR %s" % (idx + 1, len(chars), ch, e))
        manifest["chars"].append(rec)

    manifest["ok"] = done
    manifest["total_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    emit("BATCH_DONE ok=%d/%d total_sec=%.1f" % (done, len(chars), time.time() - t0))


if __name__ == "__main__":
    main()

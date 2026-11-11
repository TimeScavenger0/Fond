"""诊断模块 C 各损失的 NaN 来源：受控小 batch，逐项打印数值与 isnan。"""
import os

import torch
from PIL import Image

from models.init_model import device, load_model
from models.lora import LoRAConfig
from models.aux_modules import (
    ContentStyleProjector,
    style_contrast_loss,
    subspace_orth_loss,
    reconstruction_loss,
)

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "data_cn", "mixed", "mixed_images")
CKPT = os.path.join(BASE, "model_checkpoints", "fontclip_lora_best.pt")


def main():
    lora = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    model = load_model(
        CKPT, model_name="ViT-B/32", use_oft_vision=False, use_oft_text=False,
        use_lora_text=True, use_lora_vision=False, lora_config_text=lora,
        use_coop_text=False, use_coop_vision=False, precontext_dropout_rate=0,
        pt_applied_layers=None,
    )
    model.float().eval()
    from models.init_model import preprocess

    files = sorted(os.listdir(IMG))[:3]
    paths = [os.path.join(IMG, files[0]), os.path.join(IMG, files[0]),
             os.path.join(IMG, files[1]), os.path.join(IMG, files[2])]
    x = torch.cat([preprocess(Image.open(p)).unsqueeze(0) for p in paths], dim=0)
    font_idx = torch.tensor([0, 0, 1, 2])

    with torch.no_grad():
        feat = model.encode_image(x)
    print("feat isnan:", torch.isnan(feat).any().item(), "norm:",
          feat.norm(dim=1).tolist())
    fn = feat / feat.norm(dim=1, keepdim=True)
    print("feat_norm isnan:", torch.isnan(fn).any().item())

    proj = ContentStyleProjector(512)
    c, s = proj.split(fn)
    print("content isnan:", torch.isnan(c).any().item(),
          "style isnan:", torch.isnan(s).any().item())

    import torch.nn.functional as F
    n = s.shape[0]
    sn = F.normalize(s, dim=1)
    print("normalized style norm:", sn.norm(dim=1).tolist(),
          "isnan:", torch.isnan(sn).any().item())
    logits = (sn @ sn.t()) / 0.1
    same_font = font_idx.view(-1, 1) == font_idx.view(1, -1)
    self_mask = torch.eye(n, dtype=torch.bool)
    pos_mask = same_font & ~self_mask
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    print("logits:\n", logits)
    print("pos_mask:\n", pos_mask)
    print("n_pos:", n_pos.tolist(), "valid:", valid.tolist())
    logits2 = logits.masked_fill(self_mask, float("-inf"))
    log_prob = F.log_softmax(logits2, dim=1)
    print("log_prob:\n", log_prob)
    summed = (log_prob * pos_mask).sum(dim=1)
    print("summed:", summed.tolist())
    m = summed[valid] / n_pos[valid]
    print("mean per valid:", m.tolist())
    l_style = -m.mean()
    l_orth = subspace_orth_loss(proj)
    l_recon = reconstruction_loss(proj, fn)
    print("style_contrast:", l_style.item(), "isnan:", torch.isnan(l_style).item())
    print("subspace_orth:", l_orth.item(), "isnan:", torch.isnan(l_orth).item())
    print("recon:", l_recon.item(), "isnan:", torch.isnan(l_recon).item())
    print("DIAG DONE")


if __name__ == "__main__":
    main()

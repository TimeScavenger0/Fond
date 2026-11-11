"""
S2/S3 可消融辅助模块（独立于核心 CLIP 模型，避免污染 strict 加载）。

模块 B：AttributeHeads，归一化图像特征上的逐属性线性判别头。
模块 C：ContentStyleProjector，内容/风格正交投影 + 风格对比 + 重构保持。
模块 D：内容分量锚定（结构稳定），在 trainer 中作为模块 C 的强化项。

所有模块参数少，适配小数据集；由 trainer 用独立优化器管理，可独立开关。
"""
import torch
from torch import nn
import torch.nn.functional as F


class AttributeHeads(nn.Module):
    """模块 B：每个活跃属性一个独立线性二分类头（共享输入、参数相互独立）。"""

    def __init__(self, feature_dim, n_attr):
        super().__init__()
        self.linear = nn.Linear(feature_dim, n_attr)

    def forward(self, image_features):
        return self.linear(image_features)


class ContentStyleProjector(nn.Module):
    """模块 C：把特征线性投影到内容、风格两个子空间。"""

    def __init__(self, feature_dim):
        super().__init__()
        self.content_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.style_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def split(self, feat):
        return self.content_proj(feat), self.style_proj(feat)


def style_contrast_loss(style_feat, font_indices, temperature=0.1):
    """同字体（风格恒定）的风格分量应聚拢，不同字体应分离（supervised InfoNCE）。

    batch 内没有同字体伙伴的样本被忽略；若整批都没有正样本对，返回 0 标量。
    """
    n = style_feat.shape[0]
    s = F.normalize(style_feat, dim=1)
    logits = (s @ s.t()) / temperature
    same_font = font_indices.view(-1, 1) == font_indices.view(1, -1)
    self_mask = torch.eye(n, dtype=torch.bool, device=style_feat.device)
    pos_mask = same_font & ~self_mask

    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return style_feat.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = F.log_softmax(logits, dim=1)
    # 用 where 把非正样本位置（含自身的 -inf）替换为 0，避免 -inf*0=NaN
    picked = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    mean_log_prob_pos = picked.sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


def subspace_orth_loss(projector):
    """模块 C：内容与风格投影方向正交，||Wc Ws^T||_F^2 / d，按输出维度归一化。"""
    wc = projector.content_proj.weight
    ws = projector.style_proj.weight
    d = wc.shape[0]
    return (wc @ ws.t()).pow(2).sum() / d


def reconstruction_loss(projector, feat):
    """信息保持：内容 + 风格分量应能还原原特征，避免拆分丢失信息。"""
    c, s = projector.split(feat)
    return (c + s - feat).pow(2).sum(dim=1).mean()


def content_anchor_loss(content_feat, anchor_content):
    """模块 D：内容分量相对标准锚点的内容投影稳定，风格更新不带动内容漂移。"""
    return (content_feat - anchor_content).pow(2).sum(dim=1).mean()

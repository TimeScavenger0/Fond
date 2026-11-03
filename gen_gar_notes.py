# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
for section in doc.sections:
    section.page_width = Cm(21); section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5); section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5); section.right_margin = Cm(2.5)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"; style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.first_line_indent = Cm(0.74)

def title(t):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(t); r.bold = True; r.font.size = Pt(18)
    r.font.name = "Times New Roman"; r.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    p.paragraph_format.space_after = Pt(18); p.paragraph_format.first_line_indent = Cm(0)

def h1(t):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = True; r.font.size = Pt(15)
    r.font.name = "Times New Roman"; r.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(0)

def body(t):
    doc.add_paragraph(t)

title("GAR-Font 论文可借鉴要点")

body("本文档梳理 GAR-Font（Beyond Patches: Global-aware Autoregressive Model for Multimodal Few-Shot Font Generation, 北大王选所, 2026）中可迁移到 FontCLIP 画笔局部优化工作的技术思路，作为后续实验设计和论文撰写的参考。")

h1("一、文本输入方式：language-style adapter")

body("GAR-Font 没有让用户自由组织自然语言描述，而是在视觉模态预训练完成后，加一个轻量 language-style adapter，把 Flan-T5 的文本嵌入投影到视觉风格空间，用对比损失对齐文本和视觉特征。这个思路对应当前自由文本框的改进方向：与其让用户自己写风格描述，不如在后端加一个投影层，把用户输入映射到 FontCLIP 微调后 latent 空间里有意义的方向上。当前系统直接把用户输入喂给 CLIP 文本编码器，微调后的 latent 空间对自由文本的响应可能不够精准，加一个轻量 adapter 层是可实验的方向。")

h1("二、两阶段后处理：NFA 加结构精修")

body("GAR-Font 的 post-refinement 分两步：先在少量参考字上微调 LoRA 适应新字体风格（NFA），再用 GRPO 强化学习优化字形结构正确性（SE），奖励来自分类器对字形结构的判别。当前画笔优化是一次性跑 100 步出结果，可以拆成两阶段：第一步用较大学习率快速把笔画拉向目标风格方向，第二步用较小学习率做结构精修，防止过冲和笔画崩坏。特别是字形识别损失的思路——在优化 loss 里加一个字符分类损失，确保变形后的字符仍然可辨认，而不是笔画缠成一团。")

h1("三、Soft decoding 思想")

body("GAR-Font 对比了 hard decoding（直接选最近 codebook 条目）和 soft decoding（按概率加权采样），实验证明 soft decoding 在 RMSE、SSIM、LPIPS、FID 所有指标上都更优，配合 pixel-level 监督时效果最明显。这个不要硬选、要加权融合的思想，对应矢量优化中用平滑插值代替硬转折的设计——当前已经把直线段路径改成二次贝塞尔曲线，本质上就是用平滑插值代替硬转折，可以把 soft 作为自觉的设计原则贯穿后续改进。")

h1("四、全局感知与局部控制互补")

body("GAR-Font 的核心贡献 G-Tok 用 hybrid CNN-ViT 架构，CNN 提取局部笔画细节，ViT 自注意力捕捉整字全局结构。画笔局部优化是精细到单笔画的局部操作，两者天然互补：全局正则保证局部变形不破坏整字骨架，局部控制实现精细风格调整。当前优化 loss 主要是 FontCLIP 余弦距离，可以借鉴全局上下文思想，加一个全局结构正则项，让圈选变形区域和圈外冻结区域之间保持平滑过渡，避免边界处出现突兀的形状断裂。")

h1("五、评估指标借鉴")

body("GAR-Font 除了 RMSE、SSIM、LPIPS 这些像素级指标，还用了字形分类准确率（Acc）衡量结构正确性，用 FID 衡量生成分布真实性。做对比实验时可以直接借用这些指标，特别是 Acc——变形后的字符是否仍被正确分类为目标字母，直接衡量风格变形但字符身份不丢失这个关键性质。")

doc.save(r"D:\VScode\VScodeProjects\Fond\GAR-Font可借鉴要点.docx")
print("done")

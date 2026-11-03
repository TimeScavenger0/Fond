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

title("VecFontLLM 论文可借鉴要点")

body("本文档梳理 VecFontLLM（Anchor-Guided Direct Synthesis of Chinese Vector Fonts, ACM TOG, 2026）中可迁移到 FontCLIP 画笔局部优化工作的技术思路。该论文用 StarCoder-1B 作为基座，通过锚点引导的三阶段分解直接生成中文矢量字体 SVG path。")

h1("一、三阶段分解：骨架到精修到贝塞尔补全")

body("VecFontLLM 把矢量字形生成拆成三步：先预测锚点骨架（路径分段端点和角点），再回填修正锚点结构，最后补全贝塞尔控制点。当前画笔优化是一次性同时优化所有控制点，可以拆成两步：先用较大学习率优化路径分段的端点和角点，快速确定笔画走向，再用较小学习率微调贝塞尔控制点调整曲线弧度和粗细。粗调阶段不会因为控制点乱跑导致路径自相交，精修阶段只在骨架稳定后做形状微调，崩坏概率大幅降低。")

h1("二、Confidence-guided test-time scaling")

body("VecFontLLM 在每个阶段采样 20 个候选，按置信度选最优的进入下一阶段。当前优化一次出结果，可以借鉴这个策略：用不同随机种子或不同学习率跑 3 到 5 次，选 FontCLIP 余弦距离最小的结果作为最终输出。CPU 上一次 100 步约 70 秒，跑 3 次约 3 分半钟，但能显著减少偶发的优化失败。")

h1("三、锚点与控制点分离优化")

body("VecFontLLM 把路径分段的端点和角点与曲线控制点分开预测。做矢量优化时可以区分两类参数：端点位移决定笔画走向，控制点位移决定曲线形状。分别设置不同学习率，端点用大学习率快速移动，控制点用小学习率微调弧度，避免控制点过冲导致曲线变形或路径自相交。")

h1("四、逐组件优化与选最优再拼接")

body("VecFontLLM 做复杂汉字时逐部件生成，先做一个部件选最优再拼下一个。画笔模式本身就是逐笔画优化，但可以借鉴它的选最优再拼接逻辑：优化完一个笔画后，先评估这个笔画的变形质量（FontCLIP 距离是否收敛、路径是否自相交），通过了再继续圈下一个笔画，不通过就重置重跑。")

h1("五、轻量少样本微调适配新风格")

body("VecFontLLM 只用 75 个参考字做 SFT 就能快速适配新字体风格。这个少样本微调思路对中文书法方向有参考，不需要收集大量书法字体做全量训练，用几十个书法字做 LoRA 微调就能把书法方向注入 latent 空间。")

doc.save(r"D:\VScode\VScodeProjects\Fond\VecFontLLM可借鉴要点.docx")
print("done")

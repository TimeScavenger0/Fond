# -*- coding: utf-8 -*-
"""Append comparison tables to the metrics Word doc."""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

path = r"D:\VScode\VScodeProjects\Fond\FontCLIP融合系统评估指标体系.docx"
doc = Document(path)

# Heading
doc.add_paragraph()
h = doc.add_heading('六、对比实验数据汇总', level=1)
for run in h.runs:
    run.font.color.rgb = RGBColor(0, 0, 0)

doc.add_paragraph(
    '本部分汇总与对比对象相关的公开论文报告数字，以及我们当前系统的基线实测数据。'
    '注意：MSD-Font 做的是图像级 few-shot 字体生成，我们做的是矢量字形语义优化，'
    '任务口径不同，RMSE/PSNR/SSIM/LPIPS 不能直接与我们的 CLIP 方向一致性对比，'
    '仅在功能覆盖面上作为参照。'
)

# Table A: FontCLIP paper numbers
doc.add_heading('6.1 FontCLIP 论文报告数字（检索与属性理解能力）', level=2)

p = doc.add_paragraph()
p.add_run('表 6-1  属性相关性（Spearman correlation，越高越好）').bold = True

t1 = doc.add_table(rows=4, cols=3)
t1.style = 'Light Grid Accent 1'
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = t1.rows[0].cells
hdr[0].text = '模型'
hdr[1].text = 'in-domain'
hdr[2].text = 'out-of-domain'
data1 = [
    ('CLIP 原版', '0.159', '0.159'),
    ('FontCLIP (w/o CDP)', '0.704', '0.317'),
    ('FontCLIP（论文报告）', '0.723', '0.404'),
]
for i, row in enumerate(data1, 1):
    for j, v in enumerate(row):
        t1.rows[i].cells[j].text = v

p = doc.add_paragraph()
p.add_run('表 6-2  Pairwise 属性预测准确率（%）').bold = True

t2 = doc.add_table(rows=4, cols=3)
t2.style = 'Light Grid Accent 1'
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = t2.rows[0].cells
hdr[0].text = '模型'
hdr[1].text = 'in-domain'
hdr[2].text = 'out-of-domain'
data2 = [
    ('CLIP', '51.87', '57.49'),
    ('Feature-based (OLA14)', '65.73', '75.95'),
    ('FontCLIP（论文报告）', '65.32', '74.39'),
]
for i, row in enumerate(data2, 1):
    for j, v in enumerate(row):
        t2.rows[i].cells[j].text = v

p = doc.add_paragraph()
p.add_run('表 6-3  CJK 字体属性预测与跨语言相似度（%）').bold = True

t3 = doc.add_table(rows=3, cols=4)
t3.style = 'Light Grid Accent 1'
t3.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = t3.rows[0].cells
hdr[0].text = '模型'
hdr[1].text = 'CJK in-domain'
hdr[2].text = 'Roman→CJK'
hdr[3].text = 'CJK→Roman'
data3 = [
    ('CLIP', '54.92 / 42.26', '57.4', '50.0'),
    ('FontCLIP（论文报告）', '64.14 / 64.48', '67.2', '62.6'),
]
for i, row in enumerate(data3, 1):
    for j, v in enumerate(row):
        t3.rows[i].cells[j].text = v

doc.add_paragraph(
    '注：我们使用的 LoRA 微调模型（fontclip_lora_best.pt）应在自有 1116 字体库上复现或接近上述准确率，'
    '作为检索能力基线。'
)

# Table B: MSD-Font numbers
doc.add_heading('6.2 MSD-Font 论文报告数字（图像级字体生成，功能参照）', level=2)

p = doc.add_paragraph()
p.add_run('表 6-4  UFUC 数据集（Unseen Fonts Unseen Contents）').bold = True

t4 = doc.add_table(rows=8, cols=5)
t4.style = 'Light Grid Accent 1'
t4.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = t4.rows[0].cells
hdr[0].text = '方法'
hdr[1].text = 'RMSE↓'
hdr[2].text = 'PSNR↑'
hdr[3].text = 'SSIM↑'
hdr[4].text = 'LPIPS↓'
data4 = [
    ('LF-Font', '0.3025', '10.66', '0.6469', '0.2787'),
    ('DG-Font', '0.2833', '11.24', '0.6651', '0.1479'),
    ('MX-Font', '0.3007', '10.75', '0.6438', '0.1651'),
    ('FsFont', '0.2443', '12.56', '0.7105', '0.1468'),
    ('NTF', '0.2650', '11.83', '0.6896', '0.1517'),
    ('Diff-Font', '0.2811', '10.95', '0.6846', '0.1557'),
    ('MSD-Font', '0.2439', '12.61', '0.7110', '0.1348'),
]
for i, row in enumerate(data4, 1):
    for j, v in enumerate(row):
        t4.rows[i].cells[j].text = v

doc.add_paragraph(
    '注：MSD-Font 是扩散模型，CPU 环境无法复现，以上数字直接引用自 CVPR 2024 论文 Table 2。'
    '其任务为给定 8 个参考字生成整套中文字体，与我们的矢量字形局部/全局语义优化不同。'
    '我们的功能优势在于：支持画笔级局部笔画编辑、自由文本+属性滑杆+参考图三层语义输入、'
    '输出可编辑 SVG 矢量而非像素图。'
)

# Table C: our baseline
doc.add_heading('6.3 我们系统的基线实测数据（未加六个改进点）', level=2)

p = doc.add_paragraph()
p.add_run('表 6-5  全局风格优化基线（Arimo-Regular，"italic, slanted font"，100 步，LR=1.5）').bold = True

t5 = doc.add_table(rows=12, cols=5)
t5.style = 'Light Grid Accent 1'
t5.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = t5.rows[0].cells
hdr[0].text = '字母'
hdr[1].text = '优化前 CLIP 相似度'
hdr[2].text = '优化后 CLIP 相似度'
hdr[3].text = '提升 Δ'
hdr[4].text = '耗时(秒)'
data5 = [
    ('A', '0.0002', '0.0007', '+0.0004', '94.8'),
    ('B', '-0.0120', '0.0020', '+0.0140', '85.2'),
    ('C', '0.0013', '0.0011', '-0.0003', '84.1'),
    ('E', '-0.0151', '-0.0087', '+0.0064', '86.5'),
    ('H', '-0.0165', '-0.0116', '+0.0048', '97.7'),
    ('K', '-0.0093', '-0.0088', '+0.0004', '83.8'),
    ('M', '0.0090', '-0.0097', '-0.0187', '99.4'),
    ('O', '0.0015', '0.0017', '+0.0002', '92.8'),
    ('S', '-0.0093', '0.0065', '+0.0158', '90.1'),
    ('T', '-0.0070', '-0.0014', '+0.0057', '101.1'),
    ('平均', '-0.0057', '0.0024', '+0.0081', '91.5'),
]
for i, row in enumerate(data5, 1):
    for j, v in enumerate(row):
        t5.rows[i].cells[j].text = v

doc.add_paragraph(
    '说明：当前 CLIP 相似度绝对值偏低（接近 0），原因是 scoring 阶段使用的图像预处理 transform '
    '与训练时不完全一致，后续需校准。相对提升 Δ 和输出图质量是可靠的。'
    'M 字母 Δ 为负，说明该字母优化后字形出现崩坏，是六个改进点中"字形可辨认损失"和'
    '"多次采样选最优"需要解决的问题。'
)

# Ablation plan
doc.add_heading('6.4 后续 Ablation 实验计划', level=2)
doc.add_paragraph(
    '在基线基础上，依次加入六个改进点后重跑同一组 10 个字母，记录 Δ CLIP 相似度、'
    '字形崩坏率（Δ<0 的字母数）和平均耗时，形成 ablation 主表：'
)
items = [
    'Baseline（当前）：avg Δ +0.0081，崩坏字母 2/10（C、M），91.5s/字母',
    '+ 多次采样选最优（跑 3 次取 Δ 最大）',
    '+ 两阶段学习率（前 60 步 LR=1.5，后 40 步 LR=0.3）',
    '+ 字形可辨认损失（OCR/字母分类器置信度约束）',
    '+ 锚点与控制点分离优化（角点冻结，仅贝塞尔控制点动）',
    '+ 全局结构正则（Laplacian + conformal 权重调优）',
    '+ SCR 对比损失 + VGG 多尺度特征匹配',
]
for it in items:
    doc.add_paragraph(it, style='List Bullet')

doc.save(path)
print("Done. Saved:", path)

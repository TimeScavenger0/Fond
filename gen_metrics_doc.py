# -*- coding: utf-8 -*-
"""Generate evaluation metrics Word document."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

# Page setup
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# Default style
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def set_cn_font(run, name_cn='宋体', name_en='Times New Roman', size=12, bold=False):
    run.font.name = name_en
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element
    r.rPr.rFonts.set(qn('w:eastAsia'), name_cn)


def add_heading_cn(text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    if level == 0:
        set_cn_font(run, '黑体', 'Times New Roman', 18, True)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(18)
    elif level == 1:
        set_cn_font(run, '黑体', 'Times New Roman', 16, True)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
    elif level == 2:
        set_cn_font(run, '黑体', 'Times New Roman', 14, True)
        p.paragraph_format.space_before = Pt(11)
        p.paragraph_format.space_after = Pt(5)
    return p


def add_body(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_cn_font(run, '宋体', 'Times New Roman', 12)
    p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(0)
    return p


# ===== Title =====
add_heading_cn('FontCLIP 融合系统评估指标体系说明', level=0)

# ===== Section 1 =====
add_heading_cn('一、方法定位与对比原则', level=1)
add_body('本系统定位为统一的语义驱动矢量字形编辑器，融合三类能力：自由文本描述整体风格方向、属性滑杆连续精调风格强度、画笔圈选笔画做局部变形，并支持参考图风格迁移。在与已有方法对比时，只选取功能重叠的部分在统一测试设置下比较，不跨任务强行对齐指标。具体而言，与 MSD-Font 等扩散式字体生成方法只比全局风格迁移质量；与原始 FontCLIP 比语义对齐能力；局部笔画控制为本文独有贡献，单独报告新指标。')

# ===== Section 2 =====
add_heading_cn('二、全局风格迁移质量指标', level=1)
add_body('这一组指标衡量系统把一个初始字形（指定字母和字体）优化到目标风格描述的能力，与 MSD-Font、FontDiffuser 等方法在"给定源字形、目标风格，输出风格化字形"这一共同任务上对比。所有渲染图统一在 200×200 分辨率、白底黑字下计算。')

table1 = doc.add_table(rows=1, cols=5)
table1.style = 'Table Grid'
table1.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table1.rows[0].cells
headers1 = ['指标名称', '方向', '中文解释', '计算方式', '主要对比对象']
for i, h in enumerate(headers1):
    hdr[i].text = ''
    p = hdr[i].paragraphs[0]
    run = p.add_run(h)
    set_cn_font(run, '黑体', 'Times New Roman', 10.5, True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

rows1 = [
    ['RMSE', '越低越好', '均方根误差，衡量生成图与目标参考图之间像素值的平均偏差。',
     '逐像素计算生成图与目标图灰度值差的平方，取均值后开平方。',
     'MSD-Font、FontDiffuser'],
    ['PSNR', '越高越好', '峰值信噪比，基于 RMSE 换算的图像质量度量，值越大说明失真越小。',
     '20·log10(250/RMSE)，单位分贝(dB)。',
     'MSD-Font、FontDiffuser'],
    ['SSIM', '越高越好', '结构相似性，从亮度、对比度、结构三方面衡量两张图的相似程度，比 PSNR 更符合人眼感知。',
     '在 11×11 滑动窗口内计算亮度、对比度、结构三项相似度的加权乘积。',
     'MSD-Font、FontDiffuser'],
    ['LPIPS', '越低越好', '学习感知图像块相似度，用预训练深度网络(VGG/AlexNet)提取多层特征计算距离，衡量人眼感知差异。',
     '对生成图和目标图分别过 VGG 网络，取多层特征图的 L2 距离加权求和。',
     'MSD-Font、FontDiffuser'],
    ['FID', '越低越好', 'Fréchet 初始距离，衡量生成字形分布与目标风格字体库整体分布的差距，用于参考图风格迁移任务。',
     '用 InceptionV3 提取生成图与真实目标字体图的特征，计算两组特征分布的 Fréchet 距离。',
     'FontDiffuser'],
]
for row_data in rows1:
    cells = table1.add_row().cells
    for i, val in enumerate(row_data):
        cells[i].text = ''
        p = cells[i].paragraphs[0]
        run = p.add_run(val)
        set_cn_font(run, '宋体', 'Times New Roman', 10.5)

# ===== Section 3 =====
add_heading_cn('三、语义对齐指标', level=1)
add_body('这一组指标衡量系统输出与用户文本描述的语义一致程度，与原始 FontCLIP 及其前身 Attribute2Font 对比。')

table2 = doc.add_table(rows=1, cols=5)
table2.style = 'Table Grid'
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr2 = table2.rows[0].cells
for i, h in enumerate(headers1):
    hdr2[i].text = ''
    p = hdr2[i].paragraphs[0]
    run = p.add_run(h)
    set_cn_font(run, '黑体', 'Times New Roman', 10.5, True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

rows2 = [
    ['CLIP 方向一致性', '越高越好',
     '生成字形图像与目标文本描述在 CLIP 语义空间中的余弦相似度，是本系统优化的直接目标。',
     '用 FontCLIP 视觉编码器提取渲染图 embedding，用文本编码器提取目标描述 embedding，归一化后计算余弦值。',
     '原始 FontCLIP'],
    ['字体验证准确率', '越高越好',
     '用预训练字体分类器判断生成字形是否被归类到目标风格类别，衡量风格迁移是否成功。',
     '将生成字形输入在 1116 个字体库上训练的分类器，取 top-1 预测是否命中目标风格字体。',
     'FontCLIP、Attribute2Font'],
    ['属性分数相关系数', '越高越好',
     '对 37 个字体属性(serif/italic/bold 等)，系统预测的属性强度与人工标注评分之间的 Spearman 秩相关系数。',
     '取生成字形过 FontCLIP 属性预测头得到各属性分数，与人工评分做 Spearman ρ 相关计算。',
     'FontCLIP、Attribute2Font'],
    ['参考图视觉相似度', '越高越好',
     '参考图风格迁移模式下，生成字形与用户上传的参考风格图之间的 CLIP 视觉 embedding 余弦相似度。',
     '参考图与渲染图分别过 CLIP 视觉编码器，计算 embedding 余弦值。',
     'FontDiffuser'],
]
for row_data in rows2:
    cells = table2.add_row().cells
    for i, val in enumerate(row_data):
        cells[i].text = ''
        p = cells[i].paragraphs[0]
        run = p.add_run(val)
        set_cn_font(run, '宋体', 'Times New Roman', 10.5)

# ===== Section 4 =====
add_heading_cn('四、局部笔画控制指标（本文新贡献）', level=1)
add_body('这一组指标是画笔局部优化功能的专属度量，MSD-Font 和原始 FontCLIP 均无此能力，为本系统首创。')

table3 = doc.add_table(rows=1, cols=5)
table3.style = 'Table Grid'
table3.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr3 = table3.rows[0].cells
for i, h in enumerate(headers1):
    hdr3[i].text = ''
    p = hdr3[i].paragraphs[0]
    run = p.add_run(h)
    set_cn_font(run, '黑体', 'Times New Roman', 10.5, True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

rows3 = [
    ['圈外冻结保持度', '越低越好',
     '画笔圈选区域之外的所有控制点在优化过程中的平均位移量，衡量"只动我圈的地方、不圈的地方不动"的能力。',
     '对圈外每个控制点，计算优化前后坐标差的 L2 范数，取所有圈外点的平均值。',
     '本文独有'],
    ['圈内变形幅度', '适中为佳',
     '画笔圈选区域内部控制点的平均位移量，衡量局部变形是否真正发生。',
     '对圈内每个控制点，计算优化前后坐标差的 L2 范数，取平均值。',
     '本文独有'],
    ['局部方向一致性', '越高越好',
     '圈选区域变形后，该区域局部渲染与目标语义描述在 CLIP 空间的一致性，衡量局部变形是否朝期望方向。',
     '将圈选区域的局部图像裁剪后过 CLIP 视觉编码器，与目标文本 embedding 计算余弦值。',
     '本文独有'],
    ['字形可辨认度', '越高越好',
     '优化后的字形是否仍能被正确识别为原字母(A/B/C 等)，防止局部变形把笔画改得认不出来。',
     '用预训练的英文字母分类模型(26 类)对渲染图做 top-1 预测，统计是否命中原字母。',
     '本文独有'],
]
for row_data in rows3:
    cells = table3.add_row().cells
    for i, val in enumerate(row_data):
        cells[i].text = ''
        p = cells[i].paragraphs[0]
        run = p.add_run(val)
        set_cn_font(run, '宋体', 'Times New Roman', 10.5)

# ===== Section 5 =====
add_heading_cn('五、效率与鲁棒性指标', level=1)
add_body('这一组指标衡量系统的实际可用性，是相对扩散模型的优势所在。')

table4 = doc.add_table(rows=1, cols=5)
table4.style = 'Table Grid'
table4.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr4 = table4.rows[0].cells
for i, h in enumerate(headers1):
    hdr4[i].text = ''
    p = hdr4[i].paragraphs[0]
    run = p.add_run(h)
    set_cn_font(run, '黑体', 'Times New Roman', 10.5, True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

rows4 = [
    ['单字母优化时间', '越低越好',
     '在 CPU 环境下优化一个字母达到收敛所需的墙上时间(秒)。',
     '从点击 Start optimization 到输出最终图的实际计时。',
     'MSD-Font、FontDiffuser'],
    ['是否需要训练', '零样本为佳',
     '对比方法是否需要针对目标风格训练额外权重，本系统直接在预训练 CLIP 上优化矢量路径，无需训练。',
     '定性说明：本系统零样本，扩散模型需训练或微调。',
     'MSD-Font、FontDiffuser'],
    ['内存占用', '越低越好',
     '运行时峰值显存/内存占用。',
     '记录优化过程中的最大进程内存。',
     'MSD-Font、FontDiffuser'],
]
for row_data in rows4:
    cells = table4.add_row().cells
    for i, val in enumerate(row_data):
        cells[i].text = ''
        p = cells[i].paragraphs[0]
        run = p.add_run(val)
        set_cn_font(run, '宋体', 'Times New Roman', 10.5)

# ===== Section 6 =====
add_heading_cn('六、对比实验设计原则', level=1)
add_body('第一，统一测试集。选取 26 个大写英文字母为基础测试集，每个字母用同一基准字体(Arimo-Regular)渲染，在相同的 200×200 分辨率下计算所有像素级指标。')
add_body('第二，统一评估协议。CLIP 方向一致性使用本项目训练的 fontclip_lora_best.pt 权重；字体验证分类器使用 1116 字体库上训练的同一分类器；所有方法在相同的风格描述文本下输出结果。')
add_body('第三，公平对比。与 MSD-Font 只在"全局风格迁移"这一共同功能上比 RMSE/PSNR/SSIM/LPIPS；与 FontCLIP 只在"文本语义对齐"上比 CLIP 方向一致性和属性相关系数；局部控制指标为本系统独有，单列报告，不要求对比方法具备。')
add_body('第四，诚实报告。MSD-Font 为扩散模型，CPU 环境下若无法完整复现推理，论文中将引用其论文报告的数字并注明"原文报告，未在本环境复现"，不伪造对比数据。所有自跑指标保留输出图片、SVG 和 loss 曲线存档以便追溯。')

out = r'D:\VScode\VScodeProjects\Fond\FontCLIP融合系统评估指标体系.docx'
doc.save(out)
print(f'Saved: {out}')

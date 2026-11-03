# -*- coding: utf-8 -*-
"""Generate today's discussion notes as a Word document."""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"D:\VScode\VScodeProjects\Fond\FontCLIP画笔交互方案讨论记录.docx"

doc = Document()

# page setup
for section in doc.sections:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# base style
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.first_line_indent = Pt(24)


def set_run_font(run, size=12, bold=False, eastasia="宋体", latin="Times New Roman"):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), eastasia)


def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(text)
    set_run_font(r, size=18, bold=True, eastasia="黑体")


def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run_font(r, size=16, bold=True, eastasia="黑体")


def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_run_font(r, size=14, bold=True, eastasia="黑体")


def add_body(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, size=12)


def add_table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    # header
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p0 = cell.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.paragraph_format.first_line_indent = Pt(0)
        r = p0.add_run(h)
        set_run_font(r, size=10.5, bold=True, eastasia="黑体")
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "F2F2F2")
        tcPr = cell._tc.get_or_add_tcPr()
        vAl = tcPr.find(qn("w:vAlign"))
        if vAl is not None:
            vAl.addprevious(shd)
        else:
            tcPr.append(shd)
    # body
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p0 = cell.paragraphs[0]
            p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p0.paragraph_format.first_line_indent = Pt(0)
            r = p0.add_run(val)
            set_run_font(r, size=10.5)


# ===== content =====
add_title("FontCLIP 画笔交互局部微调方案讨论记录")

add_body(
    "本文记录 2026 年 10 月 15 日关于 FontCLIP 笔画级局部优化通用化方案的讨论要点、可行性验证结论与下一步开工计划。"
)

add_h1("1  当前进展")
add_h2("1.1  已完成并验证的功能")
add_body(
    "字体检索模块已训练成功，Gradio 检索界面在 http://127.0.0.1:7860 正常运行，font_db 1116 个字体特征已缓存。"
    "全局矢量优化已跑通：以 Arimo-Regular 字符 A 为例，目标语义为斜体圆润风格，50 次迭代后字形成功从标准无衬线体变形为右倾斜体加圆润笔画。"
    "笔画局部优化已实现梯度掩码机制：在 optimizer/optimize.py 中新增 compute_frozen_mask 函数，按 region 参数冻结指定笔画的控制点梯度。"
    "实测左斜杠 77 个控制点可动（最大位移 4.23），右斜杠 82 个点与字腔 54 个点完全冻结（最大位移 0.0000），验证了只动指定笔画的效果。"
)

add_h2("1.2  已有脚本清单")
add_table(
    ["脚本", "用途", "状态"],
    [
        ["font_retrieval.py", "字体检索 Gradio 界面", "已完成"],
        ["run_vector_opt.py", "全局矢量优化入口", "已完成"],
        ["run_stroke_opt.py", "region 方式笔画局部优化", "已完成"],
        ["verify_stroke.py", "冻结点位移逐坐标验证", "已完成"],
        ["probe_letters.py", "多字母 path 结构探查", "已完成"],
        ["make_comparison.py", "前后对比图生成", "已完成"],
    ],
)

add_h1("2  通用化设计讨论")
add_h2("2.1  难点分层")
add_body(
    "通用化的真正难点不在识别外轮廓和字腔，而在单条连通外轮廓内部如何切出有语义的笔画。"
    "第一层识别外轮廓与字腔相对容易：按各条 path 的包围盒面积排序，最大的判为外轮廓，再用质心点是否落在外轮廓多边形内部判断其余 path 是否为字腔。这步是确定性几何计算，能解决 B 多字腔、i 点杆不混淆的问题。"
    "第二层才是硬骨头：像 A、X、K、H 这类字母，笔画在外轮廓上是连在一起的，坐标中位线切分对 A 碰巧有效，但对 K、R、B 切不出有排版意义的笔画。这一步没有唯一标准答案，因为一条轮廓在几何上是连续的环。"
)

add_h2("2.2  三条候选路线")
add_body(
    "第一条是方位启发式规则，按控制点坐标分布和方向突变拐点把轮廓切段，再按坐标特征归类为竖笔、横笔、斜笔。对 I、H、T 结构规整的字母效果好，但对 A、K 斜笔连笔归类模糊，阈值难调。"
    "第二条是用户交互圈选：渲染字形图后让用户用画笔涂出想动的笔画，程序把涂到区域内的控制点设为可动。这条路绕开了自动切笔画的无解问题，对任意字母都成立。"
    "第三条是按 shape 粒度，对多部件字母天然成立（如 i 的点和杆是两条独立 path），但对单连通字母用不上。"
)

add_h2("2.3  选定方案")
add_body(
    "最终选定组合：第一层自动识别外轮廓与字腔（几何计算），第二层用户画笔圈选指定笔画。"
    "方位启发式作为 I、H、T、A 等对称字母的快捷选项保留。"
    "梯度掩码核心机制完全复用，变的只是如何算出可动点集合这一前置步骤。"
)

add_h1("3  可行性验证结果")
add_h2("3.1  Gradio 画笔接口")
add_body(
    "实测环境为 gradio 3.40.0。gr.Image 构造函数支持 source 取 upload、webcam、canvas，tool 取 editor、select、sketch、color-sketch。"
    "关键发现：当 source 为 upload 或 webcam 且 tool 为 sketch 时，回调函数会收到一个字典，键为 image 和 mask，其中 mask 是用户画笔绘制的独立二值图。"
    "brush_radius 控制画笔粗细，brush_color 指定画笔颜色，mask_opacity 控制遮罩透明度，参数齐全。"
)

add_h2("3.2  坐标映射")
add_body(
    "实测 diffvg 渲染画布为 200×200，控制点坐标本身就是该像素坐标系下的浮点值。"
    "将每个控制点 round(x)、round(y) 后叠加红点到渲染图上，红点准确压在轮廓线上，无缩放、无翻转、无偏移。"
    "模拟左半画布 mask（x<100）时，外轮廓 159 个控制点中圈出 82 个可动、字腔 54 个点中圈出 35 个，计数合理。"
    "结论：控制点坐标到像素 mask 的映射零误差，直接用整数取整查表即可。"
)

add_h2("3.3  已知小工程细节")
add_body(
    "Gradio 前端若缩放显示图片，返回 mask 时需确认分辨率与预渲染图一致，必要时 resize 对齐。"
    "字腔默认锁定做成界面勾选框，自动识别出的字腔 path 除非用户主动涂到否则全冻结。"
)

add_h1("4  不影响旧功能的接入设计")
add_body(
    "梯度掩码在 optimizer/optimize.py 中只有两个落点：第 710 至 719 行生成 frozen_point_mask，第 936 至 940 行对冻结点梯度清零。"
    "现有逻辑是 frozen_point_mask 初始为 None，仅当 cfg.optimize_region 存在时才走 region 规则。"
    "新增 brush 路径时，在该 if 后加一个独立分支：若 cfg 携带预先算好的 brush_frozen_mask 则直接使用，否则保持 None 或走原 region 逻辑。"
    "梯度清零段是通用的，只要 frozen_point_mask 不为 None 就执行，与 mask 来源无关。"
)

add_table(
    ["路径", "触发条件", "行为", "改动量"],
    [
        ["全局优化", "不设任何 mask", "所有控制点更新", "无改动"],
        ["region 笔画优化", "cfg.optimize_region", "按几何规则冻结", "无改动"],
        ["画笔交互优化", "cfg.brush_frozen_mask", "按用户 mask 冻结", "optimize.py 约加 3 行"],
    ],
)

add_body(
    "已有脚本 font_retrieval.py、run_vector_opt.py、run_stroke_opt.py、verify_stroke.py、probe_letters.py、make_comparison.py 一行都不改。"
    "新功能全部放在独立脚本 run_brush_opt.py 中。"
)

add_h1("5  下一步开工计划")
add_body(
    "新建 run_brush_opt.py，流程为：用户选择字符与字体 → 自动用 diffvg 渲染初始字形图 → 在 Gradio 界面中用画笔圈选要动的笔画 → 输入风格语义描述 → 调用 train() 执行优化 → 输出 SVG 与前后对比图。"
    "optimize.py 仅在第 710 行附近加一个 elif 分支接收 brush_frozen_mask，不修改任何已有逻辑。"
    "字腔自动识别按包围盒面积排序加质心包含判断实现，默认全冻结。"
)

doc.save(OUT)
print("saved:", OUT)

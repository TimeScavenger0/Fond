# -*- coding: utf-8 -*-
"""Generate the FontCLIP stroke-local optimization explanation as a .docx."""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = r"D:\VScode\VScodeProjects\Fond"
OUT = os.path.join(BASE, "FontCLIP笔画级局部优化改动说明.docx")


def set_run_font(run, cn="宋体", en="Times New Roman", size=12, bold=False, color=(0,0,0)):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), cn)


def style_base(doc):
    # Normal
    n = doc.styles['Normal']
    n.font.name = 'Times New Roman'
    n.font.size = Pt(12)
    n.font.color.rgb = RGBColor(0, 0, 0)
    rpr = n.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.append(rf)
    rf.set(qn('w:eastAsia'), '宋体')


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(text)
    set_run_font(r, cn="黑体", en="Times New Roman", size=18, bold=True)
    return p


def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    pf = p._p.get_or_add_pPr()
    ol = OxmlElement('w:outlineLvl'); ol.set(qn('w:val'), '0'); pf.append(ol)
    r = p.add_run(text)
    set_run_font(r, cn="黑体", en="Times New Roman", size=16, bold=True)
    return p


def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(5)
    pf = p._p.get_or_add_pPr()
    ol = OxmlElement('w:outlineLvl'); ol.set(qn('w:val'), '1'); pf.append(ol)
    r = p.add_run(text)
    set_run_font(r, cn="黑体", en="Times New Roman", size=14, bold=True)
    return p


def add_body(doc, text, indent=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if indent:
        # first-line indent of 2 chars (0.74 cm ~ 21pt for 12pt font)
        pf.first_line_indent = Pt(24)
    r = p.add_run(text)
    set_run_font(r, cn="宋体", en="Times New Roman", size=12)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.left_indent = Pt(24)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    r = p.add_run(text)
    set_run_font(r, cn="宋体", en="Times New Roman", size=12)
    return p


def setup_page(doc):
    for sec in doc.sections:
        sec.page_height = Cm(29.7)
        sec.page_width = Cm(21.0)
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)


def add_page_number(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    fld1 = OxmlElement('w:fldChar'); fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve'); instr.text = 'PAGE'
    fld2 = OxmlElement('w:fldChar'); fld2.set(qn('w:fldCharType'), 'end')
    r._element.append(fld1); r._element.append(instr); r._element.append(fld2)
    set_run_font(r, cn="宋体", en="Times New Roman", size=9)


def build_table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htext in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.line_spacing = 1.0
        r = cp.add_run(htext)
        set_run_font(r, cn="宋体", en="Times New Roman", size=10.5, bold=True)
        # header shading light gray (insert BEFORE vAlign to keep OOXML schema order)
        shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), 'F2F2F2')
        tcPr = cell._tc.get_or_add_tcPr()
        vAl = tcPr.find(qn('w:vAlign'))
        if vAl is not None:
            vAl.addprevious(shd)
        else:
            tcPr.append(shd)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cell = cells[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cp = cell.paragraphs[0]
            cp.paragraph_format.line_spacing = 1.0
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (1,) else WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(str(val))
            set_run_font(r, cn="宋体", en="Times New Roman", size=10.5)
    return t


def main():
    doc = Document()
    style_base(doc)
    setup_page(doc)
    add_page_number(doc)

    add_title(doc, "FontCLIP 笔画级局部优化改动说明")

    # 1
    add_h1(doc, "1  背景与目标")
    add_h2(doc, "1.1  FontCLIP 与矢量优化")
    add_body(doc, "FontCLIP 把一个能理解图像与语义的大型视觉语言模型（CLIP）微调成了懂字体排版的模型。微调后，它既能理解“粗体、斜体、衬线、圆润”这类语义描述，也能理解字形图片。基于它可实现两类应用：输入风格描述检索最匹配的字体（字体检索）；或给定一个初始字符与目标风格，让模型引导字形自身变形过去（矢量优化）。本项目已跑通字体检索训练，并在其上完成了矢量优化。")
    add_h2(doc, "1.2  本次改动目的")
    add_body(doc, "原矢量优化是整字形全局变形，即优化器同时更新一个字符的所有控制点。本项目的目标是把变形约束到指定的一条笔画上，实现“某一个笔画的局部形变”：让选中的笔画在语义驱动下改变，其余笔画（包括字腔）保持原位不动，以支持对字体局部形态的精细微调。")

    # 2
    add_h1(doc, "2  基础概念")
    add_h2(doc, "2.1  path（路径）")
    add_body(doc, "path 是 SVG 矢量图里的一条轮廓线。一个字符由一条或多条 path 组成。以大写 A 为例，它被拆成两条 path：一条是外轮廓（“A”的整体外形），另一条是中间三角形的字腔（counter）。在 diffvg 中，每条 path 对应一个独立的形状对象，每个形状拥有自己的控制点集合。")
    add_h2(doc, "2.2  控制点")
    add_body(doc, "控制点是定义贝塞尔曲线的点。每一段曲线由起点、终点和两个控制点共同决定，控制点像“磁铁”一样拉出曲线的弯曲。因此一条 path 的所有控制点坐标就完整决定了这条 path 的形状。只要改动控制点坐标，就能改变字形外观，这正是矢量优化能够“变形”的根本原因——字形在底层就是一堆可控的控制点坐标。")

    # 3
    add_h1(doc, "3  原方法的实现流程")
    add_body(doc, "整体是一条闭环迭代：先用字体工具把指定字符的字形轮廓提取为 SVG 的 path，得到初始控制点，并将其作为可训练参数交给 Adam 优化器。随后每一轮执行四步：第一，用可微渲染器 diffvg 把当前控制点实时渲染成位图；第二，将位图送入 FontCLIP，与目标语义描述计算语义潜空间的余弦距离，作为语义损失；第三，叠加保形损失与色调损失，防止字形变形后笔画交叉扭曲、粗细分布失调；第四，对总损失反向传播，得到每个控制点的梯度，用 Adam 更新控制点坐标。如此反复迭代，字形逐渐逼近目标风格，最终输出优化后的 SVG。")
    add_body(doc, "关键点在于：原方法每一轮都会更新所有字符的所有控制点，因此是全局变形。")

    # 4
    add_h1(doc, "4  本次改动：笔画级局部形变")
    add_h2(doc, "4.1  实现机制：梯度掩码")
    add_body(doc, "改动引入梯度掩码（gradient masking）。在每一轮损失反向传播之后、优化器更新之前，将“不需要移动”的笔画所对应控制点的梯度清零。这样这些点在优化器更新时步长为零、坐标纹丝不动；而未冻结的控制点梯度保留，照常被语义损失驱动变形。由于掩码从第 0 步起逐轮应用，被冻结点的动量始终为零，位移恒为零，因此能严格保证“只动指定笔画”。")
    add_h2(doc, "4.2  改动位置与功能")
    add_body(doc, "全部改动位于 optimizer/optimize.py，共三处，另新增两个脚本。第一处，新增掩码计算函数 compute_frozen_mask（第 570 行起）。它根据指定区域，遍历所有 path，按控制点的空间坐标计算“要冻结的控制点行索引”，返回 {形状索引: 要冻结的行索引}。第二处，在训练函数中（第 710 至 719 行）根据配置项 optimize_region 生成该掩码；仅当显式设置区域时才触发，未设置则保持为空，故对原有全局优化流程无影响。第三处，在 loss.backward() 之后、optim.step() 之前（第 936 至 940 行），把掩码标记行的梯度就地清零，实现锁点。新增 run_stroke_opt.py 作为笔画局部优化的入口，将区域与相关参数做成命令行参数；新增 verify_stroke.py 用于验证冻结点位移。")
    add_h2(doc, "4.3  设计逻辑：为何用梯度清零")
    add_body(doc, "锁点采用梯度清零而非关闭 requires_grad，原因有二。其一，requires_grad 是整张量属性，而左斜杠与右斜杠属于同一条外轮廓 path 的同一个张量，必须能精确锁到“行”级别。其二，被冻结的点仍参与整幅图像的渲染和像素级损失的传播，不能从计算图中摘除；梯度清零不改变计算图结构，实现最干净。")
    add_h2(doc, "4.4  验证结果")
    add_body(doc, "以 A 为例，指定只动左斜杠（region=left），目标语义为斜体，100 次迭代。外轮廓左斜杠 77 个控制点可动，右斜杠 82 个点与字腔 54 个点被冻结。逐点坐标比对结果：右斜杠最大位移 0.0000，字腔最大位移 0.0000，左斜杠最大位移 4.23。该结果以数据证明“只有指定笔画发生变形，其余笔画与字腔保持原位”。")

    # 5
    add_h1(doc, "5  语义描述与局部微调的关系")
    add_h2(doc, "5.1  当前分工")
    add_body(doc, "在当前实现中，语义描述不能决定“动哪条笔画”。“动哪里”由 region 参数（left、right、cross、counter、outer）通过控制点几何坐标的规则决定；“往什么方向变”由语义描述（如 italic, slanted font）决定。两者相互独立、正交。语义损失是像素级、作用于整幅字形的，FontCLIP 对其整体打分，并不存在“只给某一条笔画单独打分”的能力，因此语义对单条笔画的针对性有限，这也是局部变形的幅度可能不如全局优化的原因。")
    add_h2(doc, "5.2  让语义真正控制局部的方向")
    add_body(doc, "若希望语义控制局部，可沿三条路径推进：维持现状，语义驱动方向、region 决定位置，二者配合；做“语义到笔画”的自动映射，例如语义含 italic 就自动选出应倾斜的笔画，属半手工规则；或将单一语义扩展为按笔画分组的多目标，给不同笔画分别配不同的语义描述并各自计算损失，这才能实现“每一笔都由语义指挥”，但改动较大，且笔画越多越容易与保形、色调约束冲突，需要更多调参。")

    # 6
    add_h1(doc, "6  其他字母的适用性")
    add_h2(doc, "6.1  字母 path 结构探查")
    add_body(doc, "为验证当前区域规则能否推广到其他字母，对 Arimo 字体多个字符做了 path 结构探查，结果如下表所示。")
    build_table(doc, ["字母", "path 数量", "结构说明", "当前规则是否适用"],
        [["A", "2", "外轮廓 + 1 个字腔", "适用"],
         ["B", "3", "外轮廓 + 2 个字腔", "部分适用（只识别 1 个字腔，会漏）"],
         ["O", "2", "外轮廓 + 1 个字腔", "适用"],
         ["X", "1", "仅外轮廓，无字腔", "counter 区域无效"],
         ["K", "1", "仅外轮廓，无字腔", "counter 区域无效"],
         ["R", "2", "外轮廓 + 1 个字腔", "适用"],
         ["H", "1", "仅外轮廓，无字腔", "counter 区域无效"],
         ["I", "1", "一条竖杠", "左右/字腔规则无意义"],
         ["i", "2", "上方的点 + 下方竖杆", "非“外轮廓+字腔”结构，需特判"]])
    add_h2(doc, "6.2  结论与适配方法")
    add_body(doc, "当前区域规则是按 A 的几何特性定制的，假设了“shape 0 为外轮廓、shape 1 为字腔”，并对左右、横梁用坐标中位线切分。这一假设对其他字母并不普遍成立：多字腔字母（如 B）会漏处理第二个字腔；无字腔字母（如 X、K、H、I）的 counter 区域无效；复合笔画字母（如 i）结构与 A 完全不同。")
    add_body(doc, "若要通用，需要将掩码规则从“按固定 shape 索引”改为“按几何自动分类”：以最大包围盒识别外轮廓，以被外轮廓包含识别字腔；对“某一条笔画”的切分则须逐字母定义（例如 X 的哪一笔、K 的哪一笔），几何规则难以一概而论，通常要人工指定笔画对应的控制点范围。")

    # 7
    add_h1(doc, "7  使用说明")
    add_body(doc, "在 fontclip 环境、项目根目录下运行，需先设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE 以避免 OpenMP 运行库冲突。默认示例只动 A 的左斜杠：python run_stroke_opt.py。")
    add_body(doc, "主要命令行参数：--region 指定要动的笔画，可选 left、right、cross、counter、outer，其余笔画锁死；--semantic_concept 为目标语义描述，决定可动笔画往哪个方向变形；--letter 指定字符，默认 A；--font_path 指定初始字体文件；--num_iter 为迭代次数，默认 100；--lr 为学习率，默认 1.5。")
    add_body(doc, "输出位于 output 目录下自动命名的实验子目录中，关键产物为 output-svg/output.svg（最终矢量字形）与 svg-init/init.svg（初始字形）。运行后可执行 verify_stroke.py 对冻结点与移动点做逐坐标比对，验证“只动了指定笔画”。")

    doc.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()

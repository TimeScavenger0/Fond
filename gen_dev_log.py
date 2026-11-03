# -*- coding: utf-8 -*-
"""Generate the FontCLIP brush-opt development log Word document."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

doc = Document()

# page setup A4, 2.5cm margins
for section in doc.sections:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# default style
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.first_line_indent = Cm(0.74)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    run.font.size = Pt(18)
    run.bold = True
    p.paragraph_format.space_after = Pt(18)
    p.paragraph_format.first_line_indent = Cm(0)

def add_h1(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    run.font.size = Pt(16)
    run.bold = True
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(0)

def add_h2(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    run.font.size = Pt(14)
    run.bold = True
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.first_line_indent = Cm(0)

def add_body(text):
    p = doc.add_paragraph(text)
    return p

add_title("FontCLIP 画笔交互笔画优化开发日志")

add_body("本文档记录在 FontCLIP 项目基础上构建画笔交互式笔画局部优化功能的完整开发过程，包括遇到的技术问题、根因分析和对应解决方案。开发目标是在不影响原有字体检索和全局矢量优化功能的前提下，新增用户画笔圈选笔画、输入语义描述、仅对圈选区域进行局部风格变形的能力。")

add_h1("一、项目背景")

add_body("FontCLIP 原始项目提供两个核心功能：字体检索和矢量优化。矢量优化通过在 FontCLIP latent 空间中最小化余弦距离，将 SVG 字形的控制点逐步变形以匹配目标语义风格。原始实现是全局优化——所有控制点同时移动，无法指定只变形某一个笔画。本次开发在 optimizer/optimize.py 中新增了基于画笔掩码的局部优化分支，并在 run_brush_opt.py 中构建了完整的 Gradio 交互界面。旧脚本 font_retrieval.py、run_vector_opt.py 和 run_stroke_opt.py 一行未改，原有功能完全不受影响。")

add_h1("二、遇到的问题与解决方案")

add_h2("问题 1：画笔掩码与变形区域错位")

add_body("现象：用户在画布下半部分画了掩码，但实际变形发生在上半部分。根因分析：Gradio 的 sketch 组件返回的 mask 尺寸与 diffvg 渲染的 200×200 画布尺寸不一致，导致掩码坐标和 SVG 路径控制点坐标错位。解决方案：在 run_brush_opt.py 中强制将 mask resize 到 200×200（使用 NEAREST 插值保持二值边界），确保掩码像素坐标与 SVG 路径控制点的像素坐标一一对应。")

add_h2("问题 2：上传照片无法直接优化——需要矢量化")

add_body("现象：用户希望直接上传一张字体照片进行优化，但 diffvg 只能操作 SVG 矢量路径，不能直接在像素图上做梯度下降。根因分析：diffvg 的优化对象是 SVG path 的控制点坐标，像素图没有可微分的控制点。解决方案：编写 raster_to_svg 函数，用 skimage.measure.find_contours 做 marching-squares 轮廓提取，将黑白二值图的外轮廓转换为 SVG path。矢量化后的 SVG 加载到 diffvg 中，控制点坐标直接对应 200×200 画布上的像素位置。")

add_h2("问题 3：控制点爆炸导致路径自相交崩坏")

add_body("现象：上传 K 的照片优化后，右上斜杠变成一团乱麻。根因分析：find_contours 输出的是像素级密集轮廓，包含 700 多个顶点，每个顶点之间用直线段连接。优化时 700 多个控制点同时受梯度拉动，路径自相交缠成一团。解决方案：引入 Douglas-Peucker 多边形简化算法（skimage.measure.approximate_polygon），tolerance 参数从最初的 1.8 逐步调到 0.8，将控制点数量降到约 80-120 个，接近真实 TTF 字体的控制点密度。tolerance 太大导致字形失真，太小则控制点过多，0.8 是经验最优值。")

add_h2("问题 4：优化后边缘呈锯齿状")

add_body("现象：笔画变形后边缘不是平滑曲线，而是锯齿状折线。根因分析：矢量化输出的 SVG path 使用 L 命令（直线段），相邻简化点之间是硬转折，不是平滑曲线。真实 TTF 字体使用贝塞尔曲线（C/Q 命令），边缘天然圆滑。优化时梯度把直线段顶点往目标方向拉，折线变形后锯齿更加明显。解决方案：将 SVG path 从 L 命令改为 Q 命令（二次贝塞尔曲线）。具体做法是对简化后的多边形点，每个原始顶点作为 Q 控制点，相邻边的中点作为曲线经过的终点，形成平滑闭合曲线。这样 diffvg 渲染出来的边缘是圆滑的，和真实字体一致。")

add_h2("问题 5：combine_word 函数越界崩溃")

add_body("现象：上传照片矢量化后运行优化，报 IndexError: shapes_word[s] = shapes[j] 索引越界。根因分析：combine_word 函数假设 SVG 中的 shape 数量和结构与正常 TTF 渲染一致（一个字母由固定数量的 shape 组成），但外部矢量化 SVG 的 shape 数量和分组方式不同。解决方案：在 optimizer/optimize.py 中检测 cfg 是否带 brush_mask_image 属性，如果是画笔模式则跳过 combine_word 调用，直接使用原始 shapes。")

add_h2("问题 6：to_pil_image 通道数报错")

add_body("现象：调用 torchvision.utils.to_pil_image 时报错 Got 200 channels。根因分析：to_pil_image 期望输入形状为 (C, H, W)，但传入的 numpy 数组形状是 (H, W, 3)，被错误解析。解决方案：改用 PIL.Image.fromarray 直接从 numpy 数组创建图像，绕过 torchvision 的格式假设。")

add_h2("问题 7：Gradio sketch 组件作为输出时报错")

add_body("现象：将画布组件同时绑定为输入和输出时，Gradio 报 Cannot process this value as an Image。根因分析：gradio 3.40.0 中，source=\"upload\" + tool=\"sketch\" 的组件作为输入时 preprocess 返回 dict {image, mask}，但作为输出时 postprocess 只接受 numpy 或 PIL，不接受 dict。解决方案：run_btn 的 outputs 只绑定 [result_img, status]，不绑定画布。优化完成后结果显示在独立的 Result 图像组件中，用户点 Render glyph 按钮将结果载入画布继续下一笔。")

add_h2("问题 8：二次优化接续逻辑")

add_body("现象：用户完成一笔优化后想在变形结果上继续画另一笔，但画布上还是原始字形。根因分析：优化完成后输出了新的 SVG 文件，但界面没有自动加载它。解决方案：通过 STATE 字典实现链式优化。优化完成后设置 STATE[\"chained_ready\"]=True 和 STATE[\"init_svg_path\"]=输出 SVG 路径。用户点 Render glyph 时，如果 chained_ready 为 True 且字母未变，就从输出 SVG 渲染到画布；如果字母变了（如从 E 切到 A），则重置 chained_ready 走 TTF 重新渲染；如果上传了新图片，也重置。")

add_h2("问题 9：画布显示矢量化近似图与原图有出入")

add_body("现象：用户上传照片后点 Render glyph，画布上显示的不是原图，而是矢量化后重新渲染的近似图，边缘和形状有差异。根因分析：矢量化过程中 Douglas-Peucker 简化和贝塞尔曲线化都会损失细节，重新渲染出来的图和原图不完全一致。解决方案：画布上直接显示上传的原图（resize 到 200×200），不经过矢量化渲染。后台矢量化仅供优化使用，因为两边都是 200×200 像素坐标，mask 位置和 SVG 路径控制点一一对应。用户看到的是自己上传的原图，画的位置就是优化的位置。")

add_h2("问题 10：切换字母后画布残留上次结果")

add_body("现象：完成 E 的优化后切换到 A，点 Render glyph 画布上仍显示 E 的变形结果。根因分析：chained_ready 标志在优化完成后设为 True，切换字母时没有重置，render_glyph 直接载入了上次的输出 SVG。解决方案：在 render_glyph 中检测 STATE[\"letter\"] 与当前输入字母是否一致，如果不一致则重置 chained_ready，强制走 TTF 重新渲染新字母。")

add_h2("问题 11：Font Path 手动输入不便")

add_body("现象：gwfonts 目录下有数百个 TTF 字体，手动输入路径容易出错。解决方案：将 gr.Textbox 改为 gr.Dropdown，启动时用 glob 动态扫描 gwfonts/*.ttf 生成选项列表，默认选中 Arimo-Regular.ttf。")

add_h2("问题 12：无画笔时默认全局优化")

add_body("现象：用户想对整个字形做风格优化而不画局部时，系统报错要求先画笔画。解决方案：检测 mask 是否为空（n_painted == 0），如果为空则不传 brush_mask_image 属性，让优化走原有全局路径，所有控制点同时移动。画了笔画才走局部冻结路径。")

add_h1("三、当前系统架构")

add_body("交互流程为：用户选择字体和字母（或上传照片），点 Render glyph 渲染到画布；在画布上用白色画笔圈选要变形的笔画（不画则全局优化）；在 Semantic target 输入框写风格描述（如 italic, slanted font 或 brush calligraphy, ink stroke）；设置 Iterations 和 Learning rate；点 Start optimization 运行。CPU 模式下 100 步约 70 秒，200 步约 3 分钟。优化完成后结果显示在 Result 区域，点 Render glyph 可载入结果继续下一笔。")

add_body("代码改动集中在两个文件：run_brush_opt.py 是画笔交互主脚本，包含 raster_to_svg 矢量化函数、render_glyph 渲染逻辑、run_optimization 优化入口和 Gradio UI；optimizer/optimize.py 新增了 compute_frozen_mask_from_brush 函数和 elif 分支，当 cfg 带 brush_mask_image 时走画笔掩码路径，否则保持原有 region/全局优化逻辑不变。")

add_h1("四、关键参数说明")

add_body("Iterations 控制优化步数，步数越多变形越远但越容易过冲，建议局部微调 100-200 步。Learning rate 控制每步移动幅度，建议局部微调不超过 2.0，否则笔画容易自相交。approx_tolerance 控制矢量化简化程度，当前设为 0.8 像素，平衡保真度和控制点数量。渲染画布固定 200×200，char_size=150。")

doc.save(r"D:\VScode\VScodeProjects\Fond\FontCLIP画笔优化开发日志.docx")
print("done")

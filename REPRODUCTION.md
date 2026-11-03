# FontCLIP 融合系统复现指南

本指南面向小组成员在各自电脑上复现 FontCLIP 融合系统，包括六项改进的矢量字形优化与画笔局部编辑交互界面。全部流程基于 **CPU 环境实测**，无需 GPU 即可运行；Windows 用户建议优先使用 WSL，Linux 可直接按步骤操作。

## 一、环境要求

- 操作系统：Linux（Ubuntu 20.04/22.04）或 Windows 10/11（建议 WSL2）
- Anaconda 或 Miniconda（[下载地址](https://docs.conda.io/en/latest/miniconda.html)）
- Python 3.8
- 编译工具：g++、CMake（Linux/WSL）；或 Visual Studio Build Tools + CMake（Windows 原生）
- 磁盘空间：约 3GB（代码 + 权重 + 环境）

## 二、获取代码

```bash
git clone https://github.com/TimeScavenger0/Fond.git
cd Fond
```

## 三、创建 conda 环境

```bash
conda create -y --name fontclip python=3.8
conda activate fontclip
```

## 四、安装 CPU 版 PyTorch

实测版本为 torch 1.13.1 / torchvision 0.14.1 / torchaudio 0.13.1（CPU 版）：

```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

## 五、安装其余依赖

```bash
pip install tqdm ftfy regex gdown gradio==3.40.0 httpx==0.24.1
pip install numpy==1.24.3 scipy==1.10.1 pillow==10.4.0
pip install scikit-image==0.19.3 scikit-fmm opencv-python==4.5.4.60
pip install svgwrite svgpathtools cssutils numba torch-tools
pip install easydict pyyaml matplotlib freetype-py shapely kornia==0.6.8
```

## 六、编译安装 diffvg（关键步骤）

代码通过 `pydiffvg` 导入 diffvg，它必须从源码编译，是复现过程中最容易出错的一步。

**Linux / WSL（推荐）**，先确认已安装 g++ 和 CMake：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y g++ cmake
```

然后编译安装：

```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ..
```

验证安装：

```bash
python -c "import pydiffvg; print('diffvg OK')"
```

**Windows 原生**：需安装 Visual Studio Build Tools（勾选"使用 C++ 的桌面开发"）和 CMake，再在 "x64 Native Tools Command Prompt" 中执行同样的 `python setup.py install`。由于 diffvg 官方主要支持 Linux，Windows 原生编译问题较多，强烈建议改用 WSL2。

## 七、安装 openai CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

CLIP 的 ViT-B/32 权重（ViT-B-32.pt）会在首次运行时自动下载到 `~/.cache/clip/`，无需手动放置；若电脑无法联网，需从其他机器拷贝该文件到对应目录。

## 八、下载模型权重

项目仓库不含大权重文件（`.pt`/`.pth` 已通过 `.gitignore` 排除），需从 GitHub Release 下载：

1. 打开仓库的 Releases 页面：<https://github.com/TimeScavenger0/Fond/releases>
2. 下载附件 `fontclip_weights_v1.0.zip`（约 1.43GB）
3. 将 zip 解压到项目根目录（Fond/），保持目录结构，权重会自动归位：

```
model_checkpoints/fontclip_lora_best.pt        # ViT-B/32 + LoRA 训练权重（r=256, alpha=1024）
.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth  # VGG19 感知损失权重
```

zip 内附 `README_weights.txt` 说明。若运行时设置了自定义 `TORCH_HOME`，VGG19 权重需放到 `%TORCH_HOME%/hub/checkpoints/` 下。

## 九、运行画笔局部优化界面

```bash
python run_brush_opt.py
```

启动后浏览器打开 <http://127.0.0.1:7861>，操作流程：

1. 选择字母与初始字体，输入语义描述（如 `italic, slanted font`），点击 Render glyph 渲染字形
2. 在画布上用画笔圈选想要变形的笔画区域
3. 点击 Start optimization，圈内控制点向语义目标优化，圈外完全冻结
4. 可通过 ref_weight 滑杆调节参考图视觉迁移强度（0–5.0）

## 十、其他入口脚本

| 脚本 | 功能 |
| --- | --- |
| `run_brush_opt.py` | 画笔局部编辑交互界面（Gradio，端口 7861） |
| `run_stroke_opt.py` | 笔画级字形优化 |
| `run_vector_opt.py` | 矢量字形整体优化 |
| `font_retrieval.py` | 文本/图像字体检索 |
| `output/` | 实验结果 CSV 与 JSON（主对比、消融、画笔指标） |

## 十一、常见问题

**运行时报 OpenMP 重复库错误（OMP: Error #15）**：设置环境变量后再运行。

```bash
# Linux / WSL
export KMP_DUPLICATE_LIB_OK=TRUE
# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

**VGG19 权重找不到**：确认 `.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth` 存在，或设置 `TORCH_HOME` 指向 `.cache/torch` 的上级目录。

**diffvg 编译失败**：确认 g++/CMake 已安装且版本匹配；Windows 不要在原生环境反复尝试，直接使用 WSL2。

**运行速度**：CPU 上约 1.2–1.4 步/秒，300 步优化约 4 分钟/字母，开启 best-of-3 采样约 12 分钟/字母，属正常现象。

**CSV 用 Excel 打开出现 #NAME?**：确认配置名单元格不以 `- ` 开头；本仓库提供的表格已修复，编码为带 BOM 的 UTF-8。

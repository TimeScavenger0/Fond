# 云端 GPU 训练部署操作文档（中文印刷体 LoRA 续训）

本文档指导在国内 GPU 租用平台上，用 `train_cn_mixed.py` 完成中文 + 拉丁混合 LoRA 续训（训练方案 S0）。所有命令均已在真实 Linux 环境验证，重点坑点（中文文件名解压）已在第六节标注。

## 一、方案总览与成本

平台与卡型方面，推荐 AutoDL（https://www.autodl.com）或 Featurize（https://featurize.cn），卡型选 RTX 3090 或 4090（24GB 显存），单卡即可。参考价格（2026 年 11 月，以平台实时显示为准）：AutoDL 4090/3090 约 ¥1.88/时（会员 95 折），Featurize 4090 约 ¥1.87、3090 约 ¥1.62/时，均按分钟计费，不满一小时不扣整小时。

成本方面，ViT-B/32 是小模型，正式配置 100 epoch、每 epoch 16 个 batch，单卡 S0 预计 1–2 小时（以首次实测吞吐为准），整体花费约 ¥2–4。建议先充值 10–20 元即可覆盖，无需大额充值。

## 二、注册与充值（需本人操作）

打开平台官网注册账号并完成实名认证（GPU 平台均要求实名）。在费用中心充值，建议先充 10 元。这一步助手无法代做，账号与支付信息只在你本人手中。

## 三、创建 GPU 实例

在控制台选择「租用新实例 / 租用实例」，计费方式选「按量计费」，GPU 型号选 RTX 3090 或 4090，数量 1 张，挑选一台空闲主机。

镜像选择「基础镜像」中的 PyTorch 框架，CUDA 版本选 11.8 或 12.1 均可（对应 PyTorch 2.0.x 或 2.1.x），Python 3.8–3.10 都兼容。无需选社区镜像，默认 PyTorch 镜像已预装 CUDA 版 torch、torchvision，可省去自己装 GPU 版 torch 的步骤。硬盘保持默认即可，本项目数据解压后约 300MB，不占用额外扩容盘。

创建后等待实例状态变为「运行中」，从此时开始按分钟计费。打开平台提供的 JupyterLab（AutoDL）或「工作区」（Featurize），在其中打开一个终端（Terminal），后续命令都在终端执行。

## 四、获取代码

在终端执行，把仓库克隆到家目录（AutoDL 的 JupyterLab 工作目录为 `/root`，Featurize 建议放在 `~/work`）：

```bash
cd ~
git clone https://github.com/TimeScavenger0/Fond.git
cd Fond
```

若网络克隆较慢，也可在平台选择加速地址。克隆后确认根目录存在 `train_cn_mixed.py` 与 `cloud_deploy/` 目录。

## 五、上传数据 bundle

把本地的 `cloud_deploy/fond_cloud_bundle.zip`（约 155MB）通过平台界面上传。AutoDL 在 JupyterLab 左侧文件栏点上传按钮；Featurize 在工作区左上角上传。建议直接上传到 `Fond/` 目录内，使压缩包与项目根同级。

上传后在终端确认文件就位：

```bash
cd ~/Fond
ls -la fond_cloud_bundle.zip
```

## 六、解压数据（关键：必须用 Python，不要直接用 unzip）

这是最重要的坑点。系统自带的 Info-ZIP 6.00 对带 UTF-8 标志位的中文文件名处理有缺陷，直接用 `unzip fond_cloud_bundle.zip` 会把唯一的中文文件名字体 `方正粗黑宋简体.ttf`（以及对应 PNG）解成乱码，导致后续按文件名加载失败。

务必用 Python 解压，它对 UTF-8 文件名的支持可靠：

```bash
cd ~/Fond
python3 -m zipfile -e fond_cloud_bundle.zip .
```

解压后核对目录与中文文件名：

```bash
ls cloud_fonts | wc -l                 # 应输出 64
ls cloud_fonts/方正粗黑宋简体.ttf        # 能正常列出即中文文件名正确
ls data_cn/mixed                       # 应看到 9 个 JSON 与三个图片目录
```

`cloud_fonts/` 是 64 个真实字体（24 中文 + 40 拉丁），`data_cn/mixed/` 含 9 个 JSON 与 64 张 PNG。这两个目录的位置正好对齐 `train_cn_mixed.py` 的自动重映射规则，无需再改任何路径。

## 七、环境自检

执行一键自检，它会检查 GPU、CUDA 版 torch、安装轻量依赖、下载基础权重、确认数据就位：

```bash
bash cloud_deploy/setup_cloud.sh
```

看到末尾 `SETUP COMPLETE` 即环境就绪。脚本会自动从 GitHub Release 下载 `fontclip_lora_best.pt`（约 887MB）到 `model_checkpoints/`；若机器已自带该文件则跳过。训练不需要 VGG19 权重，无需下载。

## 八、启动训练（先 smoke 再正式）

直接运行启动脚本，它会先用极小配置跑一遍 smoke 验证全链路，通过后自动进入正式训练：

```bash
bash cloud_deploy/run_train.sh
```

smoke 阶段在 GPU 上约 1 分钟内完成。确认 smoke 通过后，正式训练使用默认参数（epoch=100、batch_size=48、lr=1e-5）。如需调整，可通过环境变量覆盖，例如：

```bash
EPOCH=50 BATCH_SIZE=32 bash cloud_deploy/run_train.sh
```

若已确认环境无误、想跳过 smoke，可执行 `SKIP_SMOKE=1 bash cloud_deploy/run_train.sh`。训练过程中每个 epoch 的 loss 与验证相关系数会打印在终端，注意观察 loss 是否平稳下降。

## 九、下载训练产物

训练完成后，新的 LoRA 权重保存在 `model_checkpoints/` 下（文件名含 batch、aug、lr 等配置信息）。通过 JupyterLab/工作区右键该文件下载到本地，或用 SCP/SFTP 拉取。建议同时保存终端训练日志，便于后续写论文引用。

## 十、关机与释放（省钱关键）

训练和下载都完成后，立即在控制台「关机」实例，停止 GPU 按分钟计费。注意关机只是停止计费、数据仍保留，下次可再开机继续；若确认不再使用，选择「释放实例」彻底删除。

AutoDL 若未扩容付费数据盘，关机后不会产生额外费用；若曾扩容数据盘，该盘按约 0.0066 元/日/GB 计费直到释放，不用时记得释放。Featurize 归还实例前可用 `du -sh ~/work` 核对云盘用量，避免超额度。

## 十一、故障排查

**中文文件名乱码**：是误用了 `unzip` 命令所致。删除已解压的 `cloud_fonts`、`data_cn`、`bundle_manifest.json`，改用第六节的 `python3 -m zipfile -e` 重新解压。

**torch 报 CUDA 不可用或为 CPU 版**：说明镜像选择不对。关机后更换为带 PyTorch 的 GPU 基础镜像（更换镜像只清系统盘、不影响数据盘），或按下面附录手动安装 CUDA 版 torch。

**找不到字体 / font not found**：先确认 `cloud_fonts` 下有 64 个文件且中文文件名正常，再确认是在项目根 `Fond/` 下执行的命令。

**权重下载中断**：重新运行 `python3 cloud_deploy/download_checkpoint.py`，脚本会校验字节数，已完整下载的文件会自动跳过。

## 附录：手动安装 CUDA 版 torch（仅当镜像未预装时）

正常选 PyTorch 镜像无需此步。若确实需要自行安装，先确认 CUDA 版本（`nvidia-smi` 右上角），再执行对应命令，例如 CUDA 11.8：

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

若官方源较慢，可加平台内网加速或换用清华镜像。安装后用 `python3 -c "import torch; print(torch.cuda.is_available())"` 确认输出 `True`，再安装轻量依赖 `pip install -r cloud_deploy/requirements-cloud.txt`。

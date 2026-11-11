# S0 中文混合 LoRA 续训验证发现（#54）

本文件记录在 Featurize RTX 3090 上完成的 100 epoch S0 训练（仅主 BCE 对比损失，
未加模块 A–D）的本地客观验证结果。所有数字均来自真实实验产物，未做估算或填充。

## 一、训练过程

正式训练在混合数据集（38 字体，14 中文 + 24 拉丁 replay，29 个活跃属性）上
从 `fontclip_lora_best.pt` 续训，100 epoch 完整跑完。逐 epoch 数据见
`train_curve.csv`，曲线见 `train_curve.png`。

最佳验证相关性出现在 epoch 22（checkpoint 内部计数 21），val_correlation=0.6442、
当时 loss=4.6456，best 权重即该点保存。此后训练 loss 继续下降（epoch 100 为
3.3115），但 val_correlation 回落到 0.60–0.62 区间（epoch 100 为 0.6021），
说明 22 epoch 后出现轻微过拟合，best 权重选择正确。

## 二、中文属性对齐能力：实质性提升

用新旧两个模型在混合验证集上计算 CLIP 余弦相似度与真实属性分数的 Pearson 相关
（`evaluate_correlation_coefficient` 同口径），结果见
`cn_val_compare_summary.csv` 与 `cn_val_compare_per_attr.csv`。

整体 13 字体的平均相关从 0.4773 提升到 0.6452（+0.1680）。仅看 5 个中文验证
字体，从 0.1922 提升到 0.4809（+0.2887）：旧拉丁 LoRA 在中文上接近随机，
混合续训后中文图文对齐具备了实际可用性。在整体 13 字体的 29 个属性中约 20 个
提升，complex（0.12→0.68）、artistic、dramatic、playful、modern、technical、
disorderly 等原本接近随机的复杂/主观属性提升到 0.5–0.7，客观属性
thin（0.85→0.87）、strong、warm 保持在高位。

## 三、拉丁防退化：目标达成

8 个拉丁验证字体的平均相关从 0.8179 变为 0.8144，仅 −0.0034，在验证噪声范围内。
拉丁 replay（24 字体）与小学习率续训有效防止了加中文导致的拉丁能力退化，
原始功能未受实质影响。

## 四、属性纠缠：S0 未缓解（负面发现，S1 的动机）

按训练方案定义，属性方向取 `encode_text("a font") − encode_text("not a font")`
归一化后的方向向量，计算 29×29 方向余弦矩阵（数据见
`attribute_direction_cosine.csv`，热力图见 `attribute_direction_heatmap.png`）。

非对角方向余弦的平均绝对值从 0.2171 变为 0.2236（+0.0066），整体纠缠程度基本
持平、未下降。逐对看方向不一致：strong/angular（0.079→0.024）、
boring/playful（0.308→0.196）有改善，但 serif/formal（0.156→0.221）反而更纠缠，
thin/delicate 基本持平。多个语义对立对在 CLIP 文本空间方向并非相反，
soft/sharp 余弦高达 +0.50、calm/dramatic 为 +0.14，训练后也未被推开。

结论是：仅靠主 BCE 图文对齐与数据混合，无法系统性降低属性纠缠，因为 BCE 目标
本身不约束属性方向的正交性。这与方案预期一致，也构成 S1 加入模块 A（属性方向
软正交约束）的明确动机与基线——需要显式正交干预，而非期待其自然涌现。

## 五、口径与局限

中文验证子集仅 5 个字体、拉丁 8 个，单属性 Pearson 相关在小样本下置信区间宽、
波动大，因此单属性数字（尤其中文子集内 italic、serif 等）只作参考，结论以
子集平均相关与整体 13 字体的逐属性结果为准。中文主观属性标签来自线性回归伪标签，
其噪声会同时计入验证相关，后续若引入少量人工标注可进一步收紧。S1–S3 模块
（正交化、独立判别头、内容-风格分离与结构锚定）尚未训练，本文件只覆盖 S0。

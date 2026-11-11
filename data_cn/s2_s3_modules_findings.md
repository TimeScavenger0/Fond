# S2/S3 模块 B/C/D（独立属性判别头、内容-风格分离、结构锚定）实现与本地验证发现

本文件记录模块 B、C、D 的代码实现与本地验证。三个模块在 S1 模块 A 之外，分别从
"属性可独立读出""内容-风格子空间解耦""结构分量稳定"三个角度，针对 S0 已证实的
属性纠缠与复杂结构问题提供可消融的约束。按用户决策，模块 B/C/D 与模块 A 合并为
一次云端训练，本阶段先在本地完成代码与 smoke 验证，不单独上云。

## 一、实现内容（向后兼容，默认关闭）

新建 `models/aux_modules.py`，包含两个模块与四个损失函数。模块 B 是
`AttributeHeads`，在归一化图像特征上接一个 `nn.Linear(512, n_attr)`，每个活跃
属性对应一个相互独立的二分类头，用 BCE 监督属性可被独立读出。模块 C 是
`ContentStyleProjector`，用 `content_proj`、`style_proj` 两个无偏置
`nn.Linear(512,512)` 把特征拆到内容、风格两个子空间，配套风格对比
（supervised InfoNCE，同字体聚拢、异字体分离）、子空间正交
（`||Wc Ws^T||_F^2`）、重构保持（`Pc f + Ps f` 还原原特征）三项损失。模块 D
是模块 C 的强化项，把当前内容分量与标准锚点的内容投影对齐，约束风格更新不带动
内容漂移。

`trains/config_fine_tune.py` 新增八个字段：`use_attr_heads`、`attr_head_weight`
（默认 1.0）、`use_content_style`、`cs_style_weight`（1.0）、`cs_orth_weight`
（0.1）、`cs_recon_weight`（0.5）、`use_structure_anchor`、`anchor_weight`
（1.0），全部默认关闭或取保守值，S0/S1 行为不变。signature 依次追加
`_attrh{weight}`、`_cs{style}-o{orth}-r{recon}`、`_anchor{weight}` 标记。
`train_cn_mixed.py` 新增对应的八个命令行参数并透传。

## 二、关键设计：独立模块加独立优化器，不污染核心模型

`load_model` 用 `strict=True` 加载核心 CLIP 权重，若把辅助头挂到核心模型上，
保存与重新加载都会因参数名不匹配而报错。因此辅助模块独立于核心 model 存在，
由 trainer 单独构建、用独立 `aux_optimizer`（lr=config.lr、betas=(0.9,0.98)、
eps=1e-6、weight_decay=0.2）管理，GPU 上对半精度模块单独 `.half()`，核心模型的
strict 加载与保存逻辑完全不受影响。

模块 B 的二值目标由新增方法 `build_attr_targets` 从
`font_idx_signed_attribute_matrix` 按 `all_attributes.index(a)+1` 取列、signed>0
转 0/1 得到。模块 C/D 用归一化图像特征计算，且不 detach，使辅助损失的梯度既能
训练辅助模块本身，也能通过共享编码器反传，推动 LoRA 特征真正变得可分、可拆。
模块 D 的锚点用标准 preprocess（非随机增强的 tmp_preprocess）从 image_file_dir
的 `<font_name>.png` 逐字体编码并缓存，启用时 assert 要求模块 C 同时存在。

## 三、本地 smoke：四个配置全部跑通

| 配置 | signature 标记 | epoch_loss | val_correlation |
| --- | --- | --- | --- |
| 模块 B 单独 | `_attrh1.0` | 23.8687 | 0.4593 |
| 模块 C 单独 | `_cs1.0-o0.1-r0.5` | 137.5254 | 0.4859 |
| 模块 C+D | `_cs1.0-o0.1-r0.5_anchor1.0` | 29.6707 | 0.4721 |
| 全模块 A+B+C+D | `_orth0.1-b4.0_attrh1.0_cs1.0-o0.1-r0.5_anchor1.0` | 78.9383 | 0.5005 |

四个配置均完整跑完（smoke 19 batch）、无报错、best 正常保存，val_correlation 都
保持在 0.46–0.50 的正相关区间，说明辅助模块接入后训练链路与图文对齐能力正常。

## 四、调试中发现并修复的两个问题（重要）

第一个是风格对比损失返回 NaN。根因是 `log_softmax` 后自身位置为 `-inf`，再与
正样本掩码的 0 相乘时 `-inf * 0 = NaN`，沿 sum 污染整行与整个 total_loss。
修复方式是改用 `torch.where(pos_mask, log_prob, zeros)`，先把非正样本位置
（含自身的 -inf）替换为 0 再求和，彻底避免 NaN；整批无同字体正样本对时仍安全
返回 0 标量。修复后各 smoke 损失全部有限。

第二个是子空间正交损失尺度偏大。该损失最初用 512×512 个元素的平方和，初始值
约 56，在默认权重 0.1 下每 batch 仍贡献约 5.6，明显大于主 BCE（每 batch 1–2），
存在主导训练的风险。修复方式是按输出维度 d=512 归一化，变为每维平均，初始值
降到约 0.11，乘权重后约 0.011，不再主导。重构与锚定损失采用按 512 维求和再按
batch 平均的口径，实测尺度合理，C+D 与全模块 smoke 均无异常。

## 五、局限、正式云端判定口径与下一步

本地 smoke 只验证管线可跑通与损失有限，1 个 epoch、batch=4 无法判断辅助约束对
属性纠缠、结构偏差与最终生成质量的真实作用，相关结论必须来自完整云端训练。正式
合并训练应同时满足：属性纠缠（mean abs cosine 与三对易纠缠）相对 S0 下降，
val_correlation 与中/拉丁属性相关不明显回落，且模块 C 的子空间正交、重构误差与
模块 D 的锚定损失随训练收敛而非长期主导。若任一辅助项使 val_corr 显著回落或
loss 异常，应调小对应权重（attr_head_weight、cs_orth_weight、cs_recon_weight、
anchor_weight）而非追求各约束归零。

下一步按 #65 重建含 `models/aux_modules.py` 与最新 config/trainer/train_cn_mixed
的云端代码补丁，重新同步数据 bundle，重新租用云 GPU 后用全模块参数
（`--use_orth --use_attr_heads --use_content_style --use_structure_anchor`）一次
跑完整训练，再按上述双指标判定。是否现在上云，及各权重是否沿用默认值，待确认。

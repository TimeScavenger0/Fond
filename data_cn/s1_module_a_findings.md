# S1 模块 A（属性方向软正交）实现与本地验证发现

本文件记录模块 A 的代码实现与本地验证。模块 A 直接针对 S0 已证实的负面发现
（仅靠 BCE 与数据混合，属性纠缠不下降，mean abs cosine 0.2171→0.2236）。

## 一、实现内容（向后兼容，默认关闭）

`trains/config_fine_tune.py` 新增四个字段：`use_orth_loss`（默认 False）、
`orth_loss_weight`（默认 0.1）、`orth_entangled_boost`（默认 4.0）、
`orth_entangled_pairs`（默认 None，用内置三对）。启用时 signature 追加
`_orth{weight}-b{boost}` 标记，与 S0 权重区分。

`trains/trainer.py` 新增 `build_orth_loss`：对每个活跃属性编码
`encode_text("a font") − encode_text("not a font")`，归一化得到方向矩阵，
用方向余弦平方惩罚非对角耦合；基础权重作用于全部属性对，点名易纠缠对
（serif/formal、thin/delicate、strong/angular）额外乘 boost。训练循环在主
BCE 损失后、反向传播前接入。`train_cn_mixed.py` 新增 `--use_orth`、
`--orth_weight`、`--orth_boost` 命令行参数。所有默认值保持 S0 行为不变。

## 二、本地 smoke：管线跑通

`python train_cn_mixed.py --smoke --use_orth` 完整跑通，signature 正确含
`_orth0.1-b4.0`，正交损失参与反传并正常保存 best，无报错。

## 三、机理验证：正交确实能消除纠缠（verify_orth_effect.py）

为隔离因果，从 S0 best 权重出发，只对文本 LoRA 与 ln_final（12,583,936 个
参数）施加正交梯度，不加载图像、不跑 BCE，lr=1e-4 跑 150 步。正交损失从
step30 的 0.0074 收敛到 step90 后的 0.0000。训练前后方向余弦：

| 指标 | before | after |
| --- | --- | --- |
| 平均绝对非对角余弦 | 0.2236 | 0.0000 |
| serif/formal | 0.2214 | 0.0000 |
| thin/delicate | 0.2287 | 0.0000 |
| strong/angular | 0.0241 | 0.0000 |

结论：模块 A 的机理成立，文本 LoRA 子空间有足够自由度把属性方向推到相互正交，
S0 中无法下降的纠缠在显式正交约束下可以被系统性消除。

## 四、局限与正式 S1 的判定口径（重要）

这是一个无 BCE、无图像约束的理想化上界实验，只证明"能降纠缠"，不代表真实
训练效果。完全正交不等于语义正确：语义上客观相关的属性对（如 soft/gentle）
本就不该被强行推到零相关，过度正交可能损害图文对齐。真实 S1 是 BCE 与正交
同时作用、lr=1e-5、软权重 0.1，强度远温和于此实验。

因此正式云端 S1 训练必须同时满足两个条件才算成功，缺一不可：一是属性纠缠
（mean abs cosine 与三对易纠缠）相对 S0 明显下降；二是 val_correlation 与
中文/拉丁属性相关不能明显下降（图文对齐能力保持）。若正交使 val_corr 显著
回落，应调小 orth_loss_weight 或对客观相关对降权，而非追求余弦归零。

## 五、下一步

模块 A 代码与本地验证已完成，正式 S1 需重新租用云 GPU（Featurize 实例已退还），
同步含 S1 改动的代码后用 `--use_orth` 跑完整训练，再按上述双指标判定。
是否现在上云，以及正交权重是否沿用 0.1，待确认。

import os
from copy import copy
from dataclasses import dataclass
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim

from models.aux_modules import (
    AttributeHeads,
    ContentStyleProjector,
    style_contrast_loss,
    subspace_orth_loss,
    reconstruction_loss,
    content_anchor_loss,
)

from models.init_model import (
    device,
    preprocess,
    convert_weights,
)
from utils.initialize_font_data import (
    font_dir,
    validation_json_path,
    inclusive_attributes,
    exclusive_attributes,
    all_attributes,
)
from evals.evaluate_tools import (
    evaluate_correlation_coefficient,
)
from dataset.dataset import (
    MyDataset,
    set_image_tensors,
    TestImageDataset,
    TestTextDataset,
)
from trains.config_fine_tune import Config
from utils.tokenizer import tokenize

# 模块 A 默认点名的易纠缠属性对（训练方案第四节）
DEFAULT_ENTANGLED_PAIRS = [
    ("serif", "formal"),
    ("thin", "delicate"),
    ("strong", "angular"),
]


@dataclass
class Trainer:
    config: Config = None

    def __post_init__(self):
        pass

    def build_orth_loss(self, attributes):
        """模块 A：属性方向软正交损失。

        属性方向 d_a = encode_text("a font") - encode_text("not a font")，归一化后
        用方向余弦平方惩罚非对角耦合。基础权重 orth_loss_weight 作用于全部属性对，
        点名易纠缠对额外乘 orth_entangled_boost。梯度经文本编码器反传，推动方向正交。
        """
        model = self.config.model
        cfg = self.config

        pos_prompts = [MyDataset.generate_prompt(a) for a in attributes]
        neg_prompts = [MyDataset.generate_prompt(a, negative=True) for a in attributes]
        pos_tok = tokenize(pos_prompts).to(device)
        neg_tok = tokenize(neg_prompts).to(device)
        f_pos = model.encode_text(pos_tok)
        f_neg = model.encode_text(neg_tok)
        direction = f_pos - f_neg
        direction = direction / torch.norm(direction, dim=1, keepdim=True)
        cosine = direction @ direction.T

        n = len(attributes)
        weight = torch.full((n, n), cfg.orth_loss_weight,
                            device=device, dtype=cosine.dtype)
        pairs = cfg.orth_entangled_pairs
        if pairs is None:
            pairs = DEFAULT_ENTANGLED_PAIRS
        for a, b in pairs:
            i, j = attributes.index(a), attributes.index(b)
            boosted = cfg.orth_loss_weight * cfg.orth_entangled_boost
            weight[i, j] = boosted
            weight[j, i] = boosted

        upper = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
        orth_loss = torch.sum(weight[upper] * cosine[upper].pow(2))
        return orth_loss

    def build_attr_targets(self, dataset, font_indices, attributes):
        """模块 B 目标：从 signed matrix 提取活跃属性的二值标签（0/1）。"""
        mat = dataset.font_idx_signed_attribute_matrix
        rows = mat[font_indices.detach().cpu()]
        cols = [all_attributes.index(a) + 1 for a in attributes]
        signed = rows[:, cols]
        return (signed > 0).to(torch.float32).to(device)

    def build_weighted_sampler(self, dataset):
        """
        构造 WeightedRandomSampler，保证中文样本实际采样占比 = target_sampled_cn_share。
        每个字体内部样本权重相同；中文整体权重=target，拉丁整体=1-target。
        """
        cn_set = set(self.config.cn_font_names or [])
        font_names = dataset.font_names
        n_cn = sum(1 for n in font_names if n in cn_set)
        n_latin = len(font_names) - n_cn
        assert n_cn > 0 and n_latin > 0, (n_cn, n_latin)
        target = self.config.target_sampled_cn_share
        w_cn = target / n_cn
        w_latin = (1.0 - target) / n_latin
        per_font_w = [w_cn if name in cn_set else w_latin for name in font_names]

        if self.config.use_multiple_attributes:
            k = self.config.sample_num_each_epoch
            weights = [w for fw in per_font_w for w in [fw] * k]
        else:
            weights = []
            for name, fw in zip(font_names, per_font_w):
                n_attr = len(dataset.font_to_attributes[name])
                weights.extend([fw] * n_attr)

        weights = torch.tensor(weights, dtype=torch.double)
        assert len(weights) == len(dataset), (len(weights), len(dataset))
        sampler = WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True
        )
        print(
            f"WeightedRandomSampler: n_cn={n_cn}, n_latin={n_latin}, "
            f"target_cn_share={target}, w_cn={w_cn:.5f}, w_latin={w_latin:.5f}"
        )
        return sampler

    def train(self):
        tmp_exclusive_attributes = copy(
            self.config.exclusive_attributes_override
            if self.config.exclusive_attributes_override is not None
            else exclusive_attributes
        )
        # 基于实际生效的 exclusive 计算 inclusive，保证与数据活跃属性一致
        tmp_inclusive_attributes = [
            a for a in all_attributes if a not in tmp_exclusive_attributes
        ]
        tmp_validation_json_path = (
            self.config.validation_json_path_override
            if self.config.validation_json_path_override is not None
            else validation_json_path
        )
        val_texts_for_font_image = self.config.val_texts_for_font_image


        # define training dataset
        dataset = MyDataset(
            self.config.font_dir,
            self.config.json_path,
            texts_for_font_image=self.config.texts_for_font_image,
            use_negative=self.config.use_negative,
            use_multiple_attributes=self.config.use_multiple_attributes,
            use_random_attributes=self.config.use_random_attributes,
            random_prompts_num=self.config.random_prompts_num,
            max_sample_num=self.config.max_sample_num,
            rich_prompt=self.config.rich_prompt,
            sample_num_each_epoch=self.config.sample_num_each_epoch,
            image_file_dir=self.config.image_file_dir,
            attribute_threshold=self.config.attribute_threshold,
            attribute_under_threshold=self.config.attribute_under_threshold,
            preprocess=self.config.tmp_preprocess,
            dump_image=self.config.train_dump_image,
            exclusive_attributes=tmp_exclusive_attributes,
            geta=self.config.geta,
            use_bce_loss=True,
            char_size=self.config.char_size,
            context_length=self.config.context_length,
        )
        print(
            "train prompts num: ",
            sum([len(v) for v in dataset.font_to_attributes.values()]),
        )
        print("train dataset font num: ", len(dataset.font_paths))

        # if use_aug, then create images
        if self.config.use_aug or self.config.tmp_preprocess != preprocess:
            set_image_tensors(
                dataset,
                preprocess=self.config.tmp_preprocess,
                sample_num=self.config.sample_num,
                color_jitter_sample_num=self.config.color_jitter_sample_num if self.config.use_color_jitter else 0
            )
            print(
                f"{sum([len(image_tensors) for image_tensors in dataset.font_text_to_image_tensors])} images are randomly created."
            )


        # define validation dataset
        val_datasets = None
        val_font_dir = (
            self.config.val_font_dir
            if self.config.val_font_dir is not None
            else self.config.font_dir
        )
        val_image_dataset = TestImageDataset(
            val_font_dir,
            tmp_validation_json_path,
            val_texts_for_font_image[0],
            dump_image=self.config.tmp_dump_image,
            image_file_dir=self.config.image_file_dir_for_validation,
            preprocess=preprocess,
        )
        text_dataset = TestTextDataset(
            target_attributes=tmp_inclusive_attributes,
            context_length=self.config.context_length,
        )


        # Define dataloader
        if self.config.target_sampled_cn_share > 0:
            train_sampler = self.build_weighted_sampler(dataset)
            train_dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                sampler=train_sampler,
                pin_memory=False,
                num_workers=0,
            )
        else:
            train_dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                pin_memory=False,
                num_workers=0,
            )

        print(f"Checkpoint Path: {self.config.checkpoint_path}")

        if self.config.use_unpretrained_model:
            print("Warning: use unpretrained model")
            self.config.model.initialize_parameters()
        print(f"Signature: {self.config.signature}")

        # https://github.com/openai/CLIP/issues/57
        def convert_models_to_fp32(model):
            for p in model.parameters():
                p.data = p.data.float()
                if p.grad is not None:
                    p.grad.data = p.grad.data.float()

        if device == "cpu":
            self.config.model.float()
        else:
            # Actually this line is unnecessary since clip by default already on
            # float16
            convert_weights(self.config.model)

        # freeze the model first
        for param in self.config.model.parameters():
            param.requires_grad = False

        # unfreeze the target layers in text encoder
        for name, param in self.config.model.transformer.named_parameters():
            if any(
                [
                    name.startswith(target_layer)
                    for target_layer in self.config.target_layers_text
                ]
            ):
                param.requires_grad = True

        # unfreeze the target layers in vision encoder
        for name, param in self.config.model.visual.named_parameters():
            if any(
                [
                    name.startswith(target_layer)
                    for target_layer in self.config.target_layers_vision
                ]
            ):
                param.requires_grad = True

        # unfreeze the target modules of OFT
        if self.config.use_oft_vision or self.config.use_oft_text:
            print("Optimize OFT")
            for name, param in self.config.model.named_parameters():
                if "oft" in name:
                    param.requires_grad = True

        # unfreeze the target modules of LoRA
        if self.config.use_lora_vision or self.config.use_lora_text:
            print("Optimize LoRA")
            for name, param in self.config.model.named_parameters():
                if "lora_proj" in name or (
                    (
                        self.config.lora_config_vision.learnable_alpha
                        or self.config.lora_config_text.learnable_alpha
                    )
                    and "lora_alpha" in name
                ):
                    if "visual" in name:
                        if self.config.use_lora_vision:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        if self.config.use_lora_text:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        # unfreeze the tokens of CoOp
        if self.config.use_coop_text:
            print("Optimize prompt")
            for name, param in self.config.model.named_parameters():
                if name == "precontext":
                    param.requires_grad = True
        if self.config.use_coop_vision:
            print("Optimize vision prompt")
            for name, param in self.config.model.named_parameters():
                if name.startswith("visual.precontext_vision"):
                    param.requires_grad = True

        # print the unfreezed layers
        for name, param in self.config.model.named_parameters():
            if param.requires_grad:
                print(name)

        if len(self.config.target_layers_vision) != 0:
            print("Optimize visual proj")
            self.config.model.visual.proj.requires_grad = True
        if len(self.config.target_layers_text) != 0 or self.config.train_text_ln_final:
            print("Optimize ln_final")
            self.config.model.ln_final.requires_grad = True

        loss_img = nn.BCEWithLogitsLoss()
        loss_txt = nn.BCEWithLogitsLoss()

        # Params used from paper, the lr is smaller, more safe for fine-tuning to
        # new dataset
        if self.config.use_coop_text or self.config.use_coop_vision:
            # optimizer for config.model.precontext
            coop_optimizer = optim.Adam(
                self.config.model.coop_parameters(),
                lr=self.config.coop_lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.2,
            )
            coop_schedular = optim.lr_scheduler.LinearLR(
                coop_optimizer,
                start_factor=1,
                end_factor=self.config.lr_schedular_end_factor,
                total_iters=self.config.EPOCH * len(train_dataloader),
            )
        optimizer = optim.Adam(
            self.config.model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2,
        )
        schedular = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=self.config.lr_schedular_end_factor,
            total_iters=self.config.EPOCH * len(train_dataloader),
        )

        # ---- S2/S3 辅助模块（独立优化器，不进入核心模型，不影响 strict 加载）----
        aux_modules = []
        attr_heads = None
        cs_projector = None
        anchor_feats = None
        n_attr = len(tmp_inclusive_attributes)
        if self.config.use_attr_heads:
            attr_heads = AttributeHeads(512, n_attr).to(device)
            aux_modules.append(attr_heads)
        if self.config.use_content_style:
            cs_projector = ContentStyleProjector(512).to(device)
            aux_modules.append(cs_projector)
        if aux_modules:
            if device != "cpu":
                for m in aux_modules:
                    m.half()
            aux_optimizer = optim.Adam(
                [p for m in aux_modules for p in m.parameters()],
                lr=self.config.lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.2,
            )
        else:
            aux_optimizer = None

        # 模块 D：用标准预处理（无随机增强）编码每个字体，缓存锚点特征
        if self.config.use_structure_anchor:
            assert cs_projector is not None, "module D requires module C"
            from PIL import Image
            anchor_feats = []
            for fname in dataset.font_names:
                p = os.path.join(self.config.image_file_dir, fname + ".png")
                x = preprocess(Image.open(p)).unsqueeze(0).to(device)
                with torch.no_grad():
                    anchor_feats.append(self.config.model.encode_image(x))
            anchor_feats = torch.cat(anchor_feats, dim=0)

        self.config.model.train()
        best_val_corr = None
        best_epoch = 1
        for epoch in range(self.config.EPOCH):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                # zero the parameter gradients
                optimizer.zero_grad()
                if self.config.use_coop_text or self.config.use_coop_vision:
                    coop_optimizer.zero_grad()

                # unpack batch
                images, texts, font_indices, attribute_indices = batch
                font_indices = font_indices.to(device)
                attribute_indices = attribute_indices.to(device)
                # calculate the mask matrix for the weight of BCELoss
                # CPU 上模型为 float32，mask 须同类型；GPU 保持 float16
                mask_dtype = torch.float32 if device == "cpu" else torch.float16
                mask_matrix = dataset.mask_font_idx_signed_attribute_matrix_ground_truth_fast(
                    font_indices, attribute_indices
                ).to(device).to(mask_dtype)

                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = self.config.model(
                    images, texts
                )
                total_loss = (
                    loss_img(logits_per_image, mask_matrix)
                    + loss_txt(logits_per_text, mask_matrix.T)
                ) / 2
                if self.config.use_orth_loss:
                    orth_loss = self.build_orth_loss(tmp_inclusive_attributes)
                    total_loss = total_loss + orth_loss

                # ---- S2/S3 辅助损失（不 detach，梯度同时作用于辅助模块与共享编码器）----
                if aux_optimizer is not None:
                    aux_optimizer.zero_grad()
                    image_features = self.config.model.encode_image(images)
                    image_features_norm = image_features / image_features.norm(
                        dim=1, keepdim=True
                    )

                    if attr_heads is not None:
                        head_logits = attr_heads(image_features_norm)
                        targets = self.build_attr_targets(
                            dataset, font_indices, tmp_inclusive_attributes
                        ).to(head_logits.dtype)
                        total_loss = total_loss + self.config.attr_head_weight * loss_img(
                            head_logits, targets
                        )

                    if cs_projector is not None:
                        content_feat, style_feat = cs_projector.split(image_features_norm)
                        cs_loss = (
                            self.config.cs_style_weight
                            * style_contrast_loss(style_feat, font_indices)
                            + self.config.cs_orth_weight
                            * subspace_orth_loss(cs_projector)
                            + self.config.cs_recon_weight
                            * reconstruction_loss(cs_projector, image_features_norm)
                        )
                        total_loss = total_loss + cs_loss

                        if anchor_feats is not None:
                            anchor = anchor_feats[font_indices]
                            anchor_norm = anchor / anchor.norm(dim=1, keepdim=True)
                            anchor_content = cs_projector.content_proj(anchor_norm)
                            total_loss = total_loss + self.config.anchor_weight * (
                                content_anchor_loss(content_feat, anchor_content)
                            )

                total_loss.backward()

                if self.config.do_optimize:
                    if device != "cpu":
                        convert_models_to_fp32(self.config.model)
                    optimizer.step()
                    schedular.step()
                    if aux_optimizer is not None:
                        aux_optimizer.step()
                    if self.config.use_coop_text or self.config.use_coop_vision:
                        coop_optimizer.step()
                        coop_schedular.step()
                    if device != "cpu":
                        convert_weights(self.config.model)

                # free images, texts
                images = images.to("cpu")
                texts = texts.to("cpu")
                total_loss = total_loss.to("cpu")
                torch.cuda.empty_cache()

                epoch_loss += total_loss.item()

            if (epoch + 1) % 1 == 0:
                self.config.model.eval()
                val_corr = evaluate_correlation_coefficient(
                    self.config.model,
                    tmp_validation_json_path,
                    val_image_dataset,
                    text_dataset,
                )
                print(
                    f"EPOCH: {epoch+1}, loss: {epoch_loss}, val_correlation: {val_corr}"
                )

                if best_val_corr is None or val_corr > best_val_corr:
                    best_val_corr = val_corr
                    best_epoch = epoch + 1
                    if self.config.do_optimize:
                        print(f"Saving model...{best_epoch}")
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.config.model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": epoch_loss,
                                "signature": self.config.signature,
                            },
                            f"model_checkpoints/{self.config.signature}.pt",
                        )
                        print(
                            f"Saved model...{best_epoch}, {self.config.signature}"
                        )

                if not self.config.do_optimize:
                    break
                self.config.model.train()

        dataset.do_apotosis()
        if val_datasets is not None:
            for val_dataset in val_datasets:
                val_dataset.do_apotosis()

        del dataset
        del val_datasets
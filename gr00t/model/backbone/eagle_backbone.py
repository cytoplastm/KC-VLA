# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }

        del eagle_input["image_sizes"]

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]
    
    # def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
    #     eagle_prefix = "eagle_"
        
    #     # 1. 初步提取：去掉 "eagle_" 前缀
    #     # 此时 keys 可能是 "current_input_ids", "current_pixel_values", "history_pixel_values" 等
    #     eagle_input = {
    #         k.removeprefix(eagle_prefix): v
    #         for k, v in vl_input.items()
    #         if k.startswith(eagle_prefix)
    #     }
    #     # 🟢 [修改] 2. 再次重命名：去掉 "current_" 前缀以适配 HF 模型标准输入
    #     # 模型只接受: input_ids, attention_mask, pixel_values
    #     keys_to_rename = ["current_input_ids", "current_attention_mask", "current_pixel_values"]
    #     for k in keys_to_rename:
    #         if k in eagle_input:
    #             new_key = k.replace("current_", "") # e.g. "input_ids"
    #             eagle_input[new_key] = eagle_input.pop(k)

    #     # 安全删除 image_sizes (如果存在)
    #     eagle_input.pop("image_sizes", None)
        
    #     # 过滤掉不需要传给 Eagle 的参数 (比如 history 相关的数据)
    #     # Eagle 模型不需要知道 history_pixel_values
    #     valid_model_keys = ["input_ids", "attention_mask", "pixel_values", "images", "videos"]
    #     model_input = {k: v for k, v in eagle_input.items() if k in valid_model_keys}

    #     # Forward
    #     eagle_output = self.eagle_model(**model_input, output_hidden_states=True, return_dict=True)
        
    #     eagle_features = eagle_output.hidden_states[self.select_layer]
    #     eagle_features = self.eagle_linear(eagle_features)
        
    #     # 返回特征和 mask (注意 mask 现在的 key 已经是 attention_mask 了)
    #     return eagle_features, eagle_input.get("attention_mask")

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
    
    # 在 EagleBackbone 类内部添加：
    # def forward_vision_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
    #     """
    #     专门处理历史帧：只通过 Vision Encoder 提取特征，不进 LLM。
    #     输出经过 MLP1 投影，维度为 2048 (与 LLM hidden size 对齐)。
    #     """
    #     # 1. Vision Encoder (输出 1152 维)
    #     vision_outputs = self.eagle_model.vision_model(pixel_values)
    #     image_embeds = vision_outputs.last_hidden_state
        
    #     # 2. Projector (1152 -> 2048)
    #     # 这确保了历史帧特征与当前帧特征(来自LLM)处于相同的维度空间
    #     if hasattr(self.eagle_model, "mlp1"):
    #          image_embeds = self.eagle_model.mlp1(image_embeds)
             
    #     return image_embeds

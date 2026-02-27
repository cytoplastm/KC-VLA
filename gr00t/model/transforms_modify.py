import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
from torch.nn.utils.rnn import pad_sequence  # [新增] 用于 input_ids 的 padding

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor


def collate(features: List[dict], eagle_processor) -> dict:
    """
    修改后的 Collate 函数，专门处理拆分后的 eagle_content_split 以及动态历史帧的 Padding 和 Mask 生成。
    """
    batch = {}
    if not features:
        return batch
        
    keys = features[0].keys()

    # 🟢 1. 预先提取原始的 history mask (来自 apply_single, 基于 obs_mask)
    # 这是一个 list of ndarray, 每个形状是 [T_actual]
    # 如果样本没有历史帧，可能是 None 或 空 array
    raw_history_masks = [f.get("eagle_history_mask", None) for f in features]

    for key in keys:
        values = [elem[key] for elem in features]

        # ----------------------------------------------------------------------
        # 🟢 [核心修改] 处理拆分后的 Vision-Language 数据
        # ----------------------------------------------------------------------
        if key == "eagle_content_split":
            # values 是一个 list，每个元素是 {"eagle_current": dict, "eagle_history": dict}
            
            # =========================================================
            # Part A: 处理当前帧 (Current) -> 进完整 VLM
            # =========================================================
            current_list = [v["eagle_current"] for v in values]
            
            # 1.1 处理 input_ids (文本) - 需要 Padding
            if "input_ids" in current_list[0]:
                # squeeze: [1, Seq_Len] -> [Seq_Len]
                input_ids_list = [x["input_ids"].squeeze(0) for x in current_list]
                
                # 获取 pad_token_id
                pad_token_id = eagle_processor.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = 0 # Fallback
                
                # Stack & Pad: [B, Max_Seq_Len]
                padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
                batch["eagle_current_input_ids"] = padded_input_ids

            # 1.2 处理 attention_mask (文本) - 同样需要 Padding
            if "attention_mask" in current_list[0]:
                att_mask_list = [x["attention_mask"].squeeze(0) for x in current_list]
                # mask padding value 为 0
                padded_att_mask = pad_sequence(att_mask_list, batch_first=True, padding_value=0)
                batch["eagle_current_attention_mask"] = padded_att_mask

            # 1.3 处理 pixel_values (图像) - 直接拼接
            # Current 帧通常不需要 Pad，因为大家都是 1 帧 (x V 视角)
            if "pixel_values" in current_list[0]:
                # cat: List of [N_curr, C, H, W] -> [Total_Curr, C, H, W]
                batch["eagle_current_pixel_values"] = torch.cat([x["pixel_values"] for x in current_list], dim=0)

            # =========================================================
            # Part B: 处理历史帧 (History) -> 进 Vision Encoder
            # 难点：动态长度，需要 Pad Pixel Values 并生成对应 Mask
            # =========================================================
            history_list = [v["eagle_history"] for v in values]
            
            # 收集 pixel_values
            hist_tensors = []
            for item in history_list:
                if item and "pixel_values" in item:
                    hist_tensors.append(item["pixel_values"])
                else:
                    hist_tensors.append(None)
            
            # 过滤出有效的 tensor 以计算最大长度
            valid_tensors = [t for t in hist_tensors if t is not None]
            
            if len(valid_tensors) > 0:
                # tensor shape: [T_actual * V, C, H, W]
                # 计算这个 Batch 里最大的 Total Images 数
                max_total_imgs = max([t.shape[0] for t in valid_tensors])
                
                padded_pixel_values = []
                final_masks = [] # List of [T_max * V]
                
                # 遍历 Batch 中的每个样本
                for i, t in enumerate(hist_tensors):
                    # 获取该样本对应的逻辑 Mask [T_actual]
                    raw_mask = raw_history_masks[i] 
                    if raw_mask is None: raw_mask = np.array([], dtype=bool)
                    
                    if t is None:
                        # Case 1: 样本完全没有历史帧
                        # 需要 Pad 全 0 图像
                        C, H, W = valid_tensors[0].shape[-3:] # 借用第一个有效样本的形状
                        padded_pixel_values.append(torch.zeros(max_total_imgs, C, H, W))
                        
                        # Mask 全为 False (无效)
                        final_masks.append(torch.zeros(max_total_imgs, dtype=torch.bool))
                    else:
                        # Case 2: 样本有历史帧，但可能比 max 短
                        curr_len = t.shape[0] # T_actual * V
                        pad_len = max_total_imgs - curr_len
                        
                        # 2.1 Pad 图像
                        if pad_len > 0:
                            # 补黑图
                            padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype, device=t.device)
                            padded_pixel_values.append(torch.cat([t, padding], dim=0))
                        else:
                            padded_pixel_values.append(t)

                        # 2.2 构造 Mask (核心逻辑)
                        # 我们有 raw_mask [T_actual] (来自 Dataset, 1=有效, 0=无效)
                        # 我们需要扩展到 [T_actual * V]
                        
                        # 推断 V (视角数)
                        len_raw = len(raw_mask)
                        if len_raw > 0:
                            V = curr_len // len_raw
                            # 扩展 Mask: [T] -> [T, 1] -> [T, V] -> [T*V]
                            # 这样同一时刻的所有视角共享同一个 mask 值
                            expanded_mask = torch.from_numpy(raw_mask).unsqueeze(1).repeat(1, V).flatten()
                        else:
                            # 异常保护
                            expanded_mask = torch.tensor([], dtype=torch.bool)
                            
                        # 2.3 加上 Collate Pad 产生的 Mask (False)
                        if pad_len > 0:
                            mask_padding = torch.zeros(pad_len, dtype=torch.bool)
                            final_mask = torch.cat([expanded_mask, mask_padding])
                        else:
                            final_mask = expanded_mask
                            
                        final_masks.append(final_mask)
                
                # Stack -> [B, Max_Total, C, H, W]
                batch["eagle_history_pixel_values"] = torch.stack(padded_pixel_values)
                
                # 🟢 [重要] 展平 Batch 维度供 Backbone 使用
                # [B, Max_Total, C, H, W] -> [B * Max_Total, C, H, W]
                b, total, c, h, w = batch["eagle_history_pixel_values"].shape
                batch["eagle_history_pixel_values"] = batch["eagle_history_pixel_values"].view(-1, c, h, w)
                
                # Stack Masks -> [B, Max_Total]
                batch["eagle_history_mask"] = torch.stack(final_masks) 
            
            else:
                # 整个 Batch 都没有历史帧
                batch["eagle_history_pixel_values"] = None
                batch["eagle_history_mask"] = None

        # ----------------------------------------------------------------------
        # 跳过 eagle_history_mask，因为它已经在上面被合并处理了
        # ----------------------------------------------------------------------
        elif key == "eagle_history_mask":
            continue

        # ----------------------------------------------------------------------
        # 处理其他常规 Key
        # ----------------------------------------------------------------------
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            batch[key] = torch.cat(values)
        else:
            # state, action, task_phase ...
            batch[key] = torch.from_numpy(np.stack(values))
            
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        处理逻辑修改：拆分当前帧和历史帧。
        Args:
            batch:
                images: [V, T, C, H, W]
                language: str
        Returns: 
            dict with key "eagle_content_split" containing separate inputs.
        """
        images = batch["images"]  # [V, T, C, H, W]
        V, T, C, H, W = images.shape

        # ------------------------------------------------------------------
        # Part A: 当前帧 (Current) -> [V, 1, C, H, W]
        # ------------------------------------------------------------------
        # 取最后一帧 (T-1)
        # rearrange: [V, C, H, W] -> List of [C, H, W]
        current_images_np = rearrange(images[:, -1], "v c h w -> v c h w")
        
        # 准备 Prompt
        text_content = []
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        # 转换 PIL
        eagle_images_curr = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in current_images_np]
        
        # 构造对话: Image + Text
        eagle_msg_imgs = [{"type": "image", "image": img} for img in eagle_images_curr]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_msg_imgs + text_content,
            }
        ]

        # VLM Processor (Text + Image) -> 生成 input_ids 和 pixel_values
        inputs_curr = self.eagle_processor(
            text=[self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )],
            images=eagle_images_curr,
            return_tensors="pt",
            padding=True # 这里 padding 只是针对单条文本（如果有多个turn），Collate 里还需要 pad batch
        )

        # ------------------------------------------------------------------
        # Part B: 历史帧 (History) -> [V, T-1, C, H, W]
        # ------------------------------------------------------------------
        if T > 1:
            # 取前 T-1 帧
            # rearrange: [V, T-1, C, H, W] -> Flatten to [(V * T-1), C, H, W]
            # 这样所有的历史视角的每一帧都变成了独立的图片
            history_images_np = rearrange(images[:, :-1], "v t c h w -> (t v) c h w")
            
            eagle_images_hist = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in history_images_np]
            
            # 仅 Image Processor (只生成 pixel_values)
            inputs_hist = self.eagle_processor.image_processor(
                images=eagle_images_hist, 
                return_tensors="pt"
            )
        else:
            inputs_hist = {} # 无历史帧

        # ------------------------------------------------------------------
        # Part C: 打包返回
        # ------------------------------------------------------------------
        inputs = {}
        inputs["eagle_current"] = inputs_curr
        inputs["eagle_history"] = inputs_hist
        
        # 使用一个新的 key 包裹，方便在 collate 里识别
        return {"eagle_content_split": inputs}

    # 🟢 [关键修改] _prepare_video: 保持定长，不筛选
    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""    
        # obs_mask = data.get("obs_mask", None)  # [T_obs]

        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",  # [v, t, c, h, w]
        )
        
        # ❌ 不要在这里剔除无效帧！
        # if obs_mask is not None:
        #     obs_mask = np.asarray(obs_mask, dtype=bool) 
        #     valid_frames = np.nonzero(obs_mask)[0] 
        #     images = images[:, valid_frames, ...] 

        return images
    
    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        使用 data['obs_mask'] 把 padding 出来的时间步 state 置 0，并在 state_mask 里屏蔽掉。
        Return (state, state_mask, n_state_tokens).
        """
        # 没有 state 的情况（比如纯视觉策略）
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim), dtype=np.float32)
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]                     # [T, D_state]，这里 T 应该等于 state_horizon
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        # === 1) 读取 obs_mask（如果有的话） ===
        obs_mask = data.get("obs_mask", None)
        if obs_mask is not None:
            obs_mask = np.asarray(obs_mask)
            # 允许 int/bool 一律转成 bool
            if obs_mask.dtype != bool:
                obs_mask = obs_mask.astype(bool)
            assert (
                obs_mask.shape[0] == state.shape[0]
            ), f"obs_mask shape {obs_mask.shape} doesn't match state {state.shape}"

        # === 2) 先对通道做裁剪 / padding 到 max_state_dim ===
        n_state_dims = state.shape[-1]

        if n_state_dims > self.max_state_dim:
            # 维度太大就截断前 max_state_dim 维
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # 维度不够，就在通道维右侧 pad 0
            state = np.pad(
                state,
                pad_width=((0, 0), (0, self.max_state_dim - n_state_dims)),
                mode="constant",
            )

        # === 3) 构建 state_mask：对“真实通道”标 True ===
        state_mask = np.zeros_like(state, dtype=bool)  # [T, max_state_dim]
        state_mask[:, :n_state_dims] = True

        # === 4) 如果有 obs_mask，把时间维上无效帧全置 0 + mask 置 False ===
        if obs_mask is not None:
            # obs_mask: [T]，True 表示有效时间步
            valid = obs_mask  # bool[T]

            # 无效时间步整行归 0
            state[~valid, :] = 0.0
            state_mask[~valid, :] = False

        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens
 
    def apply_single(self, data: dict) -> dict:
            transformed_data = {}

            # 1) Prepare video and language with vlm processing.
            # images shape: [V, T, C, H, W]
            images = self._prepare_video(data)        
            images = images.astype(np.uint8)
            language = self._prepare_language(data)

            # ----------------------------------------------------------------------
            # 🟢 [修改] 处理 obs_mask 并生成 history_mask
            # ----------------------------------------------------------------------
            obs_mask = data.get("obs_mask", None) 
            eagle_history_mask = None # 初始化

            if obs_mask is not None:
                # 确保是 numpy bool 向量
                obs_mask = np.asarray(obs_mask, dtype=bool) # [T_total]
                
                # 💡 关键逻辑：
                # obs_mask 包含了 [历史帧..., 当前帧]
                # 我们需要的 history_mask 只需要 [历史帧...]
                # 所以取 obs_mask[:-1]
                if len(obs_mask) > 1:
                    eagle_history_mask = obs_mask[:-1] # [T_hist]
                else:
                    # 只有 1 帧 (当前帧)，没有历史
                    eagle_history_mask = np.array([], dtype=bool)

            # ----------------------------------------------------------------------
            # 透传 task_phase
            # ----------------------------------------------------------------------
            if "task_phase" in data:
                transformed_data["task_phase"] = np.array(data["task_phase"], dtype=np.int64)

            # 透传 keyframe_counts
            if "keyframe_counts" in data:
                transformed_data["keyframe_counts"] = data["keyframe_counts"]

            # 构造 batch_data 供 VLM 处理
            batch_data = {"images": images, "language": language}
            
            # 调用新的 vlm processing
            vlm_outputs = self._apply_vlm_processing(batch_data)

            # 2) Prepare state
            state, state_mask, _ = self._prepare_state(data)
            transformed_data["state"] = state
            transformed_data["state_mask"] = state_mask

            if self.training:
                transformed_data["segmentation_target"] = np.zeros((2,))
                transformed_data["segmentation_target_mask"] = np.zeros((1,))
                transformed_data["has_real_action"] = np.ones((), dtype=bool)
                actions, actions_mask, _ = self._prepare_action(data)
                transformed_data["action"] = actions
                transformed_data["action_mask"] = actions_mask

            if obs_mask is not None:
                transformed_data["obs_mask"] = obs_mask # 保留原始 mask 给 Action Head 用

            # 🟢 [新增] 将切分好的历史 mask 放入 transformed_data
            # 这样 Collate 就能像处理 "action" 一样自动把它 stack 起来
            if eagle_history_mask is not None:
                transformed_data["eagle_history_mask"] = eagle_history_mask

            for k, v in vlm_outputs.items():
                assert k not in transformed_data, f"Key {k} already exists in transformed_data."
                transformed_data[k] = v

            transformed_data["embodiment_id"] = self.get_embodiment_tag()

            if self.training:
                action_and_mask_keys = ["action", "action_mask"]
                assert all(
                    transformed_data[key].shape == transformed_data["action"].shape
                    for key in action_and_mask_keys
                ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

            return transformed_data
        
    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
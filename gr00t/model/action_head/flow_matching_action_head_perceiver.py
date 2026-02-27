from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish
# 引入你刚才新建的 Perceiver
from gr00t.model.perceiver import PerceiverResampler 
from .cross_attention_dit import DiT, SelfAttentionTransformer


# ... (CategorySpecificLinear, CategorySpecificMLP, MultiEmbodimentActionEncoder 保持不变，省略以节省空间) ...
# 请保留原本的这些类定义
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,).")
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    model_dtype: str = field(default="float32")
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=1536)
    backbone_embedding_dim: int = field(default=1536)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=None)
    max_num_embodiments: int = field(default=32)
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)
    load_pretrained_det_decode_layer_path: str = field(default=None)
    detection_coeff: float = field(default=1.0)
    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32)
    
    # === ⭐ 新增 Perceiver 配置 ===
    use_perceiver: bool = field(default=True, metadata={"help": "Enable Perceiver Resampler"})
    perceiver_num_latents: int = field(default=32, metadata={"help": "Output token count (compressed)"})
    perceiver_depth: int = field(default=2, metadata={"help": "Depth of Perceiver"})
    perceiver_heads: int = field(default=8, metadata={"help": "Heads of Perceiver"})
    perceiver_dim_head: int = field(default=64, metadata={"help": "Dim head of Perceiver"})
    # ============================

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FlowmatchingActionHeadConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # Encoders / Decoders
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        
        # Future / Query Tokens
        # 如果启用了 Perceiver，这里的 tokens 可以作为额外的查询，或者直接用 Perceiver 的输出
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # VLLN & VLSA (原有逻辑)
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        # === ⭐ 初始化 Perceiver ===
        if config.use_perceiver:
            self.perceiver = PerceiverResampler(
                dim=config.backbone_embedding_dim,
                depth=config.perceiver_depth,
                dim_head=config.perceiver_dim_head,
                heads=config.perceiver_heads,
                num_latents=config.perceiver_num_latents,
                max_tokens=4096, # 最大可能的输入长度
                ff_mult=4,
                activation='gelu',
                trainable=config.tune_projector
            )
            print(f"✅ Perceiver Resampler Initialized: {config.perceiver_num_latents} latents.")
        else:
            self.perceiver = None
        # ==========================

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            # 如果不微调 projector，Perceiver 也不要动
            if self.perceiver is not None:
                for param in self.perceiver.parameters():
                    param.requires_grad = False
        
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
            
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
                if self.perceiver is not None:
                    self.perceiver.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def process_backbone_output(self, backbone_output: BatchFeature, attention_mask: Optional[torch.Tensor] = None) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        # ================== 🔴 修复开始 ==================
        # 强制将 mask 转换为 Bool 类型 (Fix for RuntimeError: long int mask)
        # 原始 mask: 1=Valid, 0=Padding (Long/Int)
        # 转换后: True=Valid, False=Padding (Bool)
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = (attention_mask > 0).to(torch.bool)
        # ================== 🔴 修复结束 ==================
        
        # 1. 基础归一化 (VLLN)
        backbone_features = self.vlln(backbone_features)

        # 2. 基础 Self-Attention (可选，作为前置处理)
        # 注意：这里的 attention_mask 是 (1=Valid, 0=Padding)
        backbone_features = self.vl_self_attention(backbone_features, attention_mask=attention_mask)

        # === ⭐ 3. Perceiver Resampling (压缩变长序列) ===
        if self.perceiver is not None:
            # Perceiver 需要的 mask 格式：
            # 如果你用的是刚才提供的代码：key_padding_mask=True 表示 Padding (会被忽略), False 表示 Valid
            # 这里的 attention_mask 来自 HF, 1=Valid, 0=Padding
            # 所以我们需要取反： (attention_mask == 0) -> True(Padding)
            
            # perceiver_mask = (attention_mask == 0) # [B, N]q
            perceiver_mask = attention_mask.bool()
            
            # 输入: [B, N_var, D], Mask: [B, N_var]
            # 输出: [B, 32, D]
            compressed_features = self.perceiver(backbone_features, key_padding_mask=perceiver_mask)
            
            # 更新 features
            backbone_output["backbone_features"] = compressed_features
            
            # 更新 mask: 压缩后的 mask 全为 1 (都是有效的 latents)
            B, N_latents, _ = compressed_features.shape
            new_mask = torch.ones((B, N_latents), dtype=attention_mask.dtype, device=attention_mask.device)
            backbone_output["backbone_attention_mask"] = new_mask
        # ===============================================
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:  
        self.set_frozen_modules_to_eval_mode()

        # 1. 获取原始 Mask (1=Valid, 0=Pad)
        vl_attn_mask_original = backbone_output.backbone_attention_mask 
        
        # 2. 处理 Backbone 输出 (VLLN -> VLSA -> Perceiver)
        # 这里传入 original mask 给 SA 和 Perceiver 做参考
        backbone_output = self.process_backbone_output(
            backbone_output, 
            attention_mask=vl_attn_mask_original
        )

        # 3. 后续逻辑基本不变，只是现在的 backbone_features 长度固定为 32 了
        # ... (expand_batch 逻辑保持不变) ...
        if self.config.expand_batch is not None:
             for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch] + [1] * (ndim - 1)
                backbone_output[k] = v.repeat(*factors)
             for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch] + [1] * (ndim - 1)
                action_input[k] = v.repeat(*factors)

        vl_embs = backbone_output.backbone_features # [B, 32, D] (Perceiver Output)
        vl_attn_mask = backbone_output.backbone_attention_mask # [B, 32] (All 1s)

        # ... (State/Action encoding 逻辑保持不变) ...
        B = vl_embs.shape[0]
        embodiment_id = action_input.embodiment_id
        state = action_input.state
        state_mask = action_input.state_mask
        actions = action_input.action
        action_mask = action_input.action_mask
        
        state_features = self.state_encoder(state, embodiment_id)
        
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1.0 - t) * noise + t * actions
        velocity = actions - noise 
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0) 
            action_features = action_features + pos_embs

        # 拼接 Embeddings
        # 这里你可以选择保留 future_tokens 作为额外的 query，或者直接认为 vl_embs 就是 future knowledge
        # 为了兼容性，我们依然保留 future_tokens，把它和 Perceiver 输出拼在一起
        T_future = self.config.num_target_vision_tokens
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        
        sa_embs = torch.cat(
            [state_features, future_tokens, action_features],
            dim=1,
        ) 

        # 构造 Mask
        if state_mask.ndim == 3:
            state_time_mask = (state_mask.abs().sum(dim=-1) > 0).to(torch.float32)
        else:
            state_time_mask = state_mask.to(torch.float32)
        
        future_time_mask = torch.ones((B, T_future), device=self.device, dtype=torch.float32)

        if action_mask.ndim == 3:
            action_time_mask = (action_mask.abs().sum(dim=-1) > 0).to(torch.float32) 
        else:
            action_time_mask = action_mask.to(torch.float32)

        sa_mask = torch.cat([state_time_mask, future_time_mask, action_time_mask], dim=1)
        sa_bool_mask = (sa_mask > 0)
        
        # DiT Cross-Attention Mask
        vl_bool_mask = (vl_attn_mask > 0) # [B, 32] All True

        # === 🚀 新增：维度检查打印 ===
        if not hasattr(self, '_shapes_checked'):
            print("\n" + "·"*20 + " [Shape Check] " + "·"*20)
            print(f"sa_embs (Query) Shape:        {sa_embs.shape}")        # 预期: [B, T_state + T_future + T_action, D]
            print(f"sa_bool_mask Shape:           {sa_bool_mask.shape}")   # 预期: [B, T_total_sa]
            print("-" * 55)
            print(f"vl_embs (Encoder/Key) Shape:  {vl_embs.shape}")        # 预期: [B, 32, D] (Perceiver 压缩后)
            print(f"vl_bool_mask Shape:           {vl_bool_mask.shape}")   # 预期: [B, 32]
            print("·"*55 + "\n")
            self._shapes_checked = True 
        # ==================================

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs, # 这里是 Perceiver 压缩后的 features
            attention_mask=sa_bool_mask,
            encoder_attention_mask=vl_bool_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )

        pred = self.action_decoder(model_output, embodiment_id) 
        pred_actions = pred[:, -actions.shape[1] :] 
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        return BatchFeature(data={"loss": loss})

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # 推理时也要走同样的 Perceiver 流程
        # process_backbone_output 里面已经处理了 mask 转换
        backbone_output = self.process_backbone_output(
            backbone_output, 
            attention_mask=backbone_output.backbone_attention_mask
        )

        vl_embs = backbone_output.backbone_features # [B, 32, D]
        # ... 后续 get_action 逻辑保持不变，DiT 会自动 adapt 这个长度 ...
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(size=(batch_size, self.config.action_horizon, self.config.action_dim), dtype=vl_embs.dtype, device=device)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs, # Perceiver output
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity
            
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
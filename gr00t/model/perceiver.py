import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn

class SquaredReLU(nn.Module):
    """Squared ReLU activation function"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

def feed_forward_layer(dim: int, mult: int = 4, activation: str = 'gelu'):
    """Feed forward layer with given activation function"""
    activations = dict(gelu=nn.GELU, sqrelu=SquaredReLU, relu=nn.ReLU)
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        activations[activation](),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PerceiverAttentionLayer(nn.Module):
    """Standard Perceiver Attention Layer using SDPA"""
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents, key_attention_mask=None):
            n_heads = self.heads
            n_batch, n_queries = latents.shape[0], latents.shape[1]

            x = self.norm_media(features)
            latents = self.norm_latents(latents)

            q = rearrange(self.to_q(latents), 'b n (h d) -> b h n d', h=n_heads)

            # 这里把 features 和 latents 拼接作为 KV
            kv_input = torch.cat((x, latents), dim=-2)
            k = self.to_k(kv_input)
            v = self.to_v(kv_input)
            k, v = rearrange_many((k, v), 'b n (h d) -> b h n d', h=n_heads)

            # --- 重点：不再在内部进行拼接和维度扩展 ---
            # 直接使用外部传进来的 full_attn_mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=key_attention_mask, # 外部已经处理好了
                dropout_p=0.0,
                is_causal=False
            )

            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

class PerceiverResampler(nn.Module):
    """Perceiver Resampler module"""
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        max_tokens: int = 4096,
        ff_mult: int = 4,
        activation: str = 'gelu',
        trainable: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(max_tokens, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                    feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                ])
            )
        self.norm = nn.LayerNorm(dim)

        # # 默认使用 init_weights，但会被外部脚本的手动重置覆盖
        # self.apply(self._init_weights)
        
        for param in self.parameters():
            param.requires_grad = trainable

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x_f: torch.Tensor, key_padding_mask: torch.BoolTensor = None):
            batch_size, n_feat, _ = x_f.shape

            # 1. 确保原始有效位 Mask 是 2D [B, N_feat]
            valid_mask = key_padding_mask.bool() 
            if valid_mask.dim() == 4: # 如果传进来已经是 4D，先降回 2D 方便拼接
                valid_mask = valid_mask[:, 0, 0, :]

            # ================== 🚀 新增：倒序位置编码注入 ==================
            # 核心逻辑：无论输入 token 多少，最后一帧永远对应 self.time_pos_emb 的末尾索引
            max_pos_len = self.time_pos_emb.shape[0] # 4096

            # 生成 [-n_feat, ..., -1] 的相对索引
            # 例如推理时 n_feat=1211, 则索引为 [-1211, ..., -1]
            rel_idx = torch.arange(-n_feat, 0, device=x_f.device)
            # 转换为绝对索引：[4096-1211, ..., 4095]
            # 这样保证了最后一个 token 加的是 time_pos_emb[4095]
            abs_idx = max_pos_len + rel_idx
            # 安全截断，防止溢出
            abs_idx = abs_idx.clamp(min=0, max=max_pos_len - 1)
            # 提取对应的位置编码 [n_feat, dim]
            pos_emb = self.time_pos_emb[abs_idx] # (n_feat, dim)
            
            # 将位置编码叠加到原始特征上
            # 如果训练时有 padding，则 pos_emb 会自动对齐到有效位
            x_f = x_f + pos_emb.unsqueeze(0) 
            # ============================================================

            # 2. 构造 Latents 部分的 Mask [B, N_latents]
            n_latents = self.latents.shape[0]
            latents_mask = torch.ones((batch_size, n_latents), dtype=torch.bool, device=x_f.device)
            
            # 3. 拼接并扩展为 4D [B, 1, 1, Total_Seq_Len]
            # Total_Seq_Len = n_feat + n_latents
            full_attn_mask = torch.cat([valid_mask, latents_mask], dim=1) 
            full_attn_mask = full_attn_mask.unsqueeze(1).unsqueeze(1) 
            # # ================== 🚀 增强版：可视化掩码分布 (带计数统计) ==================
            # if not hasattr(self, '_mask_distribution_checked'):
            #     print("\n" + " 掩码空间分布与计数监控 (1=有效, 0=填充) " + "-"*20)
            #     n_latents = 32 # 你的 Latents 数量
                
            #     for i in range(batch_size):
            #         # 获取当前样本的 Mask [N]
            #         m = full_attn_mask[i, 0, 0].int()
            #         total_len = len(m)
            #         valid_count = torch.sum(m).item()
                    
            #         # 统计视觉部分（扣除末尾 Latents）
            #         visual_mask = m[:-n_latents]
            #         visual_valid_count = torch.sum(visual_mask).item()
            #         visual_total_len = len(visual_mask)

            #         # 1. 抽取 60 个点代表分布预览
            #         indices = torch.linspace(0, total_len - 1, steps=60).long()
            #         viz_seq = "".join(["█" if m[idx] == 1 else "░" for idx in indices])
                    
            #         # 2. Latents 部分预览
            #         latents_mask = m[-n_latents:]
            #         latents_viz = "".join(["█" if x == 1 else "░" for x in latents_mask])
            #         latents_valid = torch.sum(latents_mask).item()
                    
            #         # 3. 打印统计信息
            #         print(f"样本 [{i:02d}] 统计: 总位={total_len} | 有效位={valid_count} | 视觉有效={visual_valid_count}/{visual_total_len}")
            #         print(f"         分布预览: |{viz_seq}| (█=1, ░=0)")
            #         print(f"         Latents : [{latents_viz}] ({latents_valid}/{n_latents} 有效)")
                    
            #         # 极性自检：如果视觉位全是 0，可能极性写反了
            #         if visual_valid_count == 0 and visual_total_len > 0:
            #             print(f"⚠️  警告: 样本 [{i:02d}] 视觉部分有效位为 0，请检查 Mask 极性！")
                
            #     print("-" * 75 + "\n")
            #     self._mask_distribution_checked = True
            # # ===========================================================================

            # ================== 🚀 动态增强版：掩码分布与长度变化监控 ==================
            # 获取当前总长度 (N_tokens + 32_latents)
            current_total_len = full_attn_mask.shape[-1]
            
            # 检查是否是第一次运行，或者 Token 总数发生了变化
            should_print = False
            if not hasattr(self, '_last_recorded_mask_len'):
                should_print = True
            elif self._last_recorded_mask_len != current_total_len:
                print(f"\n📢 [检测到输入长度变化]: {self._last_recorded_mask_len} -> {current_total_len}")
                should_print = True

            if should_print:
                print("\n" + " 掩码空间分布与计数监控 (1=有效, 0=填充) " + "-"*20)
                n_latents = 32
                
                for i in range(batch_size):
                    m = full_attn_mask[i, 0, 0].int()
                    total_len = len(m)
                    valid_count = torch.sum(m).item()
                    
                    # 统计视觉部分
                    visual_mask = m[:-n_latents]
                    visual_valid_count = torch.sum(visual_mask).item()
                    visual_total_len = len(visual_mask)

                    # 1. 抽取 60 个点代表分布预览
                    indices = torch.linspace(0, total_len - 1, steps=60).long()
                    viz_seq = "".join(["█" if m[idx] == 1 else "░" for idx in indices])
                    
                    # 2. Latents 部分预览
                    latents_mask = m[-n_latents:]
                    latents_viz = "".join(["█" if x == 1 else "░" for x in latents_mask])
                    latents_valid = torch.sum(latents_mask).item()
                    
                    # 3. 打印统计信息
                    # 这里会清晰看到：图像 Token 数 = visual_total_len
                    print(f"样本 [{i:02d}] 统计: 总位={total_len} | 有效位={valid_count} (视觉={visual_valid_count} + Latents={latents_valid})")
                    print(f"         视觉结构: 图像特征占 {visual_total_len} 位")
                    print(f"         分布预览: |{viz_seq}| (█=1, ░=0)")
                    print(f"         Latents : [{latents_viz}] ({latents_valid}/{n_latents} 有效)")
                    
                    if visual_valid_count == 0 and visual_total_len > 0:
                        print(f"⚠️  警告: 样本 [{i:02d}] 视觉部分有效位为 0，请检查 Mask 极性！")
                
                print("-" * 75 + "\n")
                # 更新记录的长度，防止重复打印
                self._last_recorded_mask_len = current_total_len
            # ===========================================================================

            x = repeat(self.latents, 'n d -> b n d', b=batch_size)

            # 运行 Attention 循环
            for attn, ffw in self.layers:
                # 这里的 full_attn_mask 已经是标准的 4D [B, 1, 1, N]
                x = x + attn(x_f, x, full_attn_mask)
                x = x + ffw(x)

            return self.norm(x)
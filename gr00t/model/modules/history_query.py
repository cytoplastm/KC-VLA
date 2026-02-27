import torch
import torch.nn as nn

class HistoryQueryModule(nn.Module):
    def __init__(self, input_dim, query_dim, num_queries=32, num_phases=10, 
                 max_history_len=50, num_views=2): 
        super().__init__()
        
        # 🛡️ [安全检查] 确保维度对齐，否则 Attention 会报错
        assert input_dim == query_dim, f"Input dim {input_dim} must match query dim {query_dim} for standard MHA"
        
        self.num_queries = num_queries
        
        # 1. 基础侦探 Token
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, query_dim))
        
        # 2. 阶段意图
        self.phase_embedding = nn.Embedding(num_phases, query_dim)
        
        # 3. 双重位置编码
        self.history_time_embedding = nn.Embedding(max_history_len, input_dim)
        self.history_view_embedding = nn.Embedding(num_views, input_dim)
        
        # 4. Cross-Attention 组件
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        
        # 🟢 [新增] 5. 归一化与Dropout (Pre-Norm 结构，训练更稳)
        self.input_norm = nn.LayerNorm(input_dim) # 用于 history features
        self.query_norm = nn.LayerNorm(query_dim) # 用于 query tokens
        self.output_norm = nn.LayerNorm(input_dim) # 用于残差连接后
        self.dropout = nn.Dropout(0.1)

    def forward(self, history_feats, phase_ids, history_mask=None):
        """
        Args:
            history_feats: [B, T, V, N, D]
            phase_ids: [B]
            history_mask: [B, T] (Boolean Tensor) 🟢 [新增]
                          True 表示该帧是 Padding (无效)，False 表示有效。
                          (注意 PyTorch MHA 的 key_padding_mask 定义：True 为忽略)
        """
        B, T, V, N, D = history_feats.shape
        device = history_feats.device
        
        # --- A. 位置编码注入 (保持你原来的优秀逻辑) ---
        time_ids = torch.arange(T, device=device).reshape(1, T, 1, 1)
        view_ids = torch.arange(V, device=device).reshape(1, 1, V, 1)
        
        # [B, T, V, N, D]
        x = history_feats + \
            self.history_time_embedding(time_ids) + \
            self.history_view_embedding(view_ids)
        
        # 展平 T, V, N -> [B, S, D]  (S = T*V*N)
        key_value_input = x.flatten(1, 3)
        
        # 🟢 [新增] Pre-Norm: 进 Attention 前先 Norm，这是目前 LLM/ViT 的主流做法
        key_value_input = self.input_norm(key_value_input)

        # --- B. Query 准备 ---
        queries = self.query_tokens.repeat(B, 1, 1)
        if phase_ids is not None:
            queries = queries + self.phase_embedding(phase_ids).unsqueeze(1)
        
        # 🟢 [新增] Pre-Norm
        queries_norm = self.query_norm(queries)

        # --- C. Mask 处理 (至关重要) ---
        key_padding_mask = None
        if history_mask is not None:
            # history_mask 只有 [B, T]，我们需要把它扩展到 [B, T*V*N]
            # 逻辑：如果第 t 帧是 padding，那么第 t 帧下的所有 View 和 Patch (N) 都是 padding
            
            # [B, T] -> [B, T, 1, 1] -> [B, T, V, N]
            mask_expanded = history_mask.reshape(B, T, 1, 1).expand(-1, -1, V, N)
            # 展平 -> [B, T*V*N]
            key_padding_mask = mask_expanded.flatten(1, 3)
            
            # ⚠️ 确保 mask 是布尔值 (PyTorch MHA要求：True表示被Mask/无效)
            key_padding_mask = key_padding_mask.bool()

        # --- D. Attention ---
        attn_out, _ = self.cross_attn(
            query=queries_norm, 
            key=key_value_input, 
            value=key_value_input, 
            key_padding_mask=key_padding_mask
        )
        
        # 🟢 [新增] E. 残差连接 + Post-Dropout
        # Query 本身携带了侦探意图，Attention 的结果是查到的信息，加回去
        out = queries + self.dropout(attn_out)
        
        # 🟢 [新增] F. 最终 Norm (可选，但在深层网络中推荐)
        out = self.output_norm(out)
        
        return out
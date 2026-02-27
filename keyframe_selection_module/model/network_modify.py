import torch
import torch.nn as nn
import torchvision.models as models

class TransformerKeyframeSelector(nn.Module):
    def __init__(self, 
                 pretrained_backbone_path=None, 
                 use_imagenet=False, 
                 use_film=True,  # 🟢 [实验开关] True: 使用FiLM调制; False: 使用简单拼接
                 num_tasks=5, 
                 max_phases=100, 
                 window_size=3, 
                 embed_dim=128, 
                 num_heads=4):
        super().__init__()
        
        self.use_film = use_film
        
        # --- 1. Backbone 设置 (保持不变) ---
        if use_imagenet:
            print("🧊 Mode: Using Standard ImageNet Pretrained ResNet18")
            resnet = models.resnet18(weights='IMAGENET1K_V1') 
        else:
            resnet = models.resnet18(weights=None) 
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if pretrained_backbone_path:
            print(f"📥 Loading Stage 1 Custom Backbone from: {pretrained_backbone_path}")
            state_dict = torch.load(pretrained_backbone_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)
            
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("❄️ Backbone Frozen")
            
        # --- 2. 核心组件 ---
        self.vis_projector = nn.Linear(512, embed_dim)
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        self.phase_embedding = nn.Embedding(max_phases, embed_dim)
        
        # 🟢 [差异化初始化] 
        if self.use_film:
            # FiLM 模式：生成缩放(gamma)和偏移(beta)因子
            print("🔧 Query Mode: Task-Modulated FiLM (Ours)")
            self.film_generator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 2, embed_dim * 2)
            )
        else:
            # Concat 模式：简单的拼接 + 线性投影融合
            print("🔧 Query Mode: Simple Concatenation (Baseline)")
            # 输入维度是 embed_dim * 2 (task + phase)，输出映射回 embed_dim
            self.cat_projector = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True)
            )
        
        self.time_embedding = nn.Parameter(torch.randn(1, window_size, embed_dim) * 0.02)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), 
            nn.Linear(128, 1)
        )

    def forward(self, curr_window_imgs, phase_id, task_id):
        # 1. 视觉特征提取 (保持不变)
        B, T, C, H, W = curr_window_imgs.shape
        imgs_flat = curr_window_imgs.view(-1, C, H, W)
        
        with torch.no_grad():
            vis_feat_raw = self.backbone(imgs_flat).flatten(1)
            
        vis_feat = self.vis_projector(vis_feat_raw)
        vis_seq = vis_feat.view(B, T, -1) + self.time_embedding
        
        attn_out1, _ = self.self_attn(vis_seq, vis_seq, vis_seq)
        vis_contextual = self.norm1(vis_seq + attn_out1)
        
        # 2. 构造 Logic Query (核心差异点)
        t_emb = self.task_embedding(task_id) # [B, dim]
        p_emb = self.phase_embedding(phase_id) # [B, dim]
        
        if self.use_film:
            # --- 方案 A: FiLM 调制 (Ours) ---
            # Task 生成调制参数，去调节 Phase
            gamma_beta = self.film_generator(t_emb)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            
            # Logic Query = gamma * Phase + beta
            logic_query = (gamma * p_emb + beta).unsqueeze(1) # [B, 1, dim]
            
        else:
            # --- 方案 B: 简单拼接 (Ablation Baseline) ---
            # 拼接 Task 和 Phase: [B, dim*2]
            cat_feat = torch.cat([t_emb, p_emb], dim=-1)
            
            # 投影回维度: [B, dim] -> [B, 1, dim]
            logic_query = self.cat_projector(cat_feat).unsqueeze(1)
        
        # 3. 交叉注意力与分类 (保持不变)
        attn_out2, _ = self.cross_attn(query=logic_query, key=vis_contextual, value=vis_contextual)
        vis_weighted = self.norm2(logic_query + attn_out2)
        
        logits = self.classifier(vis_weighted.squeeze(1))
        return logits
import torch
import torch.nn as nn
import torchvision.models as models

class TransformerKeyframeSelector(nn.Module):
    def __init__(self, 
                 pretrained_backbone_path=None, 
                #  use_imagenet=False,  # 🟢 新增参数：显式控制是否使用 ImageNet
                 num_tasks=5, 
                 max_phases=100, 
                 window_size=3, 
                 embed_dim=128, 
                 num_heads=4):
        super().__init__()
        
        # 1. Vision Backbone 选择逻辑
        # if use_imagenet:
        #     # 🟢 直接使用 torchvision 提供的 ImageNet 预训练权重
        #     print("🧊 Mode: Using Standard ImageNet Pretrained ResNet18")
        #     resnet = models.resnet18(weights='IMAGENET1K_V1') 
        # else:
        #     # 不使用预训练，准备加载自定义权重或随机初始化
        resnet = models.resnet18(weights=None) 
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 如果提供了第一阶段路径，则覆盖 (针对微调对照组)
        if pretrained_backbone_path:
            print(f"📥 Loading Stage 1 Custom Backbone from: {pretrained_backbone_path}")
            state_dict = torch.load(pretrained_backbone_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)
            
        # 统一冻结 Backbone 参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("❄️ Backbone Frozen")
            
        # --- 后续组件保持不变 ---
        self.vis_projector = nn.Linear(512, embed_dim)
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        self.phase_embedding = nn.Embedding(max_phases, embed_dim)
        
        self.film_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim * 2)
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
        # ... forward 逻辑保持不变 ...
        B, T, C, H, W = curr_window_imgs.shape
        imgs_flat = curr_window_imgs.view(-1, C, H, W)
        
        with torch.no_grad():
            vis_feat_raw = self.backbone(imgs_flat).flatten(1)
            
        vis_feat = self.vis_projector(vis_feat_raw)
        vis_seq = vis_feat.view(B, T, -1) + self.time_embedding
        
        attn_out1, _ = self.self_attn(vis_seq, vis_seq, vis_seq)
        vis_contextual = self.norm1(vis_seq + attn_out1)
        
        t_emb = self.task_embedding(task_id)
        p_emb = self.phase_embedding(phase_id)
        
        gamma_beta = self.film_generator(t_emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        logic_query = (gamma * p_emb + beta).unsqueeze(1)
        
        attn_out2, _ = self.cross_attn(query=logic_query, key=vis_contextual, value=vis_contextual)
        vis_weighted = self.norm2(logic_query + attn_out2)
        
        logits = self.classifier(vis_weighted.squeeze(1))
        return logits
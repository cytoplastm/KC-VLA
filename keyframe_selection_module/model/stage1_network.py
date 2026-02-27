import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetContrastive(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # 1. Backbone: ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        
        # 提取特征部分 (输出维度 512)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Projector (仅保留对比学习投影头)
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim)
        )
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        feat = self.backbone(x)      # [B, 512, 1, 1]
        feat = feat.flatten(1)       # [B, 512]
        
        # 投影到嵌入空间
        emb = self.projector(feat)
        emb = F.normalize(emb, p=2, dim=1) # L2 归一化，这对 Triplet Loss 很重要
        
        return emb # 🟢 仅返回 Embedding，不再返回 task_logits
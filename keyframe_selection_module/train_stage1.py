import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.stage1_dataset import MultiTaskContrastiveDataset 
from model.stage1_network import ResNetContrastive

# ================= 配置 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 30 # 🟢 去除 LAMBDA_TASK

TASKS_CONFIG = {
    0: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PickPlaceThreetimes-v1/panda_wristcam",
    1: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PushCubeWithSignal-v1_version1/panda_wristcam",
    2: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/TeacherArmShuffle-v1_version2/panda_wristcam",
    3: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/SwapThreeCubes-v1_version2/panda_wristcam"  
}

SAVE_DIR = "/home/chenyipeng/my-Isaac-GR00T/keyframe_detection_module_multitask/checkpoints/stage1_final_version2"

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiTaskContrastiveDataset(TASKS_CONFIG, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # 🟢 模型不再需要 num_tasks 参数
    model = ResNetContrastive().to(DEVICE)
    
    # 由于只有 Triplet Loss，可以考虑将 Margin 稍微调大一点 (例如 1.0 -> 2.0)
    # 来补偿失去辅助损失带来的聚类压力
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2) 
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_triplet = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # 🟢 仅接收三个图像
        for a_img, p_img, n_img in pbar:
            a_img, p_img, n_img = a_img.to(DEVICE), p_img.to(DEVICE), n_img.to(DEVICE)
            
            optimizer.zero_grad()
            
            emb_a = model(a_img)
            emb_p = model(p_img)
            emb_n = model(n_img)
            
            # 🟢 纯 Triplet Loss
            loss = criterion_triplet(emb_a, emb_p, emb_n)
            
            loss.backward()
            optimizer.step()
            
            running_triplet += loss.item()
            pbar.set_postfix({'triplet_loss': f"{loss.item():.4f}"})
            
        avg_loss = running_triplet / len(dataloader)
        print(f"Epoch {epoch+1} Summary: Triplet Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.backbone.state_dict(), os.path.join(SAVE_DIR, "best_backbone.pth"))
            print(f"🌟 Best Model Saved (Loss: {best_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(model.backbone.state_dict(), os.path.join(SAVE_DIR, f"backbone_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train()
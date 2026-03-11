import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import KeyframeDataset 
from model.network import TransformerKeyframeSelector

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WINDOW_SIZE = 3

# 任务配置
TASKS_CONFIG = {
    0: "path/to/your/maniskill_data/PickPlaceThreetimes-v1",
    1: "path/to/your/maniskill_data/PushCubeWithSignal-v1",
    2: "path/to/your/maniskill_data/TeacherArmShuffle-v1",
    3: "path/to/your/maniskill_data/SwapThreeCubes-v1"  
}

# 权重路径
STAGE1_WEIGHTS = "path/to/stage1_checkpoint"
SAVE_DIR = "./checkpoint/stage2"

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    # 统计各任务指标
    stats = {tid: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for tid in TASKS_CONFIG.keys()}
    
    with torch.no_grad():
        for imgs, phases, task_ids, _, labels in val_loader:
            imgs, phases, task_ids = imgs.to(DEVICE), phases.to(DEVICE), task_ids.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            
            logits = model(imgs, phases, task_ids)
            val_loss += criterion(logits, labels).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            for i in range(task_ids.size(0)):
                tid = task_ids[i].item()
                p, l = preds[i].item(), labels[i].item()
                if p == 1 and l == 1: stats[tid]['tp'] += 1
                elif p == 1 and l == 0: stats[tid]['fp'] += 1
                elif p == 0 and l == 1: stats[tid]['fn'] += 1
                else: stats[tid]['tn'] += 1
                    
    print(f"\n📊 Evaluation Summary (Loss: {val_loss/len(val_loader):.4f}):")
    total_tp, total_fp, total_fn = 0, 0, 0
    for tid, s in stats.items():
        precision = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
        recall = s['tp'] / (s['tp'] + s['fn'] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        print(f"  Task {tid} | F1: {f1:.4f} | Prec: {precision:.4f} (误报:{s['fp']}) | Rec: {recall:.4f}")
        total_tp += s['tp']; total_fp += s['fp']; total_fn += s['fn']
        
    global_prec = total_tp / (total_tp + total_fp + 1e-6)
    global_rec = total_tp / (total_tp + total_fn + 1e-6)
    global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec + 1e-6)
    return global_f1, val_loss / len(val_loader)

def train_stage2():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("🚀 Stage 2 Training Start...")
    # 加载数据集
    train_dataset = KeyframeDataset(TASKS_CONFIG, mode='train', context_length=WINDOW_SIZE)
    val_dataset = KeyframeDataset(TASKS_CONFIG, mode='val', context_length=WINDOW_SIZE)
    
    # 自动统计正负样本比例
    train_labels = [s['label'] for s in train_dataset.samples]
    pos_n = sum(train_labels)
    neg_n = len(train_labels) - pos_n
    ratio = neg_n / (pos_n + 1e-6)
    print(f"📈 Ratio Stats | Pos: {int(pos_n)} | Neg: {int(neg_n)} | Imbalance: 1:{ratio:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 初始化模型 (Backbone 自动冻结)
    model = TransformerKeyframeSelector(
        pretrained_backbone_path=STAGE1_WEIGHTS, 
        num_tasks=len(TASKS_CONFIG),
        max_phases=100 
    ).to(DEVICE)
    
    # 优化器配置：权重衰减设为 0.05 以增强泛化能力
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.05)
    
    # 动态调整 pos_weight。针对 1:5 左右的比例，设置 5.5-6.0 是比较激进但有效的
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]).to(DEVICE))
    
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, phases, task_ids, _, labels in pbar:
            imgs, phases, task_ids = imgs.to(DEVICE), phases.to(DEVICE), task_ids.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(imgs, phases, task_ids)
            loss = criterion(logits, labels)
            loss.backward()
            
            # 梯度裁剪：防止 Transformer 训练不稳
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()

        # 验证
        val_f1, _ = evaluate(model, val_loader, criterion)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_stage2.pth"))
            print(f"🌟 New Best F1: {best_f1:.4f} | Model Saved")
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train_stage2()
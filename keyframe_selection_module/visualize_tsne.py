import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.stage1_dataset import MultiTaskContrastiveDataset
from model.stage1_network import ResNetContrastive

# === 配置 ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BACKBONE_PATH = "/home/chenyipeng/my-Isaac-GR00T/keyframe_detection_module_multitask/checkpoints/stage1_final_version2/backbone_epoch_30.pth"
# TASKS_CONFIG = {
#     0: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PickPlaceThreetimes-v1/panda_wristcam",
#     1: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PushCubeWithSignal-v1_version1/panda_wristcam",
#     2: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/TeacherArmShuffle-v1_version2/panda_wristcam",
# }

TASKS_CONFIG = {
    0: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PickPlaceThreetimes-v1/panda_wristcam",
    1: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PushCubeWithSignal-v1_version1/panda_wristcam",
    2: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/TeacherArmShuffle-v1_version2/panda_wristcam",
    3: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/SwapThreeCubes-v1_version2/panda_wristcam"
}

def extract_features():
    # 1. 加载模型
    model = ResNetContrastive(embed_dim=128).to(DEVICE)
    model.backbone.load_state_dict(torch.load(BACKBONE_PATH, map_location=DEVICE))
    model.eval()

    # 2. 加载数据 (不需要增强)
    dataset = MultiTaskContrastiveDataset(TASKS_CONFIG, transform=None) 
    # 注意：为了可视化，我们只需要取单张图，可以简单修改 Dataset 或直接循环 samples
    
    all_embeddings = []
    all_task_ids = []
    all_phase_ids = []

    print("🚀 Extracting features for t-SNE...")
    with torch.no_grad():
        # 我们遍历 dataset.samples，手动加载图片提取特征
        # 建议随机选 1000-2000 个样本，太多了 t-SNE 跑得慢
        indices = np.random.choice(len(dataset.samples), min(2000, len(dataset.samples)), replace=False)
        
        for idx in tqdm(indices):
            sample = dataset.samples[idx]
            img = dataset.load_image(sample['video_path'], sample['frame_idx'])
            # 简单的预处理
            # img_t = torch.from_numpy(np.array(img.resize((224,224)))).permute(2,0,1).float() / 255.0
            # img_t = torch.nn.functional.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)

            import torchvision.transforms.functional as TF # 🟢 增加这个导入
            img_resized = img.resize((224, 224))
            img_t = TF.to_tensor(img_resized) # 自动完成 /255.0 和 permute(2,0,1)
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_t = img_t.unsqueeze(0).to(DEVICE)
            
            # 提取 Embedding
            emb = model(img_t) # [1, 128]
            
            all_embeddings.append(emb.cpu().numpy())
            all_task_ids.append(sample['task_id'])
            all_phase_ids.append(sample['phase_id'])

    return np.vstack(all_embeddings), np.array(all_task_ids), np.array(all_phase_ids)

def plot_tsne(embeddings, task_ids, phase_ids):
    print("🪄 Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    vis_dims = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    unique_tasks = np.unique(task_ids)
    
    # 使用与之前一致的 tab10 色谱
    base_colors = plt.cm.get_cmap('tab10') 

    for i, t_id in enumerate(unique_tasks):
        t_mask = (task_ids == t_id)
        current_phases = np.unique(phase_ids[t_mask])
        
        for j, p_id in enumerate(current_phases):
            p_mask = t_mask & (phase_ids == p_id)
            
            # 透明度区分阶段
            alpha_val = 0.3 + (0.7 * (j + 1) / len(current_phases))
            
            plt.scatter(
                vis_dims[p_mask, 0], 
                vis_dims[p_mask, 1], 
                color=base_colors(i), 
                alpha=alpha_val,
                # 🟢 修改点：label 只显示 Task ID 数字，对应 Task 0, 1, 2...
                label=f"Task {t_id}" if j == 0 else "", 
                edgecolors='none',
                s=50
            )
            
            # 标记中心 Phase ID (P0, P1...)
            center = vis_dims[p_mask].mean(axis=0)
            plt.text(center[0], center[1], f"P{p_id}", fontsize=10, weight='extra bold', 
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # 🟢 调整图例样式，使其更接近你上传的示例
    plt.legend(
        loc='upper right',       # 右上角
        title=None,              # 去掉标题，保持简洁
        fontsize='medium',
        markerscale=1.2,         # 稍微放大图例里的色球
        frameon=True,            # 显示边框
        handletextpad=0.5,       # 调整球和数字之间的间距
        borderpad=0.8            # 调整图例内边距
    )

    plt.title("Improved t-SNE: Task (Color) & Phase (Shade) Visualization")
    plt.tight_layout()
    plt.savefig("tsne_stage1_final.png")
    print("✅ Plot saved as tsne_stage1_final.png")
    plt.show()

if __name__ == "__main__":
    embs, t_ids, p_ids = extract_features()
    plot_tsne(embs, t_ids, p_ids)
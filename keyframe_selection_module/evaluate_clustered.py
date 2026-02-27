import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import sys
import glob
import imageio
import json
from tqdm import tqdm

# === 硬件与环境配置 ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 3

# 引入网络定义
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from model.network import TransformerKeyframeSelector

# ==========================================
# 🟢 1. 配置映射 (需与训练代码一致)
# ==========================================
NAME_TO_ID = {
    "PickPlaceThreetimes-v1": 0,
    "PushCubeWithSignal-v1_version1": 1,
    "TeacherArmShuffle-v1_version2": 2,
    "SwapThreeCubes-v1_version2": 3 
}

ID_TO_ROOT = {
    0: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PickPlaceThreetimes-v1/panda_wristcam",
    1: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/PushCubeWithSignal-v1_version1/panda_wristcam",
    2: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/TeacherArmShuffle-v1_version2/panda_wristcam",
    3: "/home/chenyipeng/data/maniskill_data/lerobot_datasets/SwapThreeCubes-v1_version2/panda_wristcam"
}

# TEST_TASKS = ["PickPlaceThreetimes-v1", "PushCubeWithSignal-v1_version1", "TeacherArmShuffle-v1_version2", "SwapThreeCubes-v1_version2"]
# TEST_COUNT_PER_TASK = 20 

TEST_TASKS = ["PickPlaceThreetimes-v1"]
TEST_COUNT_PER_TASK = 100 

MODEL_PATH = "/home/chenyipeng/my-Isaac-GR00T/keyframe_detection_module_multitask/checkpoints/stage2_final_version2/best_model_stage2.pth"
RESULT_BASE_DIR = "/home/chenyipeng/my-Isaac-GR00T/keyframe_detection_module_multitask/result/stage2_final_version2_cluster_real"

# ==========================================
# 🟢 2. 评估算法 (聚类 + 最高置信度匹配)
# ==========================================
def calculate_clustered_metrics(probs, kf_indices, threshold=0.5, cluster_dist=5, tolerance=10):
    # 忽略起始关键帧 (索引为 0)
    gt_keyframes = sorted([k for k in kf_indices if k > 0])
    
    # 提取所有超过阈值的帧
    detected_frames = [i for i, p in enumerate(probs) if p > threshold]
    if not detected_frames:
        return 0, 0, len(gt_keyframes), []

    # 聚类逻辑 (间距 <= 5)
    clusters = []
    curr_cluster = [detected_frames[0]]
    for i in range(1, len(detected_frames)):
        if detected_frames[i] - detected_frames[i-1] <= cluster_dist:
            curr_cluster.append(detected_frames[i])
        else:
            clusters.append(curr_cluster)
            curr_cluster = [detected_frames[i]]
    clusters.append(curr_cluster)

    # 选取每个聚类中置信度最高的帧作为触发点
    pred_events = []
    for c in clusters:
        cluster_probs = [probs[idx] for idx in c]
        best_frame = c[np.argmax(cluster_probs)]
        pred_events.append(best_frame)
    
    tp, fp = 0, 0
    matched_gt = set()
    for p_event in pred_events:
        hit = False
        for i, gt in enumerate(gt_keyframes):
            if abs(p_event - gt) <= tolerance:
                if i not in matched_gt:
                    tp += 1
                    matched_gt.add(i)
                hit = True
                break
        if not hit:
            fp += 1
            
    fn = len(gt_keyframes) - len(matched_gt)
    return tp, fp, fn, pred_events

# ==========================================
# 🟢 3. 推理逻辑 (包含任务达成即停止)
# ==========================================
def load_model(model_path, num_tasks):
    # model = TransformerKeyframeSelector(pretrained_backbone_path=None, num_tasks=num_tasks, max_phases=100,use_film=False)
    model = TransformerKeyframeSelector(pretrained_backbone_path=None, num_tasks=num_tasks, max_phases=100)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def run_inference(video_path, save_video_path, model, kf_indices, task_id, task_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = imageio.get_writer(save_video_path, fps=fps, codec='libx264', quality=8)
    
    # 目标：找出除了起始帧之外的所有关键帧
    valid_gt_list = sorted([k for k in kf_indices if k > 0])
    total_target_milestones = len(valid_gt_list)
    
    frame_buffer = []
    probabilities = []
    
    # 模拟在线聚类计数状态
    current_detected_count = 0
    is_currently_in_cluster = False
    gap_counter = 0 
    is_module_active = True

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 任务达成终止逻辑
        if current_detected_count >= total_target_milestones:
            is_module_active = False
            
        # 2. PickPlace 忽略前 50 帧
        is_ignored_period = (task_name == "PickPlaceThreetimes-v1" and f_idx < 1)
        
        # 3. 计算当前 Phase
        current_phase = 0
        for i, kf in enumerate(valid_gt_list):
            if f_idx >= kf: current_phase = i + 1
            else: break
        
        prob = 0.0
        # 4. 执行模型推理
        if is_module_active and not is_ignored_period:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_img = transform(Image.fromarray(img_rgb))
            frame_buffer.append(tensor_img)
            if len(frame_buffer) > WINDOW_SIZE: frame_buffer.pop(0)
            
            if len(frame_buffer) == WINDOW_SIZE:
                input_win = torch.stack(frame_buffer).unsqueeze(0).to(DEVICE)
                p_in = torch.tensor([min(current_phase, 99)]).to(DEVICE)
                t_in = torch.tensor([task_id]).to(DEVICE)
                with torch.no_grad():
                    logits = model(input_win, p_in, t_in)
                    prob = torch.sigmoid(logits).item()

        # 5. 在线计数器 (确认聚类结束)
        if prob > 0.5:
            is_currently_in_cluster = True
            gap_counter = 0
        elif is_currently_in_cluster:
            gap_counter += 1
            if gap_counter > 5: # 连续 5 帧低于阈值认为一次触发完成
                current_detected_count += 1
                is_currently_in_cluster = False
        
        probabilities.append(prob)

        # 6. 可视化
        cv2.rectangle(frame, (20, 20), (480, 95), (0,0,0), -1)
        status = "ACTIVE" if is_module_active else "TERMINATED"
        color = (0, 255, 0) if is_module_active else (0, 165, 255)
        cv2.putText(frame, f"Module: {status}", (30, 45), 1, 1.0, color, 1)
        cv2.putText(frame, f"Det: {current_detected_count}/{total_target_milestones} | Prob: {prob:.2f}", (30, 75), 1, 1.0, (255,255,255), 1)
        if f_idx in valid_gt_list: cv2.circle(frame, (frame.shape[1]-30, 30), 15, (0,255,0), -1)
        
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()
    return probabilities

def plot_and_save(probs, kf_indices, pred_events, save_path, title):
    plt.figure(figsize=(15, 6))
    plt.plot(probs, label='Confidence', color='blue', alpha=0.6)
    plt.axhline(y=0.5, color='red', linestyle='--')
    for kf in kf_indices:
        if kf > 0:
            plt.axvline(x=kf, color='green', linewidth=2, label='GT' if kf == max(kf_indices) else "")
            plt.axvspan(kf-10, kf+10, color='green', alpha=0.1)
    for pe in pred_events:
        plt.scatter(pe, 0.5, color='red', marker='x', s=100, zorder=5)
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    model = load_model(MODEL_PATH, num_tasks=len(NAME_TO_ID))
    global_stats = {tid: {"tp": 0, "fp": 0, "fn": 0} for tid in NAME_TO_ID.values()}

    for t_name in TEST_TASKS:
        tid = NAME_TO_ID[t_name]
        root = ID_TO_ROOT[tid]
        res_dir = os.path.join(RESULT_BASE_DIR, t_name)
        os.makedirs(res_dir, exist_ok=True)
        
        with open(os.path.join(root, "meta/keyframes.json"), 'r') as f:
            kf_map = json.load(f)
        
        v_files = sorted(glob.glob(os.path.join(root, "videos/chunk-000/observation.images.image/*.mp4")))[-TEST_COUNT_PER_TASK:]

        print(f"\n🚀 Evaluating {t_name}...")
        for vp in tqdm(v_files):
            vn = os.path.basename(vp)
            ep = vn.replace("episode_", "").replace(".mp4", "")
            ep_key = ep if ep in kf_map else str(int(ep))
            if ep_key not in kf_map: continue
            
            gt_kf = kf_map[ep_key]
            probs = run_inference(vp, os.path.join(res_dir, vn+"_vis.mp4"), model, gt_kf, tid, t_name)
            
            if probs:
                tp, fp, fn, events = calculate_clustered_metrics(probs, gt_kf)
                global_stats[tid]["tp"] += tp
                global_stats[tid]["fp"] += fp
                global_stats[tid]["fn"] += fn
                plot_and_save(probs, gt_kf, events, os.path.join(res_dir, vn+"_plot.png"), f"{t_name} - {vn}")

    print("\n" + "="*60)
    print(f"{'TASK NAME':<35} | {'PREC':<7} | {'REC':<7} | {'F1':<7}")
    print("-" * 60)
    for t_name, tid in NAME_TO_ID.items():
        s = global_stats[tid]
        p = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
        r = s['tp'] / (s['tp'] + s['fn'] + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        print(f"{t_name:<35} | {p:.4f} | {r:.4f} | {f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
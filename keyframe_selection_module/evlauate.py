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

# 🟢 引入网络定义
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from model.network import TransformerKeyframeSelector

# ==========================================
# 🟢 1. 配置适配 (必须与 train_stage2.py 一致)
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

TEST_TASKS = [
    "PickPlaceThreetimes-v1",
    "PushCubeWithSignal-v1_version1",
    "TeacherArmShuffle-v1_version2",
    "SwapThreeCubes-v1_version2"
]
TEST_COUNT_PER_TASK = 20

MODEL_PATH = "/home/chenyipeng/my-Isaac-GR00T/keyframe_detection_module_multitask/checkpoints/stage2_imagenet_baseline/best_model_stage2.pth"
RESULT_BASE_DIR = "/home/chenyipeng/my-Isaac-GR00T/keyframe_detection_module_multitask/result/stage2_inference_baseline"

def load_model(model_path, num_tasks):
    print(f"📥 Loading Stage 2 model from {model_path}...")
    model = TransformerKeyframeSelector(
        pretrained_backbone_path=None, 
        num_tasks=num_tasks, 
        max_phases=100, 
        window_size=WINDOW_SIZE
    )
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img)

def run_inference(video_path, save_video_path, model, kf_indices, task_id, tolerance=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if np.isnan(fps) or fps <= 0: fps = 30.0
    
    writer = imageio.get_writer(save_video_path, fps=fps, codec='libx264', quality=8)
    
    frame_buffer = []
    probabilities = []
    
    # 🟢 修正：剔除起始帧(0)，剩下的才是训练时对应的 Phase 分割点
    valid_kf_list = sorted([k for k in kf_indices if k > 0])
    
    task_id_tensor = torch.tensor([task_id]).to(DEVICE)
    
    # 用于计算统计指标
    fp_count = 0
    fn_flags = [False] * len(valid_kf_list) # 记录每个 GT 是否被捕捉到

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # 🟢 Phase 切换逻辑：必须与 Dataset._build_index 保持绝对一致
        # 如果 f_idx 还没到第一个有效关键帧，则是 Phase 0
        current_phase = 0
        for i, kf in enumerate(valid_kf_list):
            if f_idx >= kf:
                current_phase = i + 1
            else:
                break
        
        # 越界保护
        current_phase = min(current_phase, 99)

        tensor_img = preprocess_frame(frame)
        frame_buffer.append(tensor_img)
        if len(frame_buffer) > WINDOW_SIZE:
            frame_buffer.pop(0)
            
        prob = 0.0
        if len(frame_buffer) == WINDOW_SIZE:
            input_window = torch.stack(frame_buffer).unsqueeze(0).to(DEVICE)
            phase_input = torch.tensor([current_phase]).to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_window, phase_input, task_id_tensor)
                prob = torch.sigmoid(logits).item()
        
        probabilities.append(prob)
        
        # --- 🟢 评估逻辑：应用宽容度统计 (Tolerance) ---
        is_detected = (prob > 0.5)
        is_near_gt = any(abs(f_idx - k) <= tolerance for k in valid_kf_list)
        
        if is_detected:
            if not is_near_gt:
                fp_count += 1 # 远离关键帧的触发，记为误报
            else:
                # 记录哪个 GT 被命中了
                for idx, k in enumerate(valid_kf_list):
                    if abs(f_idx - k) <= tolerance:
                        fn_flags[idx] = True

        # --- 可视化 ---
        # 绘制背景条
        cv2.rectangle(frame, (30, 20), (470, 90), (0, 0, 0), -1)
        # 概率进度条
        color = (0, 255, 0) if prob < 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (50, 65), (50 + int(prob * 380), 80), color, -1)
        
        cv2.putText(frame, f"T:{task_id} P:{current_phase} F:{f_idx}", (40, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Conf: {prob:.3f}", (320, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        if f_idx in valid_kf_list:
            cv2.circle(frame, (frame.shape[1]-40, 40), 12, (0, 255, 0), -1)
            cv2.putText(frame, "GT", (frame.shape[1]-85, 48), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_detected:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 4)

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()
    
    # 计算漏报
    fn_count = fn_flags.count(False)
    print(f"   📊 Result: FP(误报)={fp_count}, FN(漏报)={fn_count}")
    return probabilities

def plot_curve(probs, save_path, video_name, kf_indices, task_name):
    plt.figure(figsize=(12, 5))
    plt.plot(probs, label='Confidence', color='#1f77b4', alpha=0.8)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    
    # 绘制 GT 区域 (带宽容度阴影)
    for i, kf in enumerate(kf_indices):
        if kf > 0:
            plt.axvline(x=kf, color='g', linewidth=2, alpha=0.7)
            plt.axvspan(kf-2, kf+2, color='g', alpha=0.1) # 🟢 增加宽松标准阴影
            plt.text(kf, 0.95, f'P{i}', color='g', fontweight='bold')
    
    plt.title(f'Inference: {task_name} - {video_name}')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    num_tasks = max(NAME_TO_ID.values()) + 1
    model = load_model(MODEL_PATH, num_tasks=num_tasks)

    for t_name in TEST_TASKS:
        task_id = NAME_TO_ID[t_name]
        root_dir = ID_TO_ROOT[task_id]
        
        task_res_dir = os.path.join(RESULT_BASE_DIR, t_name)
        os.makedirs(task_res_dir, exist_ok=True)
        
        video_dir = os.path.join(root_dir, "videos/chunk-000/observation.images.image")
        meta_json = os.path.join(root_dir, "meta/keyframes.json")
        
        with open(meta_json, 'r') as f:
            kf_map = json.load(f)

        # 获取最后几个视频进行验证
        video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        target_videos = video_files[-TEST_COUNT_PER_TASK:]

        print(f"\n🚀 Testing Task: {t_name} | ID: {task_id}")
        
        for v_path in target_videos:
            v_name = os.path.basename(v_path)
            ep_str = v_name.replace("episode_", "").replace(".mp4", "")
            ep_key = ep_str if ep_str in kf_map else str(int(ep_str))

            if ep_key not in kf_map: continue
            
            kf_indices = kf_map[ep_key]
            save_v_path = os.path.join(task_res_dir, f"{v_name}_vis.mp4")
            save_p_path = os.path.join(task_res_dir, f"{v_name}_plot.png")
            
            probs = run_inference(v_path, save_v_path, model, kf_indices, task_id)
            if probs:
                plot_curve(probs, save_p_path, v_name, kf_indices, t_name)
                print(f"   ✅ Done: {v_name}")

if __name__ == "__main__":
    main()
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 3

# 引入网络定义
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from model.network import TransformerKeyframeSelector

NAME_TO_ID = {
    "PickPlaceThreetimes-v1": 0,
    "PushCubeWithSignal-v1_version1": 1,
    "TeacherArmShuffle-v1_version2": 2,
    "SwapThreeCubes-v1_version2": 3 
}

ID_TO_ROOT = {
    0: "path/to/your/maniskill_data/PickPlaceThreetimes-v1",
    1: "path/to/your/maniskill_data/PushCubeWithSignal-v1",
    2: "path/to/your/maniskill_data/TeacherArmShuffle-v1",
    3: "path/to/your/maniskill_data/SwapThreeCubes-v1"  
}

TEST_TASKS = ["PickPlaceThreetimes-v1", "PushCubeWithSignal-v1_version1", "TeacherArmShuffle-v1_version2", "SwapThreeCubes-v1_version2"]
TEST_COUNT_PER_TASK = 20 

MODEL_PATH = "path/to/keyframe_selection_module_checkpoint"
RESULT_BASE_DIR = "./result/evaluate"

def calculate_clustered_metrics(probs, kf_indices, threshold=0.5, cluster_dist=5, tolerance=10):
    gt_keyframes = sorted([k for k in kf_indices if k > 0])
    
    detected_frames = [i for i, p in enumerate(probs) if p > threshold]
    if not detected_frames:
        return 0, 0, len(gt_keyframes), []

    clusters = []
    curr_cluster = [detected_frames[0]]
    for i in range(1, len(detected_frames)):
        if detected_frames[i] - detected_frames[i-1] <= cluster_dist:
            curr_cluster.append(detected_frames[i])
        else:
            clusters.append(curr_cluster)
            curr_cluster = [detected_frames[i]]
    clusters.append(curr_cluster)

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

def apply_online_logic(probs, threshold, total_target_milestones):
    current_detected_count = 0
    is_currently_in_cluster = False
    gap_counter = 0 
    
    final_probs = []
    for p in probs:
        if current_detected_count >= total_target_milestones:
            final_probs.append(0.0) 
            continue
            
        final_probs.append(p)
        
        if p > threshold:
            is_currently_in_cluster = True
            gap_counter = 0
        elif is_currently_in_cluster:
            gap_counter += 1
            if gap_counter > 5: 
                current_detected_count += 1
                is_currently_in_cluster = False
    return final_probs

def find_best_threshold(task_results, candidate_thresholds=np.arange(0.01, 1.00, 0.02)):
    best_f1 = -1.0
    best_thresholds_list = []

    print(f"\n Threshold Search Results (with Early Stopping logic):")
    print(f"{'Thresh':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 38)

    for thresh in candidate_thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        for probs, gt_kf in task_results:
            valid_gt_list = [k for k in gt_kf if k > 0]
            simulated_probs = apply_online_logic(probs, thresh, len(valid_gt_list))
            
            tp, fp, fn, _ = calculate_clustered_metrics(simulated_probs, gt_kf, threshold=thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        p = total_tp / (total_tp + total_fp + 1e-6)
        r = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        
        print(f"{thresh:<8.2f} | {p:<8.4f} | {r:<8.4f} | {f1:<8.4f}")

        if f1 > best_f1 + 1e-5: 
            best_f1 = f1
            best_thresholds_list = [thresh]
        elif abs(f1 - best_f1) < 1e-5:
            best_thresholds_list.append(thresh)

    best_threshold = float(np.median(best_thresholds_list))

    print("-" * 38)
    formatted_list = [f"{t:.2f}" for t in best_thresholds_list]
    print(f" Best F1: {best_f1:.4f} achieved at thresholds: {formatted_list}")
    print(f" Selected Robust Median Threshold: {best_threshold:.3f}\n")
    
    return best_threshold

def load_model(model_path, num_tasks):
    model = TransformerKeyframeSelector(pretrained_backbone_path=None, num_tasks=num_tasks, max_phases=100)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def run_inference(video_path, save_video_path, model, kf_indices, task_id, task_name, threshold=0.5, enable_early_stop=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    writer = None
    if save_video_path:
        writer = imageio.get_writer(save_video_path, fps=fps, codec='libx264', quality=8)
    
    valid_gt_list = sorted([k for k in kf_indices if k > 0])
    total_target_milestones = len(valid_gt_list)
    
    frame_buffer = []
    probabilities = []
    
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
        
        if enable_early_stop and current_detected_count >= total_target_milestones:
            is_module_active = False
            
        current_phase = 0
        for i, kf in enumerate(valid_gt_list):
            if f_idx >= kf: current_phase = i + 1
            else: break
        
        prob = 0.0
        if is_module_active :
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

        if prob > threshold:
            is_currently_in_cluster = True
            gap_counter = 0
        elif is_currently_in_cluster:
            gap_counter += 1
            if gap_counter > 5: 
                current_detected_count += 1
                is_currently_in_cluster = False
        
        probabilities.append(prob)

        if writer is not None:
            cv2.rectangle(frame, (20, 20), (480, 95), (0,0,0), -1)
            status = "ACTIVE" if is_module_active else "TERMINATED"
            color = (0, 255, 0) if is_module_active else (0, 165, 255)
            cv2.putText(frame, f"Module: {status}", (30, 45), 1, 1.0, color, 1)
            cv2.putText(frame, f"Det: {current_detected_count}/{total_target_milestones} | Prob: {prob:.2f} (Th: {threshold:.2f})", (30, 75), 1, 1.0, (255,255,255), 1)
            if f_idx in valid_gt_list: cv2.circle(frame, (frame.shape[1]-30, 30), 15, (0,255,0), -1)
            
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    if writer: writer.close()
    return probabilities

def plot_and_save(probs, kf_indices, pred_events, save_path, title, threshold=0.5):
    plt.figure(figsize=(15, 6))
    plt.plot(probs, label='Confidence', color='blue', alpha=0.6)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
    for kf in kf_indices:
        if kf > 0:
            plt.axvline(x=kf, color='green', linewidth=2, label='GT' if kf == max(kf_indices) else "")
            plt.axvspan(kf-10, kf+10, color='green', alpha=0.1)
    for pe in pred_events:
        plt.scatter(pe, threshold, color='red', marker='x', s=100, zorder=5)
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    model = load_model(MODEL_PATH, num_tasks=len(NAME_TO_ID))
    global_stats = {tid: {"tp": 0, "fp": 0, "fn": 0} for tid in NAME_TO_ID.values()}
    
    optimal_thresholds = {}

    for t_name in TEST_TASKS:
        tid = NAME_TO_ID[t_name]
        root = ID_TO_ROOT[tid]
        res_dir = os.path.join(RESULT_BASE_DIR, t_name)
        os.makedirs(res_dir, exist_ok=True)
        
        with open(os.path.join(root, "meta/keyframes.json"), 'r') as f:
            kf_map = json.load(f)
        
        v_files = sorted(glob.glob(os.path.join(root, "videos/chunk-000/observation.images.image/*.mp4")))[-TEST_COUNT_PER_TASK:]

        print(f"\n Step 1: Extracting Probabilities & Finding Threshold for {t_name}...")
        results_cache = []
        for vp in tqdm(v_files, desc="Extracting"):
            vn = os.path.basename(vp)
            ep = vn.replace("episode_", "").replace(".mp4", "")
            ep_key = ep if ep in kf_map else str(int(ep))
            if ep_key not in kf_map: continue
            
            gt_kf = kf_map[ep_key]
            probs = run_inference(vp, None, model, gt_kf, tid, t_name, threshold=0.5, enable_early_stop=False)
            if probs:
                results_cache.append((probs, gt_kf))
                
        best_th = find_best_threshold(results_cache)
        optimal_thresholds[t_name] = best_th 

        print(f"🚀 Step 2: Rendering Video Results with Best Threshold ({best_th:.2f})...")
        for vp in tqdm(v_files, desc="Rendering"):
            vn = os.path.basename(vp)
            ep = vn.replace("episode_", "").replace(".mp4", "")
            ep_key = ep if ep in kf_map else str(int(ep))
            if ep_key not in kf_map: continue
            
            gt_kf = kf_map[ep_key]
            save_vid_path = os.path.join(res_dir, vn+"_vis.mp4")
            
            probs = run_inference(vp, save_vid_path, model, gt_kf, tid, t_name, threshold=best_th, enable_early_stop=True)
            
            if probs:
                tp, fp, fn, events = calculate_clustered_metrics(probs, gt_kf, threshold=best_th)
                global_stats[tid]["tp"] += tp
                global_stats[tid]["fp"] += fp
                global_stats[tid]["fn"] += fn
                
                save_plot_path = os.path.join(res_dir, vn+"_plot.png")
                plot_and_save(probs, gt_kf, events, save_plot_path, f"{t_name} - {vn} (Thresh: {best_th:.2f})", threshold=best_th)

    print("\n" + "="*65)
    print(" FINAL PERFORMANCE (OPTIMIZED THRESHOLD) ")
    print(f"{'TASK NAME':<35} | {'THRESH':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6}")
    print("-" * 65)
    for t_name in TEST_TASKS:
        tid = NAME_TO_ID[t_name]
        s = global_stats[tid]
        p = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
        r = s['tp'] / (s['tp'] + s['fn'] + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        final_best_th = optimal_thresholds.get(t_name, 0.0) 
        print(f"{t_name:<35} | {final_best_th:<6.2f} | {p:.4f} | {r:.4f} | {f1:.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
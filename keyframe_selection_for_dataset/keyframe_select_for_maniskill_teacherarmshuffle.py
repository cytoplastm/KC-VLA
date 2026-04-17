import h5py
import numpy as np
import cv2
import json
import os
from pathlib import Path

TASK_NAME = "TeacherArmShuffle-v1_version2"

DATA_ROOT = "./data/maniskill_data"
LEROBOT_ROOT = "./data/lerobot_datasets"
SAVE_ROOT = "./keyframe_selection_for_dataset/result"

H5_PATH = Path(DATA_ROOT) / "panda/TeacherArmShuffle-v1/motionplanning/trajectory_panda.rgb.pd_ee_delta_pose.physx_cpu.h5"
VIDEO_DIR = Path(LEROBOT_ROOT) / TASK_NAME / "panda_wristcam/videos/chunk-000/observation.images.image"
OUTPUT_ROOT = Path(SAVE_ROOT) / TASK_NAME

GRIPPER_OPEN_THRESH = 0.035
GRIPPER_CLOSE_THRESH = 0.020

TEACHER_GRIPPER_IDX = 20
# =========================================================

def get_teacher_gripper_state(traj_group):
    """Extract teacher_panda's finger position (Index 20) from env_states"""
    data = traj_group['env_states']['articulations']['teacher_panda'][:]
    return data[:, TEACHER_GRIPPER_IDX]

def find_keyframe_indices(finger_pos):
    """
    Logic: Identify 3 keyframes based on gripper state transitions
    1. Start Frame (Episode initialization)
    2. Release 1: Closed -> Open (<= 0.035 to > 0.035)
    3. Grasp 2: Open -> Closed (> 0.020 to <= 0.020)
    """
    rising_edges = []  # Release events
    falling_edges = [] # Grasp events
    
    for t in range(1, len(finger_pos)):
        prev = finger_pos[t-1]
        curr = finger_pos[t]
        
        if prev <= GRIPPER_OPEN_THRESH and curr > GRIPPER_OPEN_THRESH:
            rising_edges.append(t)
            
        if prev > GRIPPER_CLOSE_THRESH and curr <= GRIPPER_CLOSE_THRESH:
            falling_edges.append(t)
            
    kf_0 = 0
    
    kf_1 = rising_edges[0] if len(rising_edges) > 0 else 0
    
    kf_2 = falling_edges[1] if len(falling_edges) > 1 else kf_1

    final_frame = len(finger_pos) - 1
    if kf_1 == 0: kf_1 = final_frame

    if kf_2 <= kf_1: 
        kf_2 = min(kf_1 + 10, final_frame) 
        
    return [int(kf_0), int(kf_1), int(kf_2)]

def extract_and_save_images(ep_idx, frames, video_dir, img_root_dir):
    """Read video and save 3 frames into episode-specific subfolders"""
    video_name = f"episode_{int(ep_idx):06d}.mp4"
    video_path = os.path.join(video_dir, video_name)
    
    if not os.path.exists(video_path):
        print(f"⚠️ Warning: Video {video_name} not found, skipping extraction.")
        return False

    episode_folder_name = f"episode_{int(ep_idx):06d}"
    episode_dir = os.path.join(img_root_dir, episode_folder_name)
    os.makedirs(episode_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_name}")
        return False
    
    saved_count = 0
    for i, frame_idx in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        
        if ret:
            # Filename: episode_000000_i.jpg (i from 0 to 2)
            save_name = f"episode_{int(ep_idx):06d}_{i}.jpg"
            save_path = os.path.join(episode_dir, save_name)
            cv2.imwrite(save_path, img)
            saved_count += 1
        else:
            print(f"❌ Failed to read frame: Episode {ep_idx}, Frame {frame_idx}")

    cap.release()
    return saved_count == 3

def main():
    images_root_dir = os.path.join(OUTPUT_ROOT, "keyframes")
    os.makedirs(images_root_dir, exist_ok=True)
    
    json_path = os.path.join(OUTPUT_ROOT, "keyframes.json")
    
    keyframe_data = {}
    
    try:
        with h5py.File(H5_PATH, 'r') as f:
            traj_keys = sorted(list(f.keys()), key=lambda x: int(x.split('_')[1]))
            total_traj = len(traj_keys)
            print(f"🔢 Analyzing {total_traj} trajectories...")

            for i, traj_id in enumerate(traj_keys):
                idx_str = traj_id.split('_')[1]
                
                finger_pos = get_teacher_gripper_state(f[traj_id])
                indices = find_keyframe_indices(finger_pos)
                
                keyframe_data[idx_str] = indices
                extract_and_save_images(idx_str, indices, VIDEO_DIR, images_root_dir)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed: {i + 1}/{total_traj} (ID: {idx_str}, Indices: {indices})")

    except FileNotFoundError:
        print("❌ Error: H5 file not found. Check DATA_ROOT.")
        return
    except Exception as e:
        print(f"❌ Exception: {e}")
        return

    with open(json_path, 'w') as jf:
        json.dump(keyframe_data, jf, indent=4)
        
if __name__ == "__main__":
    main()
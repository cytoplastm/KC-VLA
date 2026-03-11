import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

TASK_NAME = "PickPlaceThreetimes-v1"

DATA_ROOT = "path/to/your/maniskill_data/lerobot_datasets"
SAVE_ROOT = "./result"

PEAK_THRESHOLD = 0.2    
TARGET_PEAK_COUNT = 3    
SKIP_FIRST_N_FRAMES = 10 

def extract_frame_from_video(video_path, frame_idx):

    if not video_path.exists():
        print(f"  [Error] Video not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def find_peaks_over_threshold(z_values, threshold, max_count, skip_frames):

    found_indices = []
    
    if len(z_values) == 0:
        return found_indices

    if z_values[0] > threshold:
        state = "waiting_for_low"
    else:
        state = "waiting_for_high"
        
    current_peak_val = -1.0
    current_peak_idx = -1
    
    for i, z in enumerate(z_values):
        if i < skip_frames:
            continue

        if state == "waiting_for_low":
            if z < threshold:
                state = "waiting_for_high"
        
        elif state == "waiting_for_high":
            if z > threshold:
                state = "in_peak"
                current_peak_val = z
                current_peak_idx = i
                
        elif state == "in_peak":
            if z > current_peak_val:
                current_peak_val = z
                current_peak_idx = i
            
            if z < threshold:
                found_indices.append(current_peak_idx)
                state = "waiting_for_high"
                
                if len(found_indices) >= max_count:
                    return found_indices

    if state == "in_peak":
        found_indices.append(current_peak_idx)
        
    return found_indices

def main():

    base_dir = Path(DATA_ROOT) / TASK_NAME / "panda_wristcam/data/chunk-000"
    dataset_root = base_dir.parent.parent
    video_dir = dataset_root / "videos" / base_dir.name / "observation.images.image"

    save_dir = Path(SAVE_ROOT) / TASK_NAME
    img_save_dir = save_dir / "keyframes" 
    
    img_save_dir.mkdir(parents=True, exist_ok=True)
    # ===================================================

    keyframes = {}
    file_list = sorted(base_dir.glob("episode_*.parquet"))
    print(f"Looking for videos in: {video_dir}")

    for parquet_path in file_list:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"Error reading {parquet_path.name}: {e}")
            continue

        if "episode_index" in df.columns:
            episode_idx = int(df["episode_index"].iloc[0])
        else:
            try:
                episode_idx = int(parquet_path.stem.split('_')[-1])
            except:
                episode_idx = 0

        video_filename = f"episode_{episode_idx:06d}.mp4"
        current_video_path = video_dir / video_filename
        
        try:
            z_axis_data = df["observation.state"].apply(lambda x: x[2]).to_numpy()
        except KeyError:
            print(f"Skipping {parquet_path.name}: 'observation.state' not found.")
            continue

        target_frames = [0] 

        peak_frames = find_peaks_over_threshold(
            z_axis_data, 
            PEAK_THRESHOLD, 
            TARGET_PEAK_COUNT, 
            SKIP_FIRST_N_FRAMES
        )
        
        target_frames.extend(peak_frames)
        
        keyframes[str(episode_idx)] = target_frames
        
        print(f"Ep {episode_idx}: Frames {target_frames}")

        for frame_idx in target_frames:
            img = extract_frame_from_video(current_video_path, frame_idx)
            if img:
                z_val = z_axis_data[frame_idx]

                save_name = f"episode_{episode_idx}_frame_{frame_idx}_z_{z_val:.3f}.png"
                img.save(img_save_dir / save_name)

    out_path = save_dir / "keyframes.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(keyframes, f, indent=2)
        print(f"\nProcessing complete.")
        print(f"Index saved to {out_path}")
        print(f"Images saved to {img_save_dir}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

if __name__ == "__main__":
    main()

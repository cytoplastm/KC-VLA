import json
import cv2
import numpy as np
from pathlib import Path

TASK_NAME = "PushCubeWithSignal-v1"


DATA_ROOT = "./data/maniskill_data/lerobot_datasets"
SAVE_ROOT = "./keyframe_selection_for_dataset/result"

# Define dynamic paths
VIDEO_DIR = Path(DATA_ROOT) / TASK_NAME / "panda_wristcam/videos/chunk-000/observation.images.image"
BASE_SAVE_DIR = Path(SAVE_ROOT) / TASK_NAME
IMAGES_SAVE_DIR = BASE_SAVE_DIR / "keyframes"

BRIGHTNESS_DROP_THRESHOLD = 20 
BRIGHTNESS_RISE_THRESHOLD = 20  
MIN_FRAME_INTERVAL = 5          

BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_SAVE_DIR.mkdir(parents=True, exist_ok=True)

keyframes = {}

video_files = sorted(VIDEO_DIR.glob("episode_*.mp4"))

if not video_files:
    print(f"Error: No .mp4 files found in {VIDEO_DIR}")
    print("Please check if the DATA_ROOT path is correctly set.")
else:
    print(f"Found {len(video_files)} videos. Starting process (Target: 5 keyframes per episode)...")

for video_path in video_files:
    print(f"Processing {video_path.name}...", end=" ", flush=True)
    
    try:
        episode_idx = int(video_path.stem.split('_')[-1])
    except:
        print("Skipping (filename format error)")
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Failed to open video")
        continue
    episode_keyframes = [0]
    
    prev_brightness = None
    
    # --- State Machine for Signal Detection ---
    # 0: Waiting for first pulse (ON 1)
    # 1: Waiting for first extinction (OFF 1)
    # 2: Waiting for second pulse (ON 2)
    # 3: Waiting for second extinction (OFF 2)
    event_stage = 0 
    
    frame_idx = 0
    last_keyframe_frame = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_idx == 0:
            save_path = IMAGES_SAVE_DIR / f"episode_{episode_idx}_00_start.jpg"
            cv2.imwrite(str(save_path), frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_brightness = np.mean(gray)

        if prev_brightness is not None:
            diff = curr_brightness - prev_brightness
            
            if (frame_idx - last_keyframe_frame > MIN_FRAME_INTERVAL):
                
                # === Stage 0: Detect First Light Pulse (ON 1) ===
                if event_stage == 0:
                    if diff > BRIGHTNESS_RISE_THRESHOLD:
                        episode_keyframes.append(frame_idx)
                        last_keyframe_frame = frame_idx
                        event_stage = 1  
                        
                        save_path = IMAGES_SAVE_DIR / f"episode_{episode_idx}_01_ON_1.jpg"
                        cv2.imwrite(str(save_path), frame)

                # === Stage 1: Detect First Pulse End (OFF 1) ===
                elif event_stage == 1:
                    if diff < -BRIGHTNESS_DROP_THRESHOLD:
                        episode_keyframes.append(frame_idx)
                        last_keyframe_frame = frame_idx
                        event_stage = 2  
                        
                        save_path = IMAGES_SAVE_DIR / f"episode_{episode_idx}_02_OFF_1.jpg"
                        cv2.imwrite(str(save_path), frame)

                # === Stage 2: Detect Second Light Pulse (ON 2) ===
                elif event_stage == 2:
                    if diff > BRIGHTNESS_RISE_THRESHOLD:
                        episode_keyframes.append(frame_idx)
                        last_keyframe_frame = frame_idx
                        event_stage = 3  
                        
                        save_path = IMAGES_SAVE_DIR / f"episode_{episode_idx}_03_ON_2.jpg"
                        cv2.imwrite(str(save_path), frame)

                # === Stage 3: Detect Second Pulse End (OFF 2) ===
                elif event_stage == 3:
                    if diff < -BRIGHTNESS_DROP_THRESHOLD:
                        episode_keyframes.append(frame_idx)
                        last_keyframe_frame = frame_idx
                        
                        save_path = IMAGES_SAVE_DIR / f"episode_{episode_idx}_04_OFF_2.jpg"
                        cv2.imwrite(str(save_path), frame)
                        
                        break
        
        prev_brightness = curr_brightness
        frame_idx += 1

    cap.release()

    keyframes[str(episode_idx)] = episode_keyframes
    
    if len(episode_keyframes) == 5:
        print(f"Done. Frames: {episode_keyframes}")
    else:
        print(f"⚠️ Incomplete: Found {len(episode_keyframes)} frames (Stuck at Stage {event_stage})")

out_path = BASE_SAVE_DIR / "keyframes.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(keyframes, f, ensure_ascii=False, indent=2)

print(f"\nProcessing complete!")
print(f"Manifest saved to: {out_path}")
print(f"Keyframe previews saved in: {IMAGES_SAVE_DIR}")
import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
from PIL import Image
import random
import cv2 
from torchvision import transforms

class KeyframeDataset(Dataset):
    def __init__(self, root_dirs_dict, mode='train', context_length=3, transform=None):
        self.root_dirs_dict = root_dirs_dict
        self.mode = mode
        self.context_length = context_length
        
        if transform is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
                    transforms.Resize((224, 224), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        self.samples = []
        print(f"🔍 [Stage 2 Dataset] Building index for {mode}...")
        self._build_index()
        print(f"✅ [Stage 2 Dataset] Total samples: {len(self.samples)}")

    def _find_video_subdir(self, root_dir):
        path_v1 = os.path.join(root_dir, "videos", "chunk-000", "observation.images.image")
        if os.path.exists(path_v1): return path_v1
        path_v2 = os.path.join(root_dir, "videos")
        if os.path.exists(path_v2): return path_v2
        return root_dir

    def _build_index(self):
        for task_id, root_dir in self.root_dirs_dict.items():
            json_path = os.path.join(root_dir, "meta", "keyframes.json")
            if not os.path.exists(json_path): continue

            with open(json_path, 'r') as f:
                kf_data = json.load(f)

            all_episodes = sorted(list(kf_data.keys()), key=lambda x: int(x) if x.isdigit() else x)
            split_idx = int(len(all_episodes) * 0.8)
            target_episodes = all_episodes[:split_idx] if self.mode == 'train' else all_episodes[split_idx:]
            video_subdir = self._find_video_subdir(root_dir)

            for ep_key in target_episodes:
                indices = kf_data[ep_key]
                try:
                    filename = f"episode_{int(ep_key):06d}.mp4"
                except:
                    filename = f"{ep_key}.mp4"
                
                vid_path = os.path.join(video_subdir, filename)
                if not os.path.exists(vid_path): continue
                
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened(): continue
                vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if vid_len <= 0: continue

                valid_indices = indices[1:] 
                total_phases = len(valid_indices)

                # 🟢 记录该任务是否为突变型任务（例如 Task 2）
                is_abrupt_task = (task_id == 1)

                for phase_id, kf_idx in enumerate(valid_indices):
                    if kf_idx >= vid_len: continue
                    
                    # --- 1. 正样本 (Label 1.0) ---
                    pos_offsets = [0] if is_abrupt_task else [-1, 0, 1]
                    for offset in pos_offsets:
                        t = np.clip(kf_idx + offset, 0, vid_len - 1)
                        self.samples.append({
                            'video_path': vid_path, 'center_idx': t,
                            'phase_id': phase_id, 'task_id': task_id,
                            'video_len': vid_len, 'label': 1.0
                        })

                    # --- 2. 轨迹负样本 (Label 0.0) ---
                    # 每段轨迹采样数量改为 4 个
                    traj_start = 0 if phase_id == 0 else valid_indices[phase_id - 1]
                    traj_end = kf_idx
                    
                    safe_margin = 6 
                    safe_start = traj_start + safe_margin
                    safe_end = traj_end - safe_margin
                    
                    if safe_end > safe_start:
                        # 🟢 修改点：将轨迹平分为 4 段
                        num_neg_per_segment = 4
                        segment_len = (safe_end - safe_start) / float(num_neg_per_segment)
                        for seg_i in range(num_neg_per_segment):
                            s = int(safe_start + seg_i * segment_len)
                            e = int(safe_start + (seg_i + 1) * segment_len)
                            if e > s:
                                mid_t = random.randint(s, e)
                                self.samples.append({
                                    'video_path': vid_path, 'center_idx': mid_t,
                                    'phase_id': phase_id, 'task_id': task_id,
                                    'video_len': vid_len, 'label': 0.0
                                })
                    
                    # --- 3. 错位负样本 (Label 0.0) ---
                    if phase_id + 1 < total_phases:
                        self.samples.append({
                            'video_path': vid_path, 'center_idx': kf_idx,
                            'phase_id': phase_id + 1, 'task_id': task_id,
                            'video_len': vid_len, 'label': 0.0
                        })

    def load_window(self, video_path, center_idx, video_len):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros(self.context_length, 3, 224, 224)

        for i in range(self.context_length - 1, -1, -1):
            idx = np.clip(center_idx - i, 0, video_len - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = self.transform(img)
            else:
                img = torch.zeros(3, 224, 224)
            frames.append(img)
        
        cap.release()
        return torch.stack(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        window_imgs = self.load_window(sample['video_path'], sample['center_idx'], sample['video_len'])
        return (
            window_imgs, 
            torch.tensor(sample['phase_id'], dtype=torch.long), 
            torch.tensor(sample['task_id'], dtype=torch.long), 
            {'vid_path': sample['video_path'], 'idx': sample['center_idx']}, 
            torch.tensor(sample['label'], dtype=torch.float)
        )
import torch
from torch.utils.data import Dataset
import random
import os
from PIL import Image
import numpy as np
import json
import cv2

class MultiTaskContrastiveDataset(Dataset):
    def __init__(self, root_dirs_dict, transform=None):
        self.transform = transform
        self.samples = []
        
        self.task_phase_to_samples = {} 
        
        self.task_to_samples = {tid: [] for tid in root_dirs_dict.keys()}
        
        self.episode_kf_map = {} 

        for task_id, root_dir in root_dirs_dict.items():
            json_path = os.path.join(root_dir, "meta", "keyframes.json")
            video_subdir = self._find_video_subdir(root_dir)
            
            with open(json_path, 'r') as f:
                kf_data = json.load(f)
            
            for episode_key, indices in kf_data.items():
                filename = f"episode_{int(episode_key):06d}.mp4"
                vid_path = os.path.join(video_subdir, filename)
                if not os.path.exists(vid_path): continue
                
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened(): continue
                v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                self.episode_kf_map[vid_path] = indices
                
                valid_indices = indices[1:]
                
                for phase_id, frame_idx in enumerate(valid_indices):
                    if frame_idx >= v_len: continue

                    sample = {
                        'video_path': vid_path,
                        'frame_idx': frame_idx,
                        'task_id': task_id,
                        'phase_id': phase_id, 
                        'video_len': v_len
                    }
                    self.samples.append(sample)
                    curr_idx = len(self.samples) - 1
                    
                    self.task_to_samples[task_id].append(curr_idx)
                    
                    key = (task_id, phase_id)
                    if key not in self.task_phase_to_samples:
                        self.task_phase_to_samples[key] = []
                    self.task_phase_to_samples[key].append(curr_idx)

    def _find_video_subdir(self, root_dir):
        path = os.path.join(root_dir, "videos", "chunk-000", "observation.images.image")
        return path if os.path.exists(path) else root_dir

    def load_image(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret: return Image.new('RGB', (224, 224))
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __getitem__(self, index):
        anchor_info = self.samples[index]
        t_id = anchor_info['task_id']
        p_id = anchor_info['phase_id'] 
        vid_path = anchor_info['video_path']
        curr_f_idx = anchor_info['frame_idx']
        v_len = anchor_info['video_len']

        candidates = self.task_phase_to_samples.get((t_id, p_id), [])
        
        if len(candidates) > 1:
            pos_idx = random.choice(candidates)
            if pos_idx == index: 
                pos_idx = random.choice(candidates)
        else:
            pos_idx = index 
            
        pos_info = self.samples[pos_idx]

        rand_val = random.random()
        
        if rand_val < 0.33:
            offsets = [-10, -9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9, 10]
            neg_f_idx = np.clip(curr_f_idx + random.choice(offsets), 0, v_len - 1)
            if neg_f_idx in self.episode_kf_map[vid_path]:
                neg_f_idx = np.clip(neg_f_idx + 15, 0, v_len - 1)
            neg_img = self.load_image(vid_path, neg_f_idx)
            
        elif rand_val < 0.66:
            all_phase_keys = [k for k in self.task_phase_to_samples.keys() if k[0] == t_id]
            other_phase_keys = [k for k in all_phase_keys if k[1] != p_id]
            
            if len(other_phase_keys) > 0:
                target_key = random.choice(other_phase_keys)
                neg_sample_idx = random.choice(self.task_phase_to_samples[target_key])
                neg_info = self.samples[neg_sample_idx]
                neg_img = self.load_image(neg_info['video_path'], neg_info['frame_idx'])
            else:
                neg_f_idx = np.clip(curr_f_idx + random.choice([-10, 10]), 0, v_len - 1)
                neg_img = self.load_image(vid_path, neg_f_idx)

        else:
            other_tasks = [tid for tid in self.task_to_samples.keys() if tid != t_id]
            if len(other_tasks) > 0:
                target_tid = random.choice(other_tasks)
                neg_sample_idx = random.choice(self.task_to_samples[target_tid])
                neg_info = self.samples[neg_sample_idx]
                neg_img = self.load_image(neg_info['video_path'], neg_info['frame_idx'])
            else:
                neg_f_idx = np.clip(curr_f_idx + random.choice([-10, 10]), 0, v_len - 1)
                neg_img = self.load_image(vid_path, neg_f_idx)

        
        anchor_img = self.load_image(vid_path, curr_f_idx)
        pos_img = self.load_image(pos_info['video_path'], pos_info['frame_idx'])

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.samples)
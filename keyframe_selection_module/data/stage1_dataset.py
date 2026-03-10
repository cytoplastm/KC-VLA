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
        
        # 🟢 升级索引结构：按 (Task, Phase) 分组
        # key: (task_id, phase_id), value: list of sample indices
        self.task_phase_to_samples = {} 
        
        # 辅助索引：按 Task 分组 (用于跨任务负采样)
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
                
                # indices[1:] 剔除起始帧
                valid_indices = indices[1:]
                
                for phase_id, frame_idx in enumerate(valid_indices):
                    if frame_idx >= v_len: continue

                    sample = {
                        'video_path': vid_path,
                        'frame_idx': frame_idx,
                        'task_id': task_id,
                        'phase_id': phase_id, # 🟢 记录 Phase ID
                        'video_len': v_len
                    }
                    self.samples.append(sample)
                    curr_idx = len(self.samples) - 1
                    
                    # 更新索引
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
        p_id = anchor_info['phase_id'] # 当前阶段
        vid_path = anchor_info['video_path']
        curr_f_idx = anchor_info['frame_idx']
        v_len = anchor_info['video_len']

        # ==========================================
        # 1. Positive 采样 (核心修改)
        # ==========================================
        # 🟢 目标：同任务 + 同阶段 (Intra-Phase Positive)
        # 必须选那些也是 Task A 且也是 Phase K 的样本
        candidates = self.task_phase_to_samples.get((t_id, p_id), [])
        
        if len(candidates) > 1:
            # 正常情况：选一个不是自己的样本 (即不同 Episode 的同一阶段)
            pos_idx = random.choice(candidates)
            # 简单的防重试逻辑 (如果选到自己，再选一次，概率极低)
            if pos_idx == index: 
                pos_idx = random.choice(candidates)
        else:
            # 极端情况：数据极少，该 Phase 只有当前这一张图
            pos_idx = index # 只能选自己，依靠数据增强来做 Positive
            
        pos_info = self.samples[pos_idx]

        # ==========================================
        # 2. Negative 采样 (混合策略升级)
        # ==========================================
        rand_val = random.random()
        
        # 🟢 策略 A (33%): 时间邻近负样本 (Temporal Hard Negative) -> 练时序敏感
        if rand_val < 0.33:
            offsets = [-10, -9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9, 10]
            neg_f_idx = np.clip(curr_f_idx + random.choice(offsets), 0, v_len - 1)
            if neg_f_idx in self.episode_kf_map[vid_path]:
                neg_f_idx = np.clip(neg_f_idx + 15, 0, v_len - 1)
            neg_img = self.load_image(vid_path, neg_f_idx)
            
        # 🟢 策略 B (33%): 同任务-不同阶段负样本 (Phase Negative) -> 解决你的担忧！
        elif rand_val < 0.66:
            # 找到当前任务所有的 Phase ID
            # 简单做法：直接从 task_phase_to_samples 的 key 里筛
            all_phase_keys = [k for k in self.task_phase_to_samples.keys() if k[0] == t_id]
            # 排除当前 p_id
            other_phase_keys = [k for k in all_phase_keys if k[1] != p_id]
            
            if len(other_phase_keys) > 0:
                target_key = random.choice(other_phase_keys)
                neg_sample_idx = random.choice(self.task_phase_to_samples[target_key])
                neg_info = self.samples[neg_sample_idx]
                neg_img = self.load_image(neg_info['video_path'], neg_info['frame_idx'])
            else:
                # 如果该任务只有一个阶段 (rare)，退化为策略 A
                neg_f_idx = np.clip(curr_f_idx + random.choice([-10, 10]), 0, v_len - 1)
                neg_img = self.load_image(vid_path, neg_f_idx)

        # 🟢 策略 C (33%): 跨任务负样本 (Cross-Task Negative) -> 练任务区分
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
        # neg_img 依然由策略 A, B, C 产生，它们是拉开特征的关键

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        # 🟢 仅返回三元组图像，不再返回 t_id
        return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.samples)
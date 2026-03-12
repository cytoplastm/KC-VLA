import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import shutil
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional, List, Callable, Dict, Any, Deque
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from websocket_policy_server import ExternalRobotInferenceClient
import os
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)
try:
    from keyframe_selection_module.model.network import TransformerKeyframeSelector
except ImportError as e:
    print(f"❌ Error: Failed to import 'keyframe_selection_module'.")
    sys.exit(1)

EMBODIMENT_TAGS = {"panda_wristcam": "panda"}

TASKS = {"panda_wristcam": ["PickPlaceThreetimes-v1","TeacherArmShuffle-v1","PushCubeWithSignal-v1","SwapThreeCubes-v1"]}

NAME_TO_ID = {
    "PickPlaceThreetimes-v1": 0,
    "PushCubeWithSignal-v1": 1, 
    "TeacherArmShuffle-v1": 2,
    "SwapThreeCubes-v1": 3 
}

TASK_INSTRUCTIONS = {
    "SwapThreeCubes-v1": "Swap the position of the bottom and middle cubes.",
    "PushCubeWithSignal-v1": "Wait for the signal light to flash twice, then push the cube to the target.",
    "TeacherArmShuffle-v1": "After the cubes are swapped, pick up the cube that was originally in the middle.",
    "PickPlaceThreetimes-v1": "First, pick up the red cube and place it back on the table. Next, do the same for the green cube. Finally, the blue cube.",
}
STEP_LENGTHS = {
    "SwapThreeCubes-v1": 1000,
    "PushCubeWithSignal-v1": 1000,
    "TeacherArmShuffle-v1": 1000,
    "PickPlaceThreetimes-v1": 1000,
}

TASK_MAX_KEYFRAMES = {
    "SwapThreeCubes-v1": 1,
    "PushCubeWithSignal-v1": 5,
    "TeacherArmShuffle-v1": 3,
    "PickPlaceThreetimes-v1": 4
}

# calualating by ~/keyframe_selection_module/threshold_finding.py
TASK_THRESHOLDS = {
    "SwapThreeCubes-v1": 0.49,
    "PushCubeWithSignal-v1": 0.49,
    "TeacherArmShuffle-v1": 0.47,
    "PickPlaceThreetimes-v1": 0.91,
}

@dataclass
class EvalConfig:
    host: str = "0.0.0.0"
    port: int = 5001
    url: Optional[str] = None
    resize_size: int = 224
    replan_steps: int = 5
    
    queue_len: int = 6
    
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    cpu_sim: bool = True
    seed: int = 0
    save_example_image: bool = False

    control_freq: Optional[int] = 20
    sim_freq: Optional[int] = 100

    render_mode: str = "rgb_array"
    save_video: bool = True
    save_results: Optional[str] = None
    
    save_path: str = './evaluate/result/video'
    keyframe_save_path: Optional[str] = '.evaluate/result'

    model_checkpoint: str = 'path to KSM checkpoint'

    shader: str = "default"
    num_per_task: int = 50

class KeyframeManagerAI:
    def __init__(self, task_name: str, model_path: str, save_root: str = None, device='cuda', max_keyframes: int = 50):
        self.task_name = task_name
        self.save_root = save_root
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_keyframes = max_keyframes
        
        self.threshold = TASK_THRESHOLDS.get(self.task_name, 0.5)
        print(f"🎯 Threshold for '{self.task_name}' set to: {self.threshold}")
        
        if task_name in NAME_TO_ID:
            self.task_id = NAME_TO_ID[task_name]
            print(f"✅ Task '{task_name}' matched to ID: {self.task_id}")
        else:
            found = False
            for k, v in NAME_TO_ID.items():
                if k in task_name or task_name in k:
                    self.task_id = v
                    found = True
                    print(f"⚠️ Fuzzy match: '{task_name}' matched to ID {v} ({k})")
                    break
            if not found:
                raise ValueError(f"❌ Critical Error: Task '{task_name}' has no mapped ID in NAME_TO_ID!")

        print(f"🔄 Loading Keyframe Model from {model_path} ...")
        self.model = TransformerKeyframeSelector(
            pretrained_backbone_path=None, 
            num_tasks=len(NAME_TO_ID),
            max_phases=100, 
            window_size=3,
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("✅ Model loaded successfully.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.k_imgs = []       
        self.k_wrist_imgs = []  
        self.k_states = []
        self.k_indices = []      
        self.current_episode_dir = None
        self.debug_frames_dir = None 
        self.history_buffer = deque(maxlen=3) 
        
        self.current_phase = 0
        self.stop_detection = False 
        self.last_trigger_step = -999 
        
        self.smoothing_window = 5       
        self.is_candidate_pending = False
        self.candidate_data = None  
        self.stable_counter = 0         
        self.cooldown_steps = 0 

    def reset(self, episode_idx: int):
        self.k_imgs = []
        self.k_wrist_imgs = []
        self.k_states = []
        self.k_indices = []
        self.history_buffer.clear()
        
        self.current_phase = 0
        self.stop_detection = False 
        
        self.is_candidate_pending = False
        self.candidate_data = None
        self.stable_counter = 0

        if self.save_root:
            self.current_episode_dir = os.path.join(self.save_root, self.task_name, f"ep_{episode_idx:03d}")
            if os.path.exists(self.current_episode_dir):
                shutil.rmtree(self.current_episode_dir)
            os.makedirs(self.current_episode_dir, exist_ok=True)
            
            self.debug_frames_dir = os.path.join(self.current_episode_dir, "debug_all_frames")
            os.makedirs(self.debug_frames_dir, exist_ok=True)

    def get_inference_context(self):
        imgs = list(self.k_imgs)
        wrists = list(self.k_wrist_imgs)
        states = list(self.k_states)
        indices = list(self.k_indices)

        if self.is_candidate_pending and self.candidate_data is not None:
            c_img, c_wrist, c_state, c_step, c_prob = self.candidate_data
            imgs.append(c_img)
            wrists.append(c_wrist)
            states.append(c_state)
            indices.append(c_step)
            
        return imgs, wrists, states, indices

    def add_keyframe(self, img, wrist_img, state, step_idx: int):
        self.k_imgs.append(img)
        self.k_wrist_imgs.append(wrist_img)
        self.k_states.append(state)
        self.k_indices.append(step_idx)
        
        # print(f"🔥 [KEYFRAME CONFIRMED] Phase {self.current_phase} -> Step {step_idx} | Total KFs: {len(self.k_imgs)}")
        
        self.current_phase += 1
        self.last_trigger_step = step_idx
        
        if self.current_phase >= self.max_keyframes:
            self.stop_detection = True
            # print(f"🛑 [STOP] Reached max keyframes ({self.max_keyframes}). Stopping detection for this episode.")
        
        if self.current_episode_dir is not None:
            try:
                img_to_save = img.astype(np.uint8)
                im = Image.fromarray(img_to_save)
                save_name = f"Confirmed_KF_Phase_{self.current_phase-1:02d}_step_{step_idx:04d}.png"
                save_full_path = os.path.join(self.current_episode_dir, save_name)
                im.save(save_full_path)
            except Exception as e:
                print(f"[Warning] Failed to save keyframe image: {e}")

    def check_and_update(self, obs_pack, step_idx: int):
            curr_img_np, curr_wrist, curr_state = obs_pack
            pil_img = Image.fromarray(curr_img_np.astype(np.uint8))
            self.history_buffer.append(pil_img)
            if len(self.history_buffer) < 3: return

            prob = 0.0
            corrected_phase = max(0, self.current_phase - 1)

            if not self.stop_detection:
                try:
                    window_tensors = [self.transform(img) for img in self.history_buffer]
                    input_imgs = torch.stack(window_tensors).unsqueeze(0).to(self.device)
                    
                    count_tensor = torch.tensor([corrected_phase], dtype=torch.long).to(self.device)
                    task_tensor = torch.tensor([self.task_id], dtype=torch.long).to(self.device)
                    
                    with torch.no_grad():
                        logits = self.model(input_imgs, count_tensor, task_tensor)
                        prob = torch.sigmoid(logits).item()

                except Exception as e:
                    print(f"Error during model inference: {e}")
                    import traceback
                    traceback.print_exc()
            
            if self.debug_frames_dir is not None:
                status_str = "PENDING" if self.is_candidate_pending else "NORMAL"
                if self.stop_detection: status_str = "STOPPED"
                save_name = f"step_{step_idx:04d}_prob_{prob:.4f}_MgrPhase_{self.current_phase}_InputPhase_{corrected_phase}_Status_{status_str}.png"
                save_path = os.path.join(self.debug_frames_dir, save_name)
                Image.fromarray(curr_img_np.astype(np.uint8)).save(save_path)

            if self.stop_detection:
                return

            if prob > self.threshold:
                if not self.is_candidate_pending:
                    print(f"👀 [NEW CANDIDATE] Step {step_idx} (prob={prob:.4f}). Using immediately.")
                else:
                    pass
                
                self.is_candidate_pending = True
                self.candidate_data = (curr_img_np, curr_wrist, curr_state, step_idx, prob)
                self.stable_counter = 0 
            
            else:
                if self.is_candidate_pending:
                    self.stable_counter += 1
                    
                    if self.stable_counter >= self.smoothing_window:
                        c_img, c_wrist, c_state, c_step, c_prob = self.candidate_data
                        # print(f"✅ [LOCKED] Confirming Step {c_step} as Phase {self.current_phase} Keyframe.")
                        
                        self.add_keyframe(c_img, c_wrist, c_state, c_step)
                        self.is_candidate_pending = False
                        self.candidate_data = None
                        self.stable_counter = 0

def construct_inference_input(manager: KeyframeManagerAI, curr_img, curr_wrist, curr_state, queue_len, task_desc, current_step_idx):
    cand_imgs, cand_wrists, cand_states, cand_indices = manager.get_inference_context()
    
    all_imgs = cand_imgs + [curr_img]
    all_wrists = cand_wrists + [curr_wrist]
    all_states = cand_states + [curr_state]
    all_indices = cand_indices + [current_step_idx]
    
    current_len = len(all_imgs)
    
    if current_len >= queue_len:
        final_imgs = all_imgs[-queue_len:]
        final_wrists = all_wrists[-queue_len:]
        final_states = all_states[-queue_len:]
        final_indices = all_indices[-queue_len:]
        obs_mask = [1] * queue_len
    else:
        pad_len = queue_len - current_len
        h, w, c = curr_img.shape
        state_dim = curr_state.shape[0]
        
        zero_img = np.zeros((h, w, c), dtype=curr_img.dtype)
        zero_state = np.zeros((state_dim,), dtype=curr_state.dtype)
        
        final_imgs = [zero_img] * pad_len + all_imgs
        final_wrists = [zero_img] * pad_len + all_wrists 
        final_states = [zero_state] * pad_len + all_states
        
        final_indices = ["PAD"] * pad_len + all_indices
        obs_mask = [0] * pad_len + [1] * current_len

    if current_step_idx % 100 == 0:
        status = "STOPPED" if manager.stop_detection else "RUNNING"
        pending_str = " [PENDING USE]" if manager.is_candidate_pending else ""
        print(f"[DEBUG] Step: {current_step_idx:04d} | Status: {status}{pending_str} | Phase: {manager.current_phase} | Input Indices: {final_indices}")

    input_imgs = np.stack(final_imgs, axis=0)          
    input_wrists = np.stack(final_wrists, axis=0)      
    input_states = np.stack(final_states, axis=0)      
    obs_mask_arr = np.array(obs_mask, dtype=np.int32)

    element = {
        "video.image": input_imgs,
        "video.wrist_image": input_wrists,
        "state.position": input_states[:, :3],
        "state.rotation": input_states[:, 3:7],
        "state.gripper": input_states[:, 7:],
        "annotation.human.task_description": [task_desc],
        "obs_mask": obs_mask_arr.tolist() 
    }
    return element

def main(args: EvalConfig):
    os.makedirs(args.save_path, exist_ok=True)
    if args.keyframe_save_path is None:
        args.keyframe_save_path = os.path.join(os.path.dirname(args.save_path), "keyframes")
    os.makedirs(args.keyframe_save_path, exist_ok=True)
    print(f"Keyframe images will be saved to: {args.keyframe_save_path}")

    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    
    if args.url:
        policy_client = WebSocketInferenceClient(url=args.url)
    else:
        policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    
    kwargs = dict()

    for robot_uids, tasks in TASKS.items():
        total_successes = 0.0
        success_dict = {}
        for env_id in tasks:
            if not args.cpu_sim:
                env = gym.make(env_id, num_envs=num_envs, obs_mode=args.obs_mode, robot_uids=robot_uids, sensor_configs=dict(shader_pack=args.shader), human_render_camera_configs=dict(shader_pack=args.shader), viewer_camera_configs=dict(shader_pack=args.shader), render_mode=args.render_mode, control_mode=args.control_mode, sim_config=sim_config, **kwargs)
                if isinstance(env.action_space, gym.spaces.Dict): env = FlattenActionSpaceWrapper(env)
                base_env = env.unwrapped
            else:
                def make_env():
                    def _init():
                        env = gym.make(env_id, obs_mode=args.obs_mode, sim_config=sim_config, robot_uids=robot_uids, sensor_configs=dict(shader_pack=args.shader), human_render_camera_configs=dict(shader_pack=args.shader), viewer_camera_configs=dict(shader_pack=args.shader), render_mode=args.render_mode, control_mode=args.control_mode, **kwargs)
                        env = CPUGymWrapper(env)
                        return env
                    return _init
                env = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver" if sys.platform == "darwin" else None) if args.num_envs > 1 else make_env()()
                base_env = make_env()().unwrapped

            base_env.print_sim_details()
            
            max_kfs = TASK_MAX_KEYFRAMES.get(env_id, 50)
            # print(f"📌 Task {env_id} will stop detection after finding {max_kfs} keyframes.")

            kf_manager = KeyframeManagerAI(
                task_name=env_id, 
                model_path=args.model_checkpoint,
                save_root=args.keyframe_save_path,
                max_keyframes=max_kfs 
            )

            task_successes = 0.0
            
            for seed in tqdm(range(args.num_per_task)):
                kf_manager.reset(episode_idx=seed)
                images = []
                video_nrows = int(np.sqrt(num_envs))
                terminated = False 

                with torch.inference_mode():
                    env.reset(seed=seed+2025)
                    env.step(env.action_space.sample())
                    obs, info = env.reset(seed=seed+2025)
                    
                    task_description = TASK_INSTRUCTIONS[env_id]
                    img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
                    wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])
                    state = np.concatenate((obs["extra"]["tcp_pose"], obs["agent"]["qpos"][-1:]), axis=0)
                    
                    kf_manager.add_keyframe(img, wrist_img, state, step_idx=0)

                    if args.save_video:
                        images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())

                    step_length = STEP_LENGTHS[env_id]
                    N = step_length // args.replan_steps
                    current_step_global = 0

                    with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
                        for i in range(N):
                            curr_img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
                            curr_wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])
                            curr_state = np.concatenate((obs["extra"]["tcp_pose"], obs["agent"]["qpos"][-1:]), axis=0)
                            
                            element_infer = construct_inference_input(
                                kf_manager, 
                                curr_img, curr_wrist_img, curr_state, 
                                args.queue_len, task_description,
                                current_step_global
                            )
                            
                            action_chunk = policy_client.get_action(element_infer)
                            
                            if "action.position" not in action_chunk:
                                print(f"\n[Error] Server Response Invalid: {action_chunk}")
                                break

                            if 'stick' in robot_uids: 
                                pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation']],axis=1)
                            elif 'widowxai' in robot_uids:
                                pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation'],action_chunk['action.gripper'][:,None],action_chunk['action.gripper'][:,None]],axis=1)
                            else:
                                pred_action = np.concatenate([
                                    action_chunk['action.position'],
                                    action_chunk['action.rotation'],
                                    action_chunk['action.gripper']
                                ], axis=1)
                            
                            pred_action = pred_action[:args.replan_steps]
                            
                            for action in pred_action:
                                current_step_global += 1
                                obs, rew, terminated, truncated, info = env.step(action)
                                
                                step_img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
                                step_wrist = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])
                                step_state = np.concatenate((obs["extra"]["tcp_pose"], obs["agent"]["qpos"][-1:]), axis=0)
                                
                                kf_manager.check_and_update((step_img, step_wrist, step_state), step_idx=current_step_global)
                                
                                if args.save_video:
                                    images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                                
                                terminated = terminated if args.cpu_sim else terminated.item()
                                if terminated:
                                    task_successes += 1
                                    total_successes += 1
                                    break
                            if terminated:
                                break
                    
                    if args.save_video:
                        images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                        images_to_video(
                            images,
                            output_dir=args.save_path,
                            video_name=f"{robot_uids}-{env_id}-{seed}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}--success={terminated}",
                            fps=30,
                        )
                        del images
            env.close()
            print(f"Task Success Rate: {task_successes / args.num_per_task}")
            success_dict[env_id] = task_successes / args.num_per_task
        
        print(f"Total Success Rate: {total_successes / (args.num_per_task * len(tasks))}")
        success_dict['total_success'] = total_successes / (args.num_per_task * len(tasks))
        with open(f"{args.save_path}/{robot_uids}_success_dict.json", "w") as f:
            json.dump(success_dict, f)

if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
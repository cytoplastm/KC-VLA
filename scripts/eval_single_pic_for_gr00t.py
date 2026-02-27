import dataclasses
import logging
import os
import json
import torch
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import imageio
import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from service import ExternalRobotInferenceClient

import argparse

import matplotlib.pyplot as plt

from collections import deque

class RowQueue:
    def __init__(self):
        # 使用 deque 作为底层容器，popleft() 的时间复杂度为 O(1)
        self.queue = deque()

    def put(self, batch_numpy):
        """存入形状为 (16, 7) 的 numpy"""
        # 将 (16, 7) 拆分为 16 个 (1, 7) 的数组并依次入队
        # np.vsplit 或直接遍历均可，遍历最直观
        for i in range(batch_numpy.shape[0]):
            self.queue.append(batch_numpy[i:i+1, :])

    def get(self):
        """取出形状为 (1, 7) 的 numpy"""
        if self.queue:
            return self.queue.popleft()
        return None # 或者抛出异常
    
    def is_empty(self):
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)

def plot_prediction_vs_groundtruth(gt: np.ndarray, pred: np.ndarray, save_path: str):
    """
    将真实值和推理值随时间变化的关系画图并保存。
    
    参数:
        gt (np.ndarray): shape 为 (n, d) 的真实值
        pred (np.ndarray): shape 为 (n, d) 的推理值
        save_path (str): 图像保存路径，如 'output/pred_vs_gt.png'
    """
    assert gt.shape == pred.shape, "gt 和 pred 的 shape 必须相同"
    n, d = gt.shape

    # 创建输出目录（如有必要）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建图像
    fig, axes = plt.subplots(d, 1, figsize=(10, 2 * d), sharex=True)

    for i in range(d):
        axes[i].plot(range(n), gt[:, i], label='Ground Truth', color='blue', linestyle='-')
        axes[i].plot(range(n), pred[:, i], label='Prediction', color='red', linestyle='--')
        axes[i].set_ylabel(f'Dim {i+1}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Step')

    plt.suptitle('Ground Truth vs Prediction Over Time (All Dimensions)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    plt.savefig(save_path, dpi=300)
    plt.close()

@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 5001
    resize_size: int = 224
    replan_steps: int = 5

    # Dataset parameters
    dataset_path: str = "/home/chenyipeng/data/real_robot_data_process/swap_three_cubes"

    # Utils
    seed: int = 7 

    task_description: str = "Swap the position of the bottom and middle cubes"

def eval_dp(args: Args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = RowQueue()
    
    # 加载数据集
    dataset = LeRobotDataset(args.dataset_path)
    episode_data_index = dataset.episode_data_index
    from_indices = episode_data_index['from']
    to_indices = episode_data_index['to']
    # 初始化推理客户端
    policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    policy_client.set_robot_uid('piper') 

    start = from_indices[3]
    end = to_indices[3]
    gt_actions = []
    pred_actions = []
    for ep_idx in range(start, end):
        data = dataset[ep_idx]
        image = (data['observation.images.image'].permute(1, 2, 0).unsqueeze(0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        wrist_image = (data['observation.images.wrist_image'].permute(1, 2, 0).unsqueeze(0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        state = data['observation.state'].to('cuda').unsqueeze(0).cpu().numpy()
        gt_action = data['action']
        element = {
            "video.image": image,
            "video.wrist_image": wrist_image,
            "state": state,
            "annotation.human.task_description": [args.task_description],
        }
        if q.is_empty():
            action_chunk = policy_client.get_action(element)
            pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation'],action_chunk['action.gripper'][:,None]],axis=1) 
            # q.put(pred_action[-args.replan_steps:])
            q.put(pred_action[:args.replan_steps])
        gt_actions.append(gt_action.cpu().numpy())
        pred_actions.append(q.get().squeeze(0))
    gt_actions = np.array(gt_actions)
    pred_actions =  np.array(pred_actions)
    plot_prediction_vs_groundtruth(gt_actions, pred_actions, save_path="/home/chenyipeng/my-Isaac-GR00T")


def main():
    parser = argparse.ArgumentParser(description="Eval DP Policy")
    default_args = Args()

    parser.add_argument("--host", type=str, default=default_args.host, help="Server host")
    parser.add_argument("--port", type=int, default=default_args.port, help="Server port")
    parser.add_argument("--resize-size", type=int, default=default_args.resize_size)
    parser.add_argument("--replan-steps", type=int, default=default_args.replan_steps)
    parser.add_argument("--dataset-path", type=str, default=default_args.dataset_path)
    parser.add_argument("--seed", type=int, default=default_args.seed)

    parsed = parser.parse_args()
    args = Args(**vars(parsed))
    
    eval_dp(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
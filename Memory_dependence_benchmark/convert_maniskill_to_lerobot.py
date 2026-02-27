"""
Script for converting a ManiSkill dataset to LeRobot format.
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/home/chenyipeng/data/maniskill_data/lerobot_datasets/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/chenyipeng/lerobot").resolve()))

import shutil
import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
from pathlib import Path
import re
from mani_skill.utils import sapien_utils
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json  # 用于加载 JSON 文件

# REPO = "AllTasks-v5"
# ROBOT_UID_LIST = [
#     "panda_wristcam",
#     # "panda_stick_wristcam",
#     "xarm6_robotiq_wristcam",
#     # "xarm6_stick_wristcam",
#     "widowxai_wristcam",
#     "xarm7_robotiq_wristcam",
# ]
# ROBOT_LIST = {
#     "panda_wristcam": "panda",
#     "panda_stick_wristcam": "panda",
#     "xarm6_robotiq_wristcam": "xarm6",
#     "xarm6_stick_wristcam": "xarm6",
#     "widowxai_wristcam": "widowxai",
#     "xarm7_robotiq_wristcam": "xarm7",
# }
# TASK_LIST = {
#     "panda_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "xarm6_robotiq_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "panda_stick_wristcam": ["DrawTriangle-v1", "DrawSVG-v1"],
#     "xarm6_stick_wristcam": ["DrawTriangle-v1", "DrawSVG-v1"],
#     "widowxai_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "xarm7_robotiq_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
# }
# TASK_INSTRUCTION_LIST = {
#     "PickCube-v1": "Pick up the cube.",
#     "PullCube-v1": "Pull the cube to the target position.",
#     "PushCube-v1": "Push the cube to the target position.",
#     "StackCube-v1": "Stack the cube on top of the other cube.",
#     "PullCubeTool-v1": "Pick up the cube tool and use it to bring the cube closer.",
#     "PlaceSphere-v1": "Pick up the ball and place it in the target position.",
#     "LiftPegUpright-v1": "Pick up the peg and place it upright.",
#     "PegInsertionSide-v1": "Pick up the peg and insert it into the container next to the peg.",
#     "DrawTriangle-v1": "Draw a triangle according to the specified trajectory.",
#     "DrawSVG-v1": "Draw a svg graph according to the specified trajectory."
# }

#SwapThreeCubes-v1 PickPlaceThreetimes-v1 PushCubeWithSignal-v1 Identification-v1 TeacherArmShuffle-v1

REPO = "mixed_data"
ROBOT_UID_LIST = ["panda_wristcam"]
ROBOT_LIST = {"panda_wristcam": "panda"}
TASK_LIST = {"panda_wristcam": ["TeacherArmShuffle-v1","SwapThreeCubes-v1","PushCubeWithSignal-v1","PickPlaceThreetimes-v1"]}
TASK_INSTRUCTION_LIST = {
    "SwapThreeCubes-v1": "Swap the position of the bottom and middle cubes.",
    "PickPlaceThreetimes-v1": "First, pick up the red cube and place it back on the table. Next, do the same for the green cube. Finally, the blue cube.",
    # "PushCubeWithSignal-v1": "Wait for the signal light to turn on and then off, then push the cube to the target.",
    "PushCubeWithSignal-v1": "Wait for the signal light to flash twice, then push the cube to the target.",
    "TeacherArmShuffle-v1": "After the cubes are swapped, pick up the cube that was originally in the middle.",
}

def load_h5_data(data):
    """
    Recursively load all HDF5 datasets into memory.
    """
    out = {}
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def main(push_to_hub: bool = False, load_count: int = 100):
    """
    将 ManiSkill 数据集转换为 LeRobot 格式并保存到 $HF_LEROBOT_HOME 下。

    参数:
      - dataset_file: ManiSkill 数据集的 .h5 文件路径。
      - push_to_hub: 是否将转换后的数据集推送到 Hugging Face Hub。
      - load_count: 加载的轨迹数量，-1 表示加载所有轨迹。
    """
    for robot_name in ROBOT_UID_LIST:
        robot = ROBOT_LIST[robot_name]
        # 设置输出仓库名称（同时也作为 Hugging Face Hub 上的 repo_id）
        REPO_NAME = f"{REPO}/{robot_name}"  # 请替换为你的 Hugging Face 用户名及仓库名称

        # 清理已存在的输出目录
        output_path = HF_LEROBOT_HOME / REPO_NAME
        if output_path.exists():
            shutil.rmtree(output_path)

        # 创建 LeRobot 数据集，定义各项特征。注意这里的 shape 需与你数据的实际形状匹配
        if robot_name in ["panda_stick_wristcam", "xarm6_stick_wristcam"]:
            action_dim = 6
            dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root=output_path,
                robot_type=robot_name,
                use_videos=True, 
                video_backend=None,  
                fps=10,
                features={
                    "observation.images.image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.images.wrist_image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (7,),  # 根据实际情况调整状态向量的维度
                        "names": ["state"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (6,),  # 根据实际情况调整动作向量的维度
                        "names": ["actions"],
                    },
                },
                image_writer_threads=24,
                image_writer_processes=12,
            )
        # elif robot_name in ["panda_wristcam", "xarm6_robotiq_wristcam"]:
        else:
            action_dim = 7
            dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root=output_path,
                robot_type=robot_name,
                use_videos=True,  
                video_backend=None,  
                fps=10,
                features={
                    "observation.images.image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.images.wrist_image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (8,),  # 根据实际情况调整状态向量的维度
                        "names": ["state"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),  # 根据实际情况调整动作向量的维度
                        "names": ["action"],
                    },
                },
                image_writer_threads=24,
                image_writer_processes=12,
            )

        task_list = TASK_LIST[robot_name]
        for task in task_list:
            dataset_file = f"/home/chenyipeng/ManiSkill/data/{robot}/{task}/motionplanning/trajectory_{robot}.rgb.pd_ee_delta_pose.physx_cpu.h5"
            # 打开 ManiSkill 数据集的 h5 文件和对应的 JSON 文件
            data = h5py.File(dataset_file, "r")
            path_obj = Path(dataset_file)
            task_name = path_obj.parent.parent.name  # set_table
            action_name = path_obj.parent.name  # pick
            # 获取文件名（不含扩展名），这里为 "013_apple"
            file_stem = path_obj.stem
            # 使用正则表达式移除文件名前缀的数字及下划线，得到 "apple"
            obj_name = re.sub(r"^\d+_", "", file_stem)
            task_desc = f"{task_name}: {action_name}_{obj_name}" # set_table: pick_apple

            json_file = dataset_file.replace(".h5", ".json")
            json_data = load_json(json_file)
            episodes = json_data["episodes"]


            if load_count == -1 or load_count > len(episodes):
                load_count = len(episodes) # load_count = 200

            print(f"Loading {load_count} episodes from {dataset_file}")
            # 遍历每个轨迹（episode）
            for eps_id in range(load_count):
            # for eps_id in range(100):
                eps = episodes[eps_id]
                traj_key = f"traj_{eps['episode_id']}"
                if traj_key not in data:
                    continue

                # 加载轨迹数据（转换为内存中的 numpy 数组） 数据已经都是ndarray
                trajectory = data[traj_key]
                trajectory = load_h5_data(trajectory)
                actions = np.array(trajectory["actions"], dtype=np.float32)
                # 假设所有观察数据均存储在 "obs" 下，并包含所需的键
                # obs: [T,Dict]
                obs = trajectory["obs"]
                # print("obs_keys:",obs)
                qposs = np.array(obs["agent"]["qpos"],dtype=np.float32) # [T+1, 12]
                # qvels = np.array(obs["agent"]["qvel"],dtype=np.float32) # [T+1, 12]
                states = np.array(obs["extra"]["tcp_pose"],dtype=np.float32) # [T+1, 7]
                gripper = qposs[:,-1:]

                if robot_name in ["panda_stick_wristcam", "xarm6_stick_wristcam"]:
                    pass
                # elif robot_name in ["panda_wristcam", "xarm6_robotiq_wristcam"]:
                else:
                    states = np.concatenate((states, gripper), axis=1)

                eps_len = len(actions)
                print(f"Processing episode {eps_id} with {eps_len} frames")

                images = obs["sensor_data"]["third_view_camera"]["rgb"]
                wrist_images = obs["sensor_data"]["hand_camera"]["rgb"]
                img_dim = (len(wrist_images),len(wrist_images[0]),len(wrist_images[0][0]),len(wrist_images[0][0][0]))

                # exit()
                # 遍历轨迹中的每一帧
                for i in range(eps_len):
                    if "third_view_camera" in obs["sensor_data"]:
                        tmp_image = images[i]
                    else:
                        tmp_image = np.zeros((256, 256, 3), dtype=np.uint8)

                    if "hand_camera" in obs["sensor_data"]:
                        tmp_wrist_image = wrist_images[i]
                    else:
                        tmp_wrist_image = np.zeros((256, 256, 3), dtype=np.uint8)
                        raise ValueError(f"Expected actions shape ({action_dim},), got {actions.shape}")

                    if "agent" in obs:
                        tmp_state = states[i]
                        # print("state:",tmp_state)
                        # tmp_state_v = qvels[i]
                    else:
                        tmp_state = np.zeros(action_dim, dtype=np.float32)
                        raise ValueError(f"Expected actions shape ({action_dim},), got {actions.shape}")
                        # tmp_state_v = np.zeros(12, dtype=np.float32)

                    # 获取动作数据
                    tmp_action = actions[i]
                    if tmp_action.shape == (8,) and robot_name in ["widowxai_wristcam"]:
                        tmp_action = tmp_action[:7]
                    if tmp_action.shape != (action_dim,):
                        # print(tmp_action.shape)
                        raise ValueError(f"Expected actions shape ({action_dim},), got {actions.shape}")
                    # print("type of tmp_state:", type(tmp_state))
                    # print("type of actions:", type(actions))


                    # 将当前帧数据添加到 LeRobot 数据集中
                    # print(tmp_action)
                    dataset.add_frame(
                        frame={
                        "observation.images.image": tmp_image,
                        "observation.images.wrist_image": tmp_wrist_image,
                        "observation.state": tmp_state,
                        "action": tmp_action,
                        "task": TASK_INSTRUCTION_LIST[task_name]
                        },
                    )

                dataset.save_episode()

    # 整理数据集（合并所有帧并生成索引，统计信息可后续再计算）
    # dataset.consolidate(run_compute_stats=False)

    # 可选：推送数据集到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["maniskill"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
    print("Transformation Done!")

    # from huggingface_hub import HfApi
    # from huggingface_hub import create_tag
    #
    # api = HfApi(token=os.getenv("HF_TOKEN"))
    # api.upload_folder(
    #     folder_path="/home/wangzhibin/openpi-main/data/AllTasks-v1/xarm6_stick_wristcam",
    #     repo_id="Johnathan218/xarm6_stick_wristcam",
    #     repo_type="dataset",
    # )
    # create_tag(
    #     repo_id="Johnathan218/xarm6_stick_wristcam",
    #     tag="v2.0",
    #     repo_type="dataset",
    # )
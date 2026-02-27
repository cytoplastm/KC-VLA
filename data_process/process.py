import os
import json
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

dataset_path = '/home/chenyipeng/data/test0123_cyp/real_robot_data_teacher_arm_shuffle'
save_path = '/home/chenyipeng/data/real_robot_data_process/teacher_arm_shuffle'

def extract_state(state_list):
    state_list = state_list.copy()
    return state_list[6:]    # 末端位姿（6维）+ 夹爪宽度（1维）+ 夹爪力矩（1维）

def extract_action(action_list):
    action_list = action_list.copy()
    return action_list[6:]
    
# os.system(f'rm -r {save_path}')
# os.system(f'mkdir -p {save_path}')
# os.system(f'cp -r {dataset_path}/*  {save_path}')

if os.path.exists(save_path):
    # 抛出异常，终止程序
    raise FileExistsError(f"\n[错误] 目标路径已存在，为了防止误删，程序已停止。\n请手动删除或更改路径: {save_path}")
else:
    # 创建目录
    os.makedirs(save_path)
    # 复制数据 (注意：cp -r src/* dest 需要 dest 存在)
    os.system(f'cp -r {dataset_path}/* {save_path}')

parquet_files_path = os.path.join(save_path, 'data/chunk-000')
for file in os.listdir(parquet_files_path):
    if file.endswith('.parquet'):
        file_path = os.path.join(parquet_files_path, file)
        df = pd.read_parquet(file_path)
        # 修改state和action
        df['observation.state']= df['observation.state'].apply(extract_state)
        df["action"] = df["action"].apply(extract_action)
        df.to_parquet(file_path)

# 修改统计量
stats_save = []
stats_path = os.path.join(save_path, 'meta/episodes_stats.jsonl')
with open(stats_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue  # 跳过空行
        obj = json.loads(line)  # 每行是一个独立的 JSON 对象
        parqurt_file = os.path.join(save_path, 'data/chunk-000', f"episode_{obj['episode_index']:06d}" + '.parquet')
        df = pd.read_parquet(parqurt_file)
        state_array = np.stack(df["observation.state"].to_numpy())
        action_array = np.stack(df["action"].to_numpy())
        obj['stats']['observation.state']['max'] = np.max(state_array,axis=0).tolist()
        obj['stats']['observation.state']['min'] = np.min(state_array,axis=0).tolist()
        obj['stats']['observation.state']['mean'] = np.mean(state_array,axis=0).tolist()
        obj['stats']['observation.state']['std'] = np.std(state_array,axis=0).tolist()
        obj['stats']['action']['max'] = np.max(action_array,axis=0).tolist()
        obj['stats']['action']['min'] = np.min(action_array,axis=0).tolist()
        obj['stats']['action']['mean'] = np.mean(action_array,axis=0).tolist()
        obj['stats']['action']['std'] = np.std(action_array,axis=0).tolist()
        stats_save.append(obj)

with open(stats_path, "w", encoding="utf-8") as f:
    for obj in stats_save:
        json_line = json.dumps(obj, ensure_ascii=False)
        f.write(json_line + "\n")
        
def modify_state(state_list):
    position = state_list[0:3] / 1000000   # 0.001mm -> 1m
    rotation = state_list[3:6] / 1000 * np.pi / 180   # 0.001deg -> rad
    gripper = state_list[6:7] / 1000000   # 0.001mm -> 1m
    gripper_force = state_list[7:8]  # 保持不变
    return np.concatenate([position, rotation, gripper, gripper_force])

def calculate_transform(A, B):
    """
    计算B相对于A的变化量
    
    参数:
    A: shape (n,8) 的numpy数组，包含[x,y,z,roll,pitch,yaw,width,force]
    B: shape (n,8) 的numpy数组，包含[x,y,z,roll,pitch,yaw,width,force]
    
    返回:
    shape (n,8) 的numpy数组，包含变化量[x,y,z,roll,pitch,yaw,width,force]
    """
    # 确保输入形状正确
    assert A.shape == B.shape, "A和B的形状必须相同"
    assert A.shape[1] == 8, "输入数组的第二维必须是8"
    
    # 分离各个分量
    A_xyz = A[:, :3]
    A_rpy = A[:, 3:6]
    A_width = A[:, 6:7]
    A_force = A[:, 7:8]
    
    B_xyz = B[:, :3]
    B_rpy = B[:, 3:6]
    B_width = B[:, 6:7]
    B_force = B[:, 7:8]
    
    # 计算xyz的变化量
    delta_xyz = B_xyz - A_xyz
    
    # 计算rpy角的变化量
    # 将rpy角转换为旋转矩阵
    R_A = Rotation.from_euler('xyz', A_rpy).as_matrix()
    R_B = Rotation.from_euler('xyz', B_rpy).as_matrix()
    
    delta_R = R_A.transpose(0, 2, 1) @ R_B
    
    # 将旋转矩阵转换回rpy角
    delta_rpy = Rotation.from_matrix(delta_R).as_euler('xyz')
    
    # 组合所有变化量
    delta = np.concatenate([delta_xyz, delta_rpy, B_width, B_force], axis=1)
    
    return delta

# ================= 步骤 1 : 处理 Parquet 数据 =================
parquet_files_path = os.path.join(save_path, 'data/chunk-000')
for file in os.listdir(parquet_files_path):
    if file.endswith('.parquet'):
        file_path = os.path.join(parquet_files_path, file)
        df = pd.read_parquet(file_path)
        # 修改state和action
        # 不修改state的夹爪宽度，只修改action的夹爪宽度
        df['observation.state'] = df['observation.state'].apply(modify_state)
        state_array = np.stack(df["observation.state"].to_numpy())
        action_array = np.concatenate([state_array[1:],state_array[-1:]])

        action_array = calculate_transform(state_array, action_array)  # 计算相对动作
        df["action"] = [row for row in action_array]
        # 根据夹爪力矩处理夹爪宽度,只修改action,不修改action
        df['observation.state'] = df['observation.state'].apply(lambda x: x[:-1])
        df['action'] = df['action'].apply(
            lambda x: np.append(x[:6], 0.0001) if x[-1] < -900 else x[:-1]
        )
        df.to_parquet(file_path)

# ================= 步骤 2: 更新统计量 (Stats) =================
stats_save = []
stats_path = os.path.join(save_path, 'meta/episodes_stats.jsonl')
with open(stats_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue  # 跳过空行
        obj = json.loads(line)  # 每行是一个独立的 JSON 对象
        parqurt_file = os.path.join(save_path, 'data/chunk-000', f"episode_{obj['episode_index']:06d}" + '.parquet')
        df = pd.read_parquet(parqurt_file)
        state_array = np.stack(df["observation.state"].to_numpy())
        action_array = np.stack(df["action"].to_numpy())
        obj['stats']['observation.state']['max'] = np.max(state_array,axis=0).tolist()
        obj['stats']['observation.state']['min'] = np.min(state_array,axis=0).tolist()
        obj['stats']['observation.state']['mean'] = np.mean(state_array,axis=0).tolist()
        obj['stats']['observation.state']['std'] = np.std(state_array,axis=0).tolist()
        obj['stats']['action']['max'] = np.max(action_array,axis=0).tolist()
        obj['stats']['action']['min'] = np.min(action_array,axis=0).tolist()
        obj['stats']['action']['mean'] = np.mean(action_array,axis=0).tolist()
        obj['stats']['action']['std'] = np.std(action_array,axis=0).tolist()
        stats_save.append(obj)

with open(stats_path, "w", encoding="utf-8") as f:
    for obj in stats_save:
        json_line = json.dumps(obj, ensure_ascii=False)
        f.write(json_line + "\n")
        
# ================= 步骤 3: 修改 info.json =================
info_path = os.path.join(save_path, 'meta/info.json')

if os.path.exists(info_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info_data = json.load(f)
        
    # 获取原始名称列表
    original_action_names = info_data['features']['action']['names']
    original_state_names = info_data['features']['observation.state']['names']
    
    # 修改 action
    if 'features' in info_data and 'action' in info_data['features']:
        info_data['features']['action']['shape'] = [7]
        info_data['features']['action']['names'] = original_action_names[6:-1]
        
    # 修改 observation.state
    if 'features' in info_data and 'observation.state' in info_data['features']:
        info_data['features']['observation.state']['shape'] = [7]
        info_data['features']['observation.state']['names'] = original_state_names[6:-1]

    # 保存修改后的 info.json
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=4, ensure_ascii=False)
else:
    print("警告: 未找到 info.json，跳过元数据更新。")
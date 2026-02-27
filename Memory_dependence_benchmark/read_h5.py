import h5py
import numpy as np

# 文件路径
h5_path = "/home/chenyipeng/ManiSkill/data/panda/TeacherArmShuffle-v1/motionplanning/trajectory_panda.rgb.pd_ee_delta_pose.physx_cpu.h5"

def print_structure(name, obj):
    """递归打印函数"""
    # 计算缩进层级
    depth = name.count('/')
    indent = "  " * depth
    
    # 获取对象名称（去除路径前缀）
    obj_name = name.split('/')[-1]
    
    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 [{obj_name}] (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 {obj_name}: Shape={obj.shape}, Type={obj.dtype}")

try:
    with h5py.File(h5_path, 'r') as f:
        print(f"=== 文件路径: {h5_path} ===")
        keys = list(f.keys())
        print(f"=== 总轨迹数: {len(keys)} 条 ===")
        print(f"=== 轨迹 ID 列表 (前5个): {keys[:5]} ... ===\n")

        if len(keys) > 0:
            first_traj = keys[0]
            print(f"=== 正在详情展示第一条轨迹: {first_traj} ===")
            
            # 使用 visititems 遍历 traj_0 下的所有内容
            # 这里的 lambda 是为了把 visititems 的输出重定向到我们的打印函数
            f[first_traj].visititems(print_structure)
            
            # 额外检查一下 actions 的具体数值范围（可选）
            actions = f[first_traj]['actions'][:]
            print(f"\n--- 动作数据预览 (前2帧) ---")
            print(actions[:2])

except FileNotFoundError:
    print("❌ 错误：找不到文件，请检查路径是否正确。")
except Exception as e:
    print(f"❌ 发生错误: {e}")
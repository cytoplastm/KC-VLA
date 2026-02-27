import os
import torch
import numpy as np
from safetensors.torch import load_file

def check_history_weights(ckpt_dir):
    print(f"🔍 正在扫描 Checkpoint 目录: {ckpt_dir}")
    
    # 1. 找到所有 safetensors 文件
    shard_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')]
    shard_files.sort()
    
    if not shard_files:
        print("❌ 未找到 .safetensors 文件！")
        return

    print(f"📦 发现 {len(shard_files)} 个权重分片文件。开始搜索关键参数...\n")

    # 需要检查的关键参数名 (模糊匹配)
    targets = [
        "proj_history_norm.weight",  # 【关键】Zero-Init 的那个参数
        "proj_history_fc.weight",    # Small-Init 的线性层
        "history_query.query_tokens" # History Attention 的 Query
    ]

    found_keys = set()

    # 2. 遍历所有分片
    for shard_name in shard_files:
        shard_path = os.path.join(ckpt_dir, shard_name)
        # print(f"正在读取: {shard_name} ...")
        
        try:
            state_dict = load_file(shard_path) # 使用 safetensors 加载
        except Exception as e:
            print(f"⚠️ 读取 {shard_name} 失败: {e}")
            continue

        for key in state_dict.keys():
            # 检查这个 key 是否是我们关注的
            for target in targets:
                if target in key:
                    found_keys.add(key)
                    param = state_dict[key].float().numpy()
                    
                    abs_mean = np.mean(np.abs(param))
                    max_val = np.max(param)
                    min_val = np.min(param)
                    
                    print(f"👉 参数名: {key}")
                    print(f"   📂 所在文件: {shard_name}")
                    print(f"   - 形状: {param.shape}")
                    print(f"   - 均值 (Mean Abs): {abs_mean:.6e}")
                    print(f"   - 范围 (Min/Max):  [{min_val:.6e}, {max_val:.6e}]")
                    
                    # 专门针对 Zero-Init 的判断
                    if "proj_history_norm.weight" in key:
                        print("   ------ [Zero-Init 检查] ------")
                        if abs_mean == 0.0 and max_val == 0.0:
                            print("   🔴 警告: 权重依然是纯 0！History 分支完全没学到东西！")
                        elif abs_mean < 1e-4:
                            print("   🟡 提示: 权重非常小。模型可能刚开始学，或者 History 作用很小。")
                        else:
                            print("   🟢 正常: 权重已更新 (非 0)，History 分支正在发挥作用。")
                    print("-" * 50)

    if not found_keys:
        print("❌ 未在任何分片中找到目标参数。请检查参数名称是否正确。")

if __name__ == "__main__":
    # 你的 checkpoint 目录
    CKPT_DIR = "/data1/chenyipeng/Isaac-GR00T/pick_place_three_historyframe=keyframe3_modify_correct_version2" 
    
    check_history_weights(CKPT_DIR)
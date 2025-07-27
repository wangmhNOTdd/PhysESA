import pickle
import torch
import os
import random

def check_sample(data):
    """检查单个数据样本的多尺度映射关系。"""
    print(f"--- 正在检查复合物: {data.complex_id} ---")
    
    num_atomic_nodes = data.num_nodes
    num_coarse_nodes = data.num_coarse_nodes
    atom_to_coarse_map = data.atom_to_coarse_idx
    
    print(f"原子图节点数: {num_atomic_nodes}")
    print(f"粗粒度图节点数: {num_coarse_nodes}")
    print(f"映射张量 'atom_to_coarse_idx' 的长度: {len(atom_to_coarse_map)}")
    
    # 验证1：映射张量的长度是否等于原子节点数
    if len(atom_to_coarse_map) == num_atomic_nodes:
        print("✅ [通过] 映射张量长度正确。")
    else:
        print(f"❌ [失败] 映射张量长度 ({len(atom_to_coarse_map)}) 与原子节点数 ({num_atomic_nodes}) 不匹配！")
        return False

    # 验证2：映射张量中的索引是否有效
    if num_coarse_nodes > 0:
        max_mapped_idx = torch.max(atom_to_coarse_map)
        if max_mapped_idx == num_coarse_nodes - 1:
            print("✅ [通过] 映射张量的最大索引值正确。")
        else:
            print(f"❌ [失败] 映射张量最大索引 ({max_mapped_idx}) 与预期 ({num_coarse_nodes - 1}) 不符！")
            return False
    elif len(atom_to_coarse_map) > 0:
         print(f"❌ [失败] 粗粒度节点数为0，但映射张量不为空！")
         return False
    else: # num_coarse_nodes is 0 and map is empty
        print("✅ [通过] 粗粒度节点数和映射张量都为空，情况一致。")


    print(f"前20个原子到粗粒度节点的映射: {atom_to_coarse_map[:20].tolist()}")
    print("-" * 30 + "\n")
    return True

def main():
    data_path = './experiments/stage2/train.pkl'
    if not os.path.exists(data_path):
        print(f"错误: 数据文件未找到 at '{data_path}'")
        print("请先运行 'experiments/stage2/prepare_stage2_data.py'")
        return

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
        
    if not dataset:
        print("错误: 数据集为空。")
        return
        
    print(f"成功加载数据集，总样本数: {len(dataset)}")
    
    # 随机抽取5个样本进行检查
    num_samples_to_check = min(5, len(dataset))
    samples_to_check = random.sample(dataset, num_samples_to_check)
    
    all_passed = True
    for sample in samples_to_check:
        if not check_sample(sample):
            all_passed = False
            
    if all_passed:
        print("\n🎉 所有随机抽样的样本均通过检查！")
    else:
        print("\n⚠️ 部分样本未通过检查，请查看上面的日志。")

if __name__ == "__main__":
    main()

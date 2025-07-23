# 阶段一问题修复总结

## 问题描述
1. **验证集缺失**: 数据预处理没有生成`valid.pkl`文件
2. **连边方式改进**: 需要从截断半径改为KNN连边

## 解决方案

### 1. 验证集生成修复
**问题原因**: 预定义分割文件中没有验证集字段，或验证集为空

**修复方法**:
```python
# 在prepare_stage1_data.py中添加逻辑
val_ids = split_data.get('val', split_data.get('valid', []))  # 兼容不同命名

# 如果预定义分割中没有验证集，从训练集中分出一部分
if len(val_ids) == 0 and len(train_ids) > 0:
    print("预定义分割中没有验证集，从训练集分出10%作为验证集")
    n_val = max(1, len(train_ids) // 10)
    val_ids = train_ids[-n_val:]
    train_ids = train_ids[:-n_val]
```

### 2. KNN连边实现
**优势**: 
- 每个原子有固定数量的邻居(k=16)
- 避免大分子边数过多的问题
- 训练更稳定

**实现方法**:
```python
def build_knn_edges(self, positions: torch.Tensor, k: int):
    # 计算距离矩阵
    distance_matrix = torch.norm(pos_expanded_i - pos_expanded_j, dim=2)
    
    # 排除自身，找到k个最近邻
    distance_matrix.fill_diagonal_(float('inf'))
    _, knn_indices = torch.topk(distance_matrix, k, dim=1, largest=False)
    
    # 构建边索引
    source_indices = torch.arange(num_atoms).unsqueeze(1).expand(-1, k)
    edge_indices = torch.stack([source_indices.flatten(), knn_indices.flatten()], dim=0).t()
    
    return edge_indices, edge_distances
```

### 3. 使用新的修复版本

#### 快速测试修复效果
```bash
# 清理旧数据并测试
bash test_fix.sh
```

#### 重新运行数据预处理
```bash
# 删除旧文件
rm -f ./experiments/stage1/*.pkl ./experiments/stage1/metadata.json

# 重新生成（使用KNN）
python prepare_stage1_data.py \
    --data_root ./datasets/pdbbind \
    --output_dir ./experiments/stage1 \
    --use_knn --k 16 \
    --test_run
```

#### 验证修复结果
```bash
python check_data.py
```

#### 运行训练
```bash
bash run_stage1.sh test
```

## 修复验证

成功修复后应该看到：
```
✅ train.pkl
✅ valid.pkl  # 现在应该存在
✅ test.pkl
✅ metadata.json

连边方式: knn
K值: 16
验证集已正确生成
```

## 参数说明

### 新增参数
- `--use_knn`: 使用KNN连边（默认启用）
- `--k 16`: KNN的k值
- `--use_radius`: 强制使用截断半径连边

### 数据格式变化
- 边数更加稳定（约等于 节点数 × k）
- 元数据包含连边方式信息
- 确保训练/验证/测试三个数据集都存在

## 预期结果

使用KNN连边后，边数分布应该更加稳定：
- 原来: 边数范围很大 [9940, 967522]
- 现在: 边数 ≈ 节点数 × 16，分布更均匀

这将显著提高训练的稳定性和效率。

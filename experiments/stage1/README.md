# 阶段一实验指南

## 概述

阶段一的目标是搭建一个能跑通的最小化可行产品（MVP），验证ESA框架在蛋白质-小分子亲和力预测任务上的基本可行性。

## 特征简化策略

根据Readme.md中的实验设计，阶段一采用以下简化策略：

### 原子特征（节点特征）
- **只使用原子类型**：采用独热编码表示常见的生物元素（C, N, O, S, P, F, Cl, Br, I, Other）
- **特征维度**：10维

### 边特征
- **只使用距离信息**：通过高斯基函数扩展距离特征
- **特征维度**：16维（16个高斯基函数）
- **暂时不包含**：方向特征和电荷相互作用特征

### 模型架构
- **编码器**：只使用MAB层（掩码自注意力），不使用SAB层
- **层数**：2-4层MAB
- **池化**：标准PMA池化
- **目标**：观察损失是否能正常下降，预测是否显著优于随机猜测

## 使用方法

### 快速开始

在Linux服务器上运行：

```bash
# 完整训练
bash run_stage1.sh

# 测试运行（快速验证）
bash run_stage1.sh test
```

在Windows上运行：

```cmd
# 完整训练
run_stage1.bat

# 测试运行
run_stage1.bat test
```

### 手动运行

#### 1. 数据预处理

```bash
python prepare_stage1_data.py \
    --data_root ./datasets/pdbbind \
    --output_dir ./experiments/stage1 \
    --split_type scaffold_split \
    --cutoff_radius 5.0 \
    --num_gaussians 16
```

可选参数：
- `--max_samples 100`: 限制样本数量（用于快速测试）
- `--test_run`: 测试运行模式，只处理50个样本

#### 2. 模型训练

```bash
python ./experiments/stage1/train_stage1.py \
    --data_dir ./experiments/stage1 \
    --output_dir ./experiments/stage1/checkpoints \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 100
```

可选参数：
- `--test_run`: 测试运行，只训练5个epochs
- `--config config.json`: 使用自定义配置文件

## 输出文件结构

```
experiments/stage1/
├── train.pkl           # 训练数据
├── valid.pkl           # 验证数据  
├── test.pkl            # 测试数据
├── metadata.json       # 数据集元信息
├── checkpoints/        # 模型检查点
│   ├── config.json     # 训练配置
│   ├── stage1-*.ckpt   # 模型检查点
│   └── stage1_logs/    # TensorBoard日志
└── train_stage1.py     # 训练脚本
```

## 数据格式

### 输入数据格式
每个样本包含以下字段：
- `edge_representations`: [num_edges, feature_dim] - 边表示矩阵
- `edge_index`: [2, num_edges] - 边索引
- `affinity`: float - 结合亲和力标签
- `num_edges`: int - 边数量
- `num_nodes`: int - 节点数量
- `complex_id`: str - 复合物ID

### 特征维度
- **输入特征维度**: 36维 (节点特征10×2 + 边特征16)
- **节点特征**: 10维（原子类型独热编码）
- **边特征**: 16维（高斯基函数扩展距离）

## 模型配置

### 默认配置
```json
{
  "graph_dim": 128,
  "hidden_dims": [128, 128],
  "num_heads": 8,
  "layer_types": ["M", "M"],
  "batch_size": 8,
  "learning_rate": 1e-4,
  "max_epochs": 100
}
```

### ESA架构
- **apply_attention_on**: "edge" (使用边注意力)
- **layer_types**: ["M", "M"] (只使用MAB层)
- **num_sabs**: 0 (不使用SAB层)

## 成功标准

阶段一的成功标准：

1. **代码能够运行**：数据预处理和模型训练流程无错误
2. **损失正常下降**：训练损失和验证损失呈下降趋势
3. **性能优于随机**：模型预测显著优于随机猜测
4. **基础框架完整**：数据处理、图构建、模型训练的完整流程

## 常见问题

### 内存不足
- 减少batch_size（如改为4或2）
- 使用--max_samples限制数据量
- 开启混合精度训练（默认已开启）

### 训练速度慢
- 减少num_workers（多进程可能在某些环境下较慢）
- 使用--test_run进行快速验证
- 考虑使用更小的模型维度

### CUDA相关错误
- 检查PyTorch和CUDA版本兼容性
- 如果GPU内存不足，模型会自动使用CPU

## 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir experiments/stage1/checkpoints/stage1_logs/
```

关键指标：
- `train_loss`: 训练损失
- `val_loss`: 验证损失  
- `val_mae`: 验证集平均绝对误差
- `val_r2`: 验证集R²分数

## 下一步

阶段一完成后，检查以下内容：

1. **训练曲线**：确认损失收敛，无明显过拟合
2. **性能指标**：验证集R²应 > 0.3，MAE应合理
3. **预测质量**：手动检查几个预测样例

如果阶段一成功，可以进入阶段二：
- 增加特征复杂度（方向、电荷等）
- 引入SAB层
- 调优超参数

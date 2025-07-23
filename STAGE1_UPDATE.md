# 阶段一实验更新总结 (v1.1 - KNN连边和验证集修复)

## 最新更新 (v1.1)

### 🔧 **重要修复**

1. **验证集生成问题修复**
   - 修复了预定义分割文件中缺少验证集的问题
   - 如果预定义分割中没有验证集，自动从训练集分出10%作为验证集
   - 确保训练/验证/测试三个数据文件都能正确生成

2. **连边方式改为KNN**
   - 默认使用KNN连边方式（每个原子连接最近的16个邻居）
   - 相比截断半径方式，KNN确保每个原子有固定数量的邻居
   - 避免了大分子中边数过多的问题，提高训练稳定性

### 🚀 **新功能**

- 支持KNN和截断半径两种连边方式
- 可配置的K值（默认k=16）
- 更详细的元数据记录（包含连边方式信息）
- 改进的数据分割逻辑，确保验证集存在

## 原始更新内容 (v1.0)

本次更新为阶段一实验提供了完整的数据处理和训练流程。主要更新包括：

### 新增文件

1. **数据预处理脚本**: `prepare_stage1_data.py`
   - 简化特征提取（只使用原子类型 + 距离）
   - 生成ESA训练所需的边表示格式
   - 输出train.pkl, valid.pkl, test.pkl

2. **训练脚本**: `experiments/stage1/train_stage1.py`
   - 使用现有ESA模型进行训练
   - 自定义数据集和collate函数
   - 支持回调、日志记录等

3. **运行脚本**: 
   - `run_stage1.sh` (Linux)
   - `run_stage1.bat` (Windows)
   - 自动化整个流程

4. **文档和配置**:
   - `experiments/stage1/README.md` - 详细使用指南
   - `experiments/stage1/config_template.json` - 配置模板
   - `test_stage1.py` - 系统测试脚本

### 关键特征简化 (v1.1)

按照Readme.md中阶段一的要求：

#### 原子特征（10维）
- 只使用原子类型独热编码
- 支持：C, N, O, S, P, F, Cl, Br, I, Other

#### 边特征（16维）  
- **连边方式**: KNN (k=16) - 每个原子连接最近的16个原子
- **距离特征**: 高斯基函数扩展距离
- **优势**: 
  - 固定邻居数量，避免大分子边数过多
  - 更稳定的训练过程
  - 计算效率更高

#### 模型架构
- 只使用MAB层（掩码自注意力）
- 不使用SAB层
- 标准PMA池化

### 使用方法 (v1.1)

#### Linux服务器上运行

```bash
# 测试运行（推荐先执行）
bash run_stage1.sh test

# 完整训练
bash run_stage1.sh

# 快速测试修复效果
bash test_fix.sh
```

#### 手动步骤

```bash
# 1. 数据预处理 (KNN连边)
python prepare_stage1_data.py \
    --data_root ./datasets/pdbbind \
    --output_dir ./experiments/stage1 \
    --use_knn --k 16 \
    --test_run  # 快速测试

# 2. 如果需要使用截断半径连边
python prepare_stage1_data.py \
    --data_root ./datasets/pdbbind \
    --output_dir ./experiments/stage1 \
    --use_radius \
    --cutoff_radius 5.0

# 3. 系统测试
python test_stage1.py

# 4. 模型训练
python ./experiments/stage1/train_stage1.py \
    --data_dir ./experiments/stage1 \
    --output_dir ./experiments/stage1/checkpoints \
    --test_run  # 快速测试
```

## 输出文件结构

```
experiments/stage1/
├── train.pkl              # 训练数据（ESA格式）
├── valid.pkl              # 验证数据
├── test.pkl               # 测试数据
├── metadata.json          # 数据集元信息
├── README.md              # 使用指南
├── config_template.json   # 配置模板
├── train_stage1.py        # 训练脚本
└── checkpoints/           # 训练输出
    ├── config.json        # 实际使用的配置
    ├── stage1-*.ckpt      # 模型检查点
    └── stage1_logs/       # TensorBoard日志
```

## 数据格式说明

### ESA输入格式
每个样本包含：
- `edge_representations`: [num_edges, 36] - 边表示矩阵
  - 36 = 节点特征(10) × 2 + 边特征(16)
- `edge_index`: [2, num_edges] - 边连接索引
- `affinity`: float - 结合亲和力标签
- 其他元信息（节点数、边数、复合物ID等）

### 模型配置
- **输入维度**: 36 (10×2 + 16)
- **应用方式**: "edge" (ESA模式)
- **层类型**: ["M", "M"] (只使用MAB)
- **批次大小**: 8 (可调节)

## 成功标准

阶段一的目标：
1. ✅ 代码能正常运行
2. ✅ 损失正常下降
3. ✅ 性能优于随机猜测
4. ✅ 完整的训练流程

## 注意事项

### 环境要求
- PyTorch + PyTorch Geometric
- PyTorch Lightning
- RDKit, BioPython
- 其他依赖见Readme.md

### 内存优化
- 默认batch_size=8，可根据GPU内存调节
- 支持混合精度训练(fp16)
- 大数据集可使用--max_samples限制

### 调试建议
1. 先运行`test_stage1.py`检查环境
2. 使用`--test_run`进行快速验证
3. 检查TensorBoard日志监控训练

## 后续开发

阶段一成功后，可进入阶段二：
1. 增加方向特征（3维单位向量）
2. 增加电荷相互作用特征
3. 引入SAB层组合使用
4. 超参数调优

## 问题排查

### 常见错误
1. **导入错误**: 确保在项目根目录运行，或检查Python路径
2. **CUDA错误**: 检查PyTorch和CUDA版本兼容性
3. **内存不足**: 减少batch_size或使用--max_samples
4. **数据缺失**: 确保PDBbind数据集完整

### 获取帮助
参考`experiments/stage1/README.md`获取详细指南。

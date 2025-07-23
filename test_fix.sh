#!/bin/bash

# 快速测试KNN连边和验证集修复
echo "=== 快速测试阶段一修复 ==="
echo "测试内容："
echo "1. KNN连边方式 (k=16)"
echo "2. 验证集生成修复"
echo "3. 小规模数据测试"
echo ""

# 清理之前的测试数据
echo "清理之前的测试数据..."
rm -rf ./experiments/stage1/*.pkl
rm -rf ./experiments/stage1/metadata.json

# 运行小规模测试
echo "开始小规模测试 (50个样本)..."
python prepare_stage1_data.py \
    --data_root ./datasets/pdbbind \
    --output_dir ./experiments/stage1 \
    --split_type scaffold_split \
    --cutoff_radius 5.0 \
    --num_gaussians 16 \
    --use_knn \
    --k 16 \
    --test_run

# 检查结果
echo ""
echo "=== 测试结果检查 ==="

# 检查文件是否生成
files=("train.pkl" "valid.pkl" "test.pkl" "metadata.json")
for file in "${files[@]}"; do
    if [ -f "./experiments/stage1/$file" ]; then
        echo "✅ $file 已生成"
    else
        echo "❌ $file 未生成"
    fi
done

# 检查metadata信息
if [ -f "./experiments/stage1/metadata.json" ]; then
    echo ""
    echo "=== Metadata 信息 ==="
    python -c "
import json
with open('./experiments/stage1/metadata.json', 'r') as f:
    meta = json.load(f)
print('连边方式:', meta['edge_connection']['method'])
if meta['edge_connection']['method'] == 'knn':
    print('K值:', meta['edge_connection']['k'])
print('特征维度:', meta['feature_dimensions'])
print('数据集统计:', list(meta['dataset_stats'].keys()))
for split, stats in meta['dataset_stats'].items():
    print(f'{split}: {stats[\"num_samples\"]} 样本')
"
fi

echo ""
echo "=== 测试完成 ==="
echo "如果所有文件都已生成且包含验证集，则修复成功！"

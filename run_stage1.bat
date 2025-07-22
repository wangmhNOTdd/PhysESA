@echo off
REM 阶段一实验运行脚本 - Windows版本

echo === 蛋白质-小分子亲和力预测 阶段一实验 ===
echo 目标：搭建基础ESA框架，验证可行性
echo.

REM 检查必要的目录和文件
echo 检查环境...
if not exist ".\datasets\pdbbind" (
    echo 错误: 找不到数据集目录 .\datasets\pdbbind
    echo 请确保PDBbind数据集已正确放置
    pause
    exit /b 1
)

if not exist ".\model\esa" (
    echo 错误: 找不到ESA模型目录 .\model\esa
    pause
    exit /b 1
)

REM 创建必要的目录
mkdir ".\experiments\stage1\checkpoints" 2>nul
mkdir ".\experiments\stage1\logs" 2>nul

echo 环境检查完成
echo.

REM 步骤1: 数据预处理
echo 步骤1: 数据预处理
echo 生成训练、验证、测试数据集...

REM 检查是否已存在预处理数据
if exist ".\experiments\stage1\train.pkl" if exist ".\experiments\stage1\valid.pkl" if exist ".\experiments\stage1\test.pkl" (
    echo 发现已存在的预处理数据
    set /p regenerate="是否重新生成数据？(y/N): "
    if /i "%regenerate%"=="y" (
        echo 重新生成数据...
        python prepare_stage1_data.py --data_root .\datasets\pdbbind --output_dir .\experiments\stage1 --split_type scaffold_split --cutoff_radius 5.0 --num_gaussians 16
    ) else (
        echo 使用现有数据
    )
) else (
    echo 开始数据预处理...
    python prepare_stage1_data.py --data_root .\datasets\pdbbind --output_dir .\experiments\stage1 --split_type scaffold_split --cutoff_radius 5.0 --num_gaussians 16
)

REM 检查数据预处理是否成功
if not exist ".\experiments\stage1\train.pkl" (
    echo 错误: 数据预处理失败，找不到训练数据文件
    pause
    exit /b 1
)

echo 数据预处理完成
echo.

REM 步骤2: 模型训练
echo 步骤2: ESA模型训练
echo 使用简化特征进行基础框架验证...

REM 检查是否为测试运行
if "%1"=="test" (
    echo *** 测试运行模式 ***
    python .\experiments\stage1\train_stage1.py --data_dir .\experiments\stage1 --output_dir .\experiments\stage1\checkpoints --batch_size 4 --learning_rate 1e-4 --max_epochs 10 --test_run
) else (
    echo 开始完整训练...
    python .\experiments\stage1\train_stage1.py --data_dir .\experiments\stage1 --output_dir .\experiments\stage1\checkpoints --batch_size 8 --learning_rate 1e-4 --max_epochs 100
)

echo.
echo === 阶段一实验完成 ===
echo.
echo 输出文件：
echo   - 数据文件: .\experiments\stage1\*.pkl
echo   - 模型检查点: .\experiments\stage1\checkpoints\
echo   - 训练日志: .\experiments\stage1\checkpoints\stage1_logs\
echo.
echo 下一步：
echo 1. 检查训练日志，确认损失正常下降
echo 2. 评估模型性能是否显著优于随机猜测
echo 3. 准备进入阶段二（增加特征复杂度）

pause

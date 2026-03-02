# 1D MT PINN 训练脚本使用说明

## 快速开始

### 1. 生成数据集（如果还没有）

```bash
cd /home/zdz/.conda/PINN/one_PINN
python scripts/generate_dataset.py
```

这将生成 10,000 个样本的数据集，保存在 `data/processed/mt1d_dataset.npz`

### 2. 运行训练

```bash
cd /home/zdz/.conda/PINN/one_PINN
python scripts/train_pinn.py
```

### 3. 查看 TensorBoard

在另一个终端中运行：

```bash
cd /home/zdz/.conda/PINN/one_PINN
tensorboard --logdir=runs
```

然后在浏览器中打开 `http://localhost:6006`

## 配置说明

配置文件位于 `configs/mt_pinn.yaml`，主要参数：

### 训练模式

- **课程学习模式**（推荐）：
  ```yaml
  training:
    curriculum:
      enable: true
      stage1_epochs: 500   # 低频训练
      stage2_epochs: 500   # 中频训练
      stage3_epochs: 1000  # 全频训练
      lbfgs_steps: 100     # LBFGS 优化
  ```

- **标准模式**：
  ```yaml
  training:
    curriculum:
      enable: false
    standard:
      adam_epochs: 2000
      lbfgs_steps: 100
  ```

### 自适应权重

```yaml
physics:
  adaptive_weighting: true  # 启用自适应权重平衡
```

### 设备选择

```yaml
training:
  device: "cuda"  # 或 "cpu"
```

## 输出文件

- **Checkpoint**: `checkpoints/best_model.pth`
- **TensorBoard 日志**: `runs/mt_pinn_1d_YYYYMMDD_HHMMSS/`
- **可视化结果**: `results/mt_inversion_YYYYMMDD_HHMMSS.png`

## 训练流程

1. 加载配置和数据
2. 初始化模型（physics-aware 初始化）
3. 创建物理损失函数和采样器
4. 训练（课程学习或标准模式）
5. 保存最佳模型
6. 自动评估和可视化

## 依赖包

确保已安装：

```bash
pip install torch numpy scikit-learn pyyaml tensorboard matplotlib
```

# PINN 深度学习科研项目

## 项目结构

```
project_root/
│
├── configs/                    # 所有实验配置
│   ├── base.yaml              # 基础配置
│   ├── pinn.yaml              # PINN 完整模型配置
│   └── ablation_no_physics.yaml  # 消融实验配置
│
├── data/
│   ├── raw/                   # 原始正演数据
│   ├── processed/             # 处理后数据
│   └── generators/            # 数据生成器代码
│       ├── __init__.py
│       ├── dataloader.py
│       └── preprocessor.py
│
├── models/
│   ├── __init__.py
│   ├── network.py             # 神经网络架构定义
│   ├── pinn_model.py          # PINN 模型封装
│   └── layers.py              # 自定义层
│
├── losses/
│   ├── __init__.py
│   ├── data_loss.py           # 数据拟合损失
│   ├── physics_loss.py        # 物理方程残差损失
│   └── regularization.py      # 正则化项
│
├── trainers/
│   ├── __init__.py
│   ├── trainer.py             # 训练循环
│   └── callbacks.py           # 训练回调
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # 评估指标
│   ├── visualization.py       # 结果可视化
│   └── tester.py              # 测试流程管理
│
├── utils/
│   ├── __init__.py
│   ├── logger.py              # 日志记录
│   ├── seed.py                # 随机种子设置
│   └── io.py                  # 文件读写
│
├── experiments/
│   ├── __init__.py
│   └── run_experiment.py      # 实验编排
│
├── main.py                     # 程序入口
└── README.md                   # 项目说明
```

## 使用说明

### 安装依赖
```bash
pip install torch numpy scikit-learn pyyaml tensorboard
```

### 运行实验
```bash
# 运行 PINN 完整模型
python main.py --config configs/pinn.yaml

# 运行消融实验
python main.py --config configs/ablation_no_physics.yaml
```

## 模块说明

- **configs/**: 配置驱动的实验管理
- **data/**: 数据加载和预处理
- **models/**: 神经网络和 PINN 模型定义
- **losses/**: 多目标损失函数
- **trainers/**: 训练流程控制
- **evaluation/**: 模型评估和可视化
- **utils/**: 通用工具函数
- **experiments/**: 实验编排和管理

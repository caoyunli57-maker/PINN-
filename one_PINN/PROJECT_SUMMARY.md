# one_PINN — 基于物理信息神经网络的1D大地电磁反演

## 目录

- [项目背景与意义](#项目背景与意义)
- [解决的核心问题](#解决的核心问题)
- [整体方法](#整体方法)
- [项目结构说明](#项目结构说明)
- [核心模块详解](#核心模块详解)
  - [神经网络模型](#神经网络模型)
  - [物理损失函数](#物理损失函数)
  - [训练策略](#训练策略)
  - [数据生成](#数据生成)
  - [评估与可视化](#评估与可视化)
- [数据流与训练流程](#数据流与训练流程)
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [技术细节](#技术细节)

---

## 项目背景与意义

### 什么是大地电磁法（MT）

大地电磁法（Magnetotelluric，MT）是一种利用天然电磁场信号探测地下电阻率结构的地球物理方法。通过在地表观测不同频率的电磁场，可以推断出地下不同深度的岩石电阻率分布——低频信号穿透深，高频信号穿透浅。其核心观测量是：

- **视电阻率（ρa）**：综合反映地下介质的导电性
- **阻抗相位（φ）**：反映电阻率随深度的变化特征

MT 法广泛应用于：油气勘探、矿产资源调查、地热资源探测、地壳结构研究、地震监测等领域。

### 反演问题的挑战

MT 反演是一个**病态逆问题**：

- **非唯一性**：不同的地下电阻率分布可以产生几乎相同的观测数据
- **非线性**：地表响应与地下结构之间是高度非线性关系
- **多尺度**：地下结构跨越从几十米到数百公里的深度范围
- **传统方法的局限**：最小二乘法依赖初始模型，容易陷入局部极小值；纯神经网络缺乏物理约束，泛化能力差

---

## 解决的核心问题

本项目提出并实现了一套基于**物理信息神经网络（PINN）**的1D MT反演方案，核心思路是：

> **将 MT 正演物理方程直接嵌入神经网络的损失函数，使神经网络在拟合观测数据的同时，必须满足物理定律。**

具体解决的问题：

1. **梯度断裂问题**：传统做法将正演计算放在 numpy 中，导致 PyTorch 自动微分图断裂，无法通过物理方程反向传播梯度。本项目实现了完全可微分的 PyTorch 正演算子，保持从损失到模型参数的完整梯度链。

2. **多尺度收敛问题**：高频分量对应浅层细节，低频分量对应深层宏观结构，直接训练容易失衡。采用**课程学习（Curriculum Learning）**策略，从低频到高频逐步引入，由易到难地学习。

3. **损失权重平衡问题**：视电阻率损失和相位损失量级差异大，固定权重会导致某个损失主导训练。采用**基于梯度范数的自适应权重**，动态平衡两个损失项。

4. **局部最优逃逸问题**：Adam 优化器擅长快速下降但精度有限，LBFGS 拟牛顿法擅长精细寻优。采用 **Adam → LBFGS 两阶段优化**，兼顾速度与精度。

---

## 整体方法

```
地表观测数据（视电阻率、相位）
         │
         ▼
┌────────────────────────────┐
│     物理信息神经网络        │
│                            │
│  输入：深度 z              │
│  输出：log(电阻率)          │
│                            │
│  训练目标：                 │
│  ① 预测电阻率 → 正演 →      │
│     预测响应 ≈ 观测响应     │
│  ② 梯度自适应权重平衡       │
└────────────────────────────┘
         │
         ▼
地下连续电阻率剖面 ρ(z)
```

**方法的核心优势**：
- 神经网络表示连续的电阻率剖面 `ρ(z)`，而非离散的层状模型
- 物理正演作为损失函数的一部分参与反向传播
- 无需大量标注样本，仅需一组观测数据即可反演

---

## 项目结构说明

```
one_PINN/
│
├── configs/                    # 实验配置文件
│   ├── base.yaml               # 基础参数模板（学习率、网络结构等通用设置）
│   ├── mt_pinn.yaml            # 正式训练配置（课程学习 + 完整参数）
│   ├── mt_pinn_test.yaml       # 快速测试配置（轻量化，用于调试）
│   ├── pinn.yaml               # 通用 PINN 配置（非 MT 任务）
│   └── ablation_no_physics.yaml # 消融实验：去掉物理约束的对照组
│
├── models/                     # 神经网络模型定义
│   ├── pinn_network.py         # 核心网络：PINNNetwork（深度→电阻率）
│   ├── pinn_model.py           # 模型工厂 + 初始化为均匀半空间
│   ├── network.py              # 基础组件：MLP、ResistivityNet
│   └── layers.py               # 自定义层（预留扩展）
│
├── losses/                     # 损失函数
│   ├── physics_loss.py         # 核心：MT 物理一致性损失
│   │                           #   ├── forward_mt_1d_torch()  可微分正演
│   │                           #   ├── forward_mt_1d_numpy()  参考验证用
│   │                           #   └── MTPhysicsLoss          损失模块
│   ├── data_loss.py            # 数据拟合损失（预留）
│   └── regularization.py      # 正则化项（预留）
│
├── trainers/                   # 训练逻辑
│   ├── pinn_trainer.py         # 核心训练器：Adam + LBFGS + 课程学习
│   ├── trainer.py              # 训练基类（预留）
│   └── callbacks.py            # 回调钩子（预留）
│
├── data/                       # 数据相关
│   ├── generators/
│   │   ├── dataset_generator.py # 随机生成合成 MT 数据集
│   │   ├── mt_forward.py        # numpy 正演（数据生成用）
│   │   ├── dataloader.py        # PyTorch DataLoader 封装
│   │   └── preprocessor.py      # MinMax 标准化 + 数据集分割
│   ├── raw/                    # 原始观测数据（预留）
│   └── processed/              # 生成的合成数据集（.npz 格式）
│
├── utils/                      # 工具库
│   ├── sampling.py             # DepthSampler：深度点采样器
│   ├── autograd_utils.py       # 梯度、二阶导数计算工具
│   ├── logger.py               # 日志（预留）
│   ├── io.py                   # 文件读写（预留）
│   └── seed.py                 # 随机种子管理（预留）
│
├── evaluation/                 # 评估与可视化
│   ├── tester.py               # MTTester：预测剖面 + 绘图
│   ├── metrics.py              # 评估指标（预留）
│   └── visualization.py        # 可视化工具（预留）
│
├── experiments/                # 实验管理（预留）
│   └── run_experiment.py
│
├── scripts/                    # 可执行脚本
│   ├── train_pinn.py           # 主训练入口
│   └── generate_dataset.py     # 数据集生成入口
│
├── checkpoints_test/           # 测试模式保存的模型权重
├── runs_test/                  # 测试模式 TensorBoard 日志
└── PROJECT_SUMMARY.md          # 本文档
```

---

## 核心模块详解

### 神经网络模型

**文件**：`models/pinn_network.py`、`models/pinn_model.py`

网络的任务是学习一个连续函数：

```
f_θ : z (深度) → log₁₀(ρ(z)) (对数电阻率)
```

**架构**：8 层全连接网络，每层 128 个神经元，激活函数为 Tanh。

选择 Tanh 的原因：
- PINN 需要可微的激活函数（用于梯度计算）
- Tanh 输出有界，数值稳定性好
- 相比 ReLU，Tanh 的高阶导数不为零

**初始化策略**（`initialize_as_halfspace`）：

将网络初始化为均匀半空间（常数电阻率），对应 100 Ωm（log₁₀(100) = 2.0）。这避免了随机初始化带来的训练不稳定，给优化提供了一个物理合理的起点。

```
输出层偏置 = log₁₀(ρ₀) = 2.0
输出层权重 = 0（初始输出为常数）
中间层权重 = Xavier 初始化
```

---

### 物理损失函数

**文件**：`losses/physics_loss.py`

这是整个项目最核心的部分。

#### 可微分 MT 正演

1D MT 正演基于**阻抗递推算法**，从最深层向地表逐层计算：

```
底层阻抗：  Z_N = √(jωμρ_N)

向上递推（j = N-1 → 0）：
    k_j  = √(jωμ/ρ_j)       # 波数
    Z_j  = √(jωμρ_j)        # 该层本征阻抗
    r    = (Z_下 - Z_j) / (Z_下 + Z_j)    # 反射系数
    Z    = Z_j × (1 - r·e^{-2k_j h_j}) / (1 + r·e^{-2k_j h_j})

地表响应：
    ρa = |Z|² / (ωμ)        # 视电阻率
    φ  = ∠Z                  # 阻抗相位
```

**为什么要用 PyTorch 重新实现正演？**

传统实现使用 numpy 复数运算，会在以下位置截断 PyTorch 自动微分图：

```
model(z) → resistivity → detach().numpy() → numpy正演 → torch.tensor() → loss
                           ^^^这里断了^^^                  ^^^无grad_fn^^^
```

本项目将所有复数运算（乘、除、指数、开方）拆解为实部和虚部的实数运算，全程保持 PyTorch 计算图：

```
model(z) → resistivity → PyTorch正演 → loss → backward() → ∇θ
              ↑_________________梯度正常回传_____________________↑
```

#### 自适应权重机制

视电阻率（Ωm 量级）和相位（度量级）数值差异大，简单加权会偏向某一项。采用梯度范数归一化：

```
λ_rhoa ← ‖∇_θ L_phase‖ / ‖∇_θ L_rhoa‖
λ_phase ← ‖∇_θ L_rhoa‖ / ‖∇_θ L_phase‖
```

直觉：如果相位梯度比视电阻率梯度大，说明视电阻率更难优化，适当提高 λ_rhoa。

---

### 训练策略

**文件**：`trainers/pinn_trainer.py`

#### 课程学习（Curriculum Learning）

MT反演中，低频对应深部宏观结构（易学），高频对应浅部细节（难学）。训练分四阶段：

```
阶段1：低频 30%  → Adam，学习宏观深层结构
阶段2：中频 60%  → Adam，引入中层特征
阶段3：全频 100% → Adam，学习所有细节
阶段4：全频      → LBFGS，精细优化收敛
```

#### 双优化器策略

- **Adam**：随机梯度下降，收敛快，适合早期大范围搜索
- **LBFGS**（Limited-memory BFGS）：拟牛顿法，利用曲率信息，适合后期精细优化

LBFGS 配置：
```python
max_iter=20, history_size=50, line_search="strong_wolfe"
```

---

### 数据生成

**文件**：`data/generators/dataset_generator.py`、`data/generators/mt_forward.py`

合成数据集的生成流程：

1. **随机生成电阻率模型**
   - 随机选取 3-8 个控制层
   - 在 log 域均匀采样电阻率值（1~1000 Ωm）
   - 在 log 域做线性插值，生成 50 层模型

2. **正演计算**
   - 对每个模型，计算 20 个频率点（0.001~1000 Hz）的 MT 响应
   - 输出：视电阻率（Nf,）+ 相位（Nf,）

3. **数据集格式**（`.npz`）
   ```
   inputs:      (N, Nf, 2)    # [log10(ρa), φ]
   labels:      (N, Nz)       # log10(ρ(z))
   depths:      (Nz,)         # 深度网格 [m]
   frequencies: (Nf,)         # 频率 [Hz]
   ```

**注意**：当前训练脚本仅使用数据集的**第一个样本**作为待反演的观测数据（单样本反演），不做批量监督学习。数据集的价值在于提供真实标签用于评估，以及未来扩展为监督预训练。

---

### 评估与可视化

**文件**：`evaluation/tester.py`

训练完成后，`MTTester` 提供三幅图的综合评估：

```
左图：预测电阻率剖面 vs 真实剖面
     纵轴为深度（对数坐标，浅层在上）
     横轴为电阻率（对数坐标）

中图：视电阻率拟合效果
     双对数坐标，对比观测值与预测值

右图：相位拟合效果
     半对数坐标，对比观测值与预测值
```

---

## 数据流与训练流程

```
generate_dataset.py
        │
        ▼
data/processed/mt1d_dataset.npz
        │
        ▼
train_pinn.py
    │
    ├─ 读取第一个样本的观测数据
    │      (observed_rhoa, observed_phase)
    │
    ├─ 创建 PINNNetwork → 初始化为均匀半空间
    │
    ├─ 创建 MTPhysicsLoss
    │      (thickness, frequencies, λ_rhoa, λ_phase)
    │
    ├─ 创建 DepthSampler
    │      (z_min=10m, z_max=100000m, n=50)
    │
    └─ train_curriculum()
           │
           ├─ Stage 1: 低频 Adam（e.g. 600 epochs）
           │       DepthSampler.sample() → model(z) → 10^z → forward_mt_1d_torch()
           │       → loss_rhoa + loss_phase → 自适应权重 → backward()
           │
           ├─ Stage 2: 中频 Adam（e.g. 600 epochs）
           │
           ├─ Stage 3: 全频 Adam（e.g. 800 epochs）
           │
           └─ Stage 4: LBFGS 精调（e.g. 100 steps）
                   │
                   └─ 保存 best_model.pth
                          │
                          ▼
                   MTTester.plot_results()
                          │
                          ▼
                   results/mt_inversion_*.png
```

---

## 快速开始

### 环境依赖

```bash
pip install torch numpy scikit-learn pyyaml tensorboard matplotlib
```

### 生成合成数据集

```bash
cd /path/to/one_PINN
python scripts/generate_dataset.py
# 输出：data/processed/mt1d_dataset.npz
```

### 运行训练

```bash
# 快速调试（轻量配置）
python scripts/train_pinn.py --config configs/mt_pinn_test.yaml

# 正式训练
python scripts/train_pinn.py --config configs/mt_pinn.yaml
```

### 查看训练曲线

```bash
tensorboard --logdir=runs
# 浏览器访问 http://localhost:6006
```

### 输出文件位置

| 文件 | 说明 |
|------|------|
| `checkpoints/best_model.pth` | 训练过程中最优模型权重 |
| `runs/mt_pinn_1d_YYYYMMDD_HHMMSS/` | TensorBoard 日志 |
| `results/mt_inversion_YYYYMMDD_HHMMSS.png` | 最终反演结果图 |

---

## 配置文件说明

### mt_pinn.yaml（正式训练）

```yaml
model:
  n_layers: 8           # 隐藏层数
  hidden_width: 128     # 每层神经元数
  activation: tanh      # 激活函数
  init_log_rho: 2.0     # 初始电阻率 = 10^2 = 100 Ωm

physics:
  lambda_rhoa: 1.0          # 视电阻率损失权重（自适应时为初始值）
  lambda_phase: 10.0        # 相位损失权重
  adaptive_weighting: true  # 是否启用梯度范数自适应权重

training:
  use_curriculum: true  # 启用课程学习
  learning_rate: 0.001
  adam_epochs_stage1: 600
  adam_epochs_stage2: 600
  adam_epochs_stage3: 800
  lbfgs_steps: 100

sampling:
  z_min: 10.0        # 最浅采样深度 [m]
  z_max: 100000.0    # 最深采样深度 [m]
  n_samples: 50      # 每次迭代采样点数
```

### 配置对比

| 配置 | 用途 | 物理约束 | 训练模式 |
|------|------|----------|----------|
| `mt_pinn.yaml` | 正式 MT 反演 | 启用 | 课程学习 |
| `mt_pinn_test.yaml` | 快速调试 | 启用 | 课程学习（缩减版） |
| `ablation_no_physics.yaml` | 消融对照 | 禁用 | 标准 |
| `pinn.yaml` | 通用 PINN | 启用 | 标准 |

---

## 技术细节

### 为什么在 log 域计算电阻率误差

视电阻率变化范围跨越多个数量级（1~10000 Ωm），直接用 MSE 会被大值主导。在对数域计算：

```
L_rhoa = MSE(log₁₀(ρa_pred), log₁₀(ρa_obs))
```

使得各频率点的贡献权重更均匀。

### 深度采样策略

每次迭代随机采样深度点，而非固定网格。这让网络在整个深度范围内均匀优化，避免过拟合到固定点。推理时切换为均匀网格采样（`sample_grid()`）。

### 复数阻抗的 PyTorch 实现

PyTorch 的复数支持在早期版本（如 Python 3.8 + PyTorch 1.x）中不稳定，且复数张量的 autograd 支持不完整。本项目采用**实部/虚部分离表示**，将所有复数运算拆解为实数运算：

```python
# 复数乘法：(a+jb)(c+jd) = (ac-bd) + j(ad+bc)
prod_re = a * c - b * d
prod_im = a * d + b * c

# 复数除法：(a+jb)/(c+jd) = [(ac+bd) + j(bc-ad)] / (c²+d²)
norm_sq = c**2 + d**2
quot_re = (a*c + b*d) / norm_sq
quot_im = (b*c - a*d) / norm_sq

# 复数指数：exp(a+jb) = exp(a)(cos(b) + j·sin(b))
mag = torch.exp(a)
e_re = mag * torch.cos(b)
e_im = mag * torch.sin(b)
```

这确保了整个正演计算对 PyTorch autograd 完全透明，梯度可以从物理损失正确传回模型参数。

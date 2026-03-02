"""
one_PINN — 基于物理信息神经网络的1D大地电磁反演
主程序入口

直接运行（VSCode F5）：默认执行训练
命令行运行：
    python main.py                                   # 默认训练
    python main.py generate                          # 生成合成数据集
    python main.py train                             # 训练
    python main.py train --config configs/mt_pinn_test.yaml  # 指定配置
"""

import sys
import os

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

from models import PINNNetwork, initialize_as_halfspace
from losses import MTPhysicsLoss
from trainers import PINNTrainer
from utils import DepthSampler
from evaluation import MTTester


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_dataset(config):
    """加载数据集"""
    dataset_path = config['data']['dataset_path']
    
    if not os.path.exists(dataset_path):
        print(f"错误：数据集不存在: {dataset_path}")
        print("请先运行: python main.py generate")
        sys.exit(1)
    
    data = np.load(dataset_path)
    
    # 获取一个样本用于训练（这里简化为使用第一个样本）
    inputs = data['inputs'][0]  # (Nf, 2)
    labels = data['labels'][0]  # (Nz,)
    depths = data['depths']
    frequencies = data['frequencies']
    
    # 提取观测数据
    log_rhoa = inputs[:, 0]  # log10(视电阻率)
    phase = inputs[:, 1]      # 相位
    
    # 转换为原始视电阻率
    rhoa = 10 ** log_rhoa
    
    return rhoa, phase, depths, frequencies, labels


def create_model(config, device):
    """创建 PINN 模型"""
    model = PINNNetwork(
        n_layers=config['model']['n_layers'],
        hidden_width=config['model']['hidden_width'],
        activation=config['model']['activation']
    )
    
    # Physics-aware 初始化
    initialize_as_halfspace(model, config['model']['init_log_rho'])
    
    model.to(device)
    return model


def create_loss_fn(config, depths, frequencies):
    """创建物理损失函数"""
    thickness = np.diff(depths)
    
    loss_fn = MTPhysicsLoss(
        thickness=thickness,
        frequencies=frequencies,
        lambda_rhoa=config['physics']['lambda_rhoa'],
        lambda_phase=config['physics']['lambda_phase'],
        adaptive_weighting=config['physics']['adaptive_weighting']
    )
    
    return loss_fn


def create_sampler(config, device):
    """创建深度采样器"""
    sampler = DepthSampler(
        z_min=config['sampling']['z_min'],
        z_max=config['sampling']['z_max'],
        n_samples=config['sampling']['n_samples'],
        device=device
    )
    return sampler


def train_with_tensorboard(trainer, config, rhoa, phase, frequencies, writer, device):
    """使用 TensorBoard 记录的训练"""
    
    # 转换为 tensor
    rhoa_tensor = torch.tensor(rhoa, dtype=torch.float32, device=device)
    phase_tensor = torch.tensor(phase, dtype=torch.float32, device=device)
    freq_tensor = torch.tensor(frequencies, dtype=torch.float32, device=device)
    
    # 创建 checkpoint 目录
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    if config['training']['curriculum']['enable']:
        # 课程学习训练
        print("=" * 60)
        print("开始课程学习训练")
        print("=" * 60)
        
        history = trainer.train_curriculum(
            observed_rhoa=rhoa_tensor,
            observed_phase=phase_tensor,
            frequencies=freq_tensor,
            adam_epochs_stage1=config['training']['curriculum']['stage1_epochs'],
            adam_epochs_stage2=config['training']['curriculum']['stage2_epochs'],
            adam_epochs_stage3=config['training']['curriculum']['stage3_epochs'],
            lbfgs_steps=config['training']['curriculum']['lbfgs_steps'],
            verbose=True
        )
        
        # 记录到 TensorBoard
        global_step = 0
        for stage_name, stage_history in history.items():
            for i, loss in enumerate(stage_history):
                writer.add_scalar(f'Loss/{stage_name}', loss, global_step)
                global_step += 1
                
                # 保存最佳模型
                if loss < best_loss:
                    best_loss = loss
                    if config['checkpoint']['save_best']:
                        torch.save({
                            'model_state_dict': trainer.model.state_dict(),
                            'loss': best_loss,
                            'stage': stage_name,
                            'step': i
                        }, os.path.join(checkpoint_dir, 'best_model.pth'))
    
    else:
        # 标准两阶段训练
        print("=" * 60)
        print("开始标准两阶段训练")
        print("=" * 60)
        
        history = trainer.train_full(
            adam_epochs=config['training']['standard']['adam_epochs'],
            lbfgs_steps=config['training']['standard']['lbfgs_steps'],
            observed_rhoa=rhoa_tensor,
            observed_phase=phase_tensor,
            verbose=True
        )
        
        # 记录到 TensorBoard
        global_step = 0
        for phase_name, phase_history in history.items():
            for i, loss in enumerate(phase_history):
                writer.add_scalar(f'Loss/{phase_name}', loss, global_step)
                global_step += 1
                
                # 保存最佳模型
                if loss < best_loss:
                    best_loss = loss
                    if config['checkpoint']['save_best']:
                        torch.save({
                            'model_state_dict': trainer.model.state_dict(),
                            'loss': best_loss,
                            'phase': phase_name,
                            'step': i
                        }, os.path.join(checkpoint_dir, 'best_model.pth'))
    
    print(f"\n训练完成！最佳损失: {best_loss:.6f}")
    return history


def evaluate_and_visualize(model, sampler, config, rhoa, phase, depths, frequencies, true_labels=None):
    """评估并可视化结果"""
    
    thickness = np.diff(depths)
    device = config['training']['device']
    
    # 创建测试器
    tester = MTTester(
        model=model,
        sampler=sampler,
        thickness=thickness,
        frequencies=frequencies,
        device=device
    )
    
    # 创建结果目录
    results_dir = config['evaluation']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制结果
    print("\n生成可视化结果...")
    tester.plot_results(rhoa, phase, true_resistivity=true_labels)
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        results_dir,
        f"mt_inversion_{timestamp}.{config['evaluation']['plot_format']}"
    )
    plt.savefig(save_path, dpi=config['evaluation']['dpi'], bbox_inches='tight')
    print(f"结果已保存至: {save_path}")
    
    # 不自动显示图像，避免阻塞
    # plt.show()
    plt.close()


def run_training(config_path=None):
    """运行训练流程"""
    
    # 默认配置文件
    if config_path is None:
        config_path = os.path.join(project_root, 'configs', 'mt_pinn.yaml')
    
    # 加载配置
    config = load_config(config_path)
    print(f"加载配置文件: {config_path}")
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        device = 'cpu'
        config['training']['device'] = 'cpu'
    
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据集...")
    rhoa, phase, depths, frequencies, true_labels = load_dataset(config)
    print(f"数据加载完成")
    print(f"  频率点数: {len(frequencies)}")
    print(f"  深度层数: {len(depths)}")
    
    # 创建模型
    print("\n创建 PINN 模型...")
    model = create_model(config, device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建损失函数
    print("\n创建物理损失函数...")
    loss_fn = create_loss_fn(config, depths, frequencies)
    
    # 创建采样器
    print("\n创建深度采样器...")
    sampler = create_sampler(config, device)
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = PINNTrainer(
        model=model,
        loss_fn=loss_fn,
        sampler=sampler,
        learning_rate=config['training']['learning_rate'],
        device=device
    )
    
    # 创建 TensorBoard writer
    log_dir = os.path.join(
        config['tensorboard']['log_dir'],
        f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")
    
    # 训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    history = train_with_tensorboard(
        trainer, config, rhoa, phase, frequencies, writer, device
    )
    
    # 关闭 TensorBoard writer
    writer.close()
    
    # 评估和可视化
    print("\n" + "=" * 60)
    print("评估和可视化")
    print("=" * 60)
    
    evaluate_and_visualize(
        model, sampler, config, rhoa, phase, depths, frequencies, true_labels
    )
    
    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)
    print(f"查看 TensorBoard: tensorboard --logdir={config['tensorboard']['log_dir']}")


def cmd_generate(args):
    """生成合成 MT 数据集"""
    import argparse
    parser = argparse.ArgumentParser(description='生成 MT 1D 合成数据集')
    parser.add_argument('--n-samples', type=int, default=100000,
                        help='生成样本数量（默认 100000）')
    parser.add_argument('--n-layers', type=int, default=50,
                        help='地下离散层数（默认 50）')
    parser.add_argument('--n-frequencies', type=int, default=20,
                        help='频率点数量（默认 20）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认 42）')
    parser.add_argument('--output', type=str, default='data/processed/mt1d_dataset.npz',
                        help='输出数据集路径')
    parsed = parser.parse_args(args)

    from data.generators import MT1DDatasetGenerator

    config = {
        'n_samples': parsed.n_samples,
        'n_layers': parsed.n_layers,
        'n_frequencies': parsed.n_frequencies,
        'depth_range': (10.0, 100000.0),
        'resistivity_range': (1.0, 1000.0),
        'frequency_range': (0.001, 1000.0),
        'control_layers_range': (3, 8),
        'seed': parsed.seed,
    }

    generator = MT1DDatasetGenerator(
        n_samples=config['n_samples'],
        n_layers=config['n_layers'],
        n_frequencies=config['n_frequencies'],
        depth_range=config['depth_range'],
        resistivity_range=config['resistivity_range'],
        frequency_range=config['frequency_range'],
        control_layers_range=config['control_layers_range'],
        seed=config['seed'],
    )

    os.makedirs(os.path.dirname(parsed.output), exist_ok=True)
    print("生成 MT 1D 合成数据集...")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    generator.save_dataset(parsed.output)
    print(f"\n数据集已保存至: {parsed.output}")


def cmd_train(args):
    """训练 PINN 反演模型"""
    import argparse
    parser = argparse.ArgumentParser(description='1D MT PINN 训练')
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, 'configs', 'mt_pinn.yaml'),
                        help='配置文件路径（默认 configs/mt_pinn.yaml）')
    parsed = parser.parse_args(args)
    
    run_training(parsed.config)


def print_help():
    print(__doc__)
    print("子命令：")
    print("  generate   生成合成 MT 数据集")
    print("  train      训练 PINN 反演模型")
    print()
    print("示例：")
    print("  python main.py                                    # 默认训练")
    print("  python main.py generate --n-samples 50000         # 生成数据")
    print("  python main.py train --config configs/mt_pinn_test.yaml")


COMMANDS = {
    'generate': cmd_generate,
    'train': cmd_train,
}


if __name__ == '__main__':
    # 如果没有命令行参数，默认执行训练
    if len(sys.argv) == 1:
        print("=" * 60)
        print("直接运行模式：开始训练")
        print("=" * 60)
        print()
        run_training()
    elif sys.argv[1] in ('-h', '--help'):
        print_help()
    else:
        subcmd = sys.argv[1]
        if subcmd not in COMMANDS:
            print(f"错误：未知子命令 '{subcmd}'")
            print(f"可用子命令：{', '.join(COMMANDS)}")
            sys.exit(1)
        
        COMMANDS[subcmd](sys.argv[2:])

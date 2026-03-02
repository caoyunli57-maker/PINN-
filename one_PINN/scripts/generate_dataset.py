"""生成 MT 1D 数据集脚本"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.generators import MT1DDatasetGenerator


def main():
    """生成数据集主函数"""
    
    # 数据集参数
    config = {
        'n_samples': 100000,
        'n_layers': 50,
        'n_frequencies': 20,
        'depth_range': (10.0, 100000.0),
        'resistivity_range': (1.0, 1000.0),
        'frequency_range': (0.001, 1000.0),
        'control_layers_range': (3, 8),
        'seed': 42
    }
    
    # 创建生成器
    generator = MT1DDatasetGenerator(
        n_samples=config['n_samples'],
        n_layers=config['n_layers'],
        n_frequencies=config['n_frequencies'],
        depth_range=config['depth_range'],
        resistivity_range=config['resistivity_range'],
        frequency_range=config['frequency_range'],
        control_layers_range=config['control_layers_range'],
        seed=config['seed']
    )
    
    # 生成并保存数据集
    save_path = 'data/processed/mt1d_dataset.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("Generating MT 1D dataset...")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    generator.save_dataset(save_path)
    
    print(f"\nDataset generation completed!")


if __name__ == '__main__':
    main()

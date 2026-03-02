"""1D MT 数据集生成器"""

import numpy as np
from typing import Tuple, Dict
from .mt_forward import mt_1d_forward


class MT1DDatasetGenerator:
    """1D MT 数据集生成器"""
    
    def __init__(
        self,
        n_samples: int,
        n_layers: int,
        n_frequencies: int,
        depth_range: Tuple[float, float],
        resistivity_range: Tuple[float, float],
        frequency_range: Tuple[float, float],
        control_layers_range: Tuple[int, int] = (3, 8),
        seed: int = 42
    ):
        """
        Args:
            n_samples: 样本数量
            n_layers: 模型层数 Nz
            n_frequencies: 频率点数 Nf
            depth_range: 深度范围 [min, max] (m)
            resistivity_range: 电阻率范围 [min, max] (Ωm)
            frequency_range: 频率范围 [min, max] (Hz)
            control_layers_range: 控制层数量范围 [min, max]
            seed: 随机种子
        """
        self.n_samples = n_samples
        self.n_layers = n_layers
        self.n_frequencies = n_frequencies
        self.depth_range = depth_range
        self.resistivity_range = resistivity_range
        self.frequency_range = frequency_range
        self.control_layers_range = control_layers_range
        self.seed = seed
        
        np.random.seed(seed)
        
        # 生成深度网格 (logspace)
        self.depths = np.logspace(
            np.log10(depth_range[0]),
            np.log10(depth_range[1]),
            n_layers
        )
        
        # 生成频率数组 (logspace)
        self.frequencies = np.logspace(
            np.log10(frequency_range[0]),
            np.log10(frequency_range[1]),
            n_frequencies
        )
    
    def _generate_random_resistivity_model(self) -> np.ndarray:
        """
        生成随机电阻率模型
        
        Returns:
            resistivity: 电阻率模型 (Nz,) [Ωm]
        """
        # 随机选择控制层数量
        n_control = np.random.randint(
            self.control_layers_range[0],
            self.control_layers_range[1] + 1
        )
        
        # 随机选择控制层深度索引
        control_indices = np.sort(
            np.random.choice(self.n_layers, n_control, replace=False)
        )
        
        # 在 log-domain 随机采样电阻率
        log_rho_min = np.log10(self.resistivity_range[0])
        log_rho_max = np.log10(self.resistivity_range[1])
        control_log_rho = np.random.uniform(
            log_rho_min,
            log_rho_max,
            n_control
        )
        
        # 在 log-domain 插值生成完整模型
        log_resistivity = np.interp(
            np.arange(self.n_layers),
            control_indices,
            control_log_rho
        )
        
        resistivity = 10 ** log_resistivity
        
        return resistivity
    
    def _compute_thickness(self) -> np.ndarray:
        """
        计算层厚度
        
        Returns:
            thickness: 层厚度 (Nz-1,) [m]
        """
        thickness = np.diff(self.depths)
        return thickness
    
    def generate_dataset(self) -> Dict[str, np.ndarray]:
        """
        生成完整数据集
        
        Returns:
            dataset: 包含以下键的字典
                - 'inputs': (N, Nf, 2) [log10(ρa), phase]
                - 'labels': (N, Nz) [log10(ρ)]
                - 'depths': (Nz,) [m]
                - 'frequencies': (Nf,) [Hz]
        """
        thickness = self._compute_thickness()
        
        inputs = []
        labels = []
        
        for i in range(self.n_samples):
            # 生成随机电阻率模型
            resistivity = self._generate_random_resistivity_model()
            
            # MT 正演计算
            app_res, phase = mt_1d_forward(
                resistivity,
                thickness,
                self.frequencies
            )
            
            # 构建输入输出对
            # input: [log10(ρa), phase]
            log_app_res = np.log10(app_res)
            input_sample = np.stack([log_app_res, phase], axis=-1)  # (Nf, 2)
            
            # label: log10(ρ)
            log_resistivity = np.log10(resistivity)  # (Nz,)
            
            inputs.append(input_sample)
            labels.append(log_resistivity)
        
        dataset = {
            'inputs': np.array(inputs),  # (N, Nf, 2)
            'labels': np.array(labels),  # (N, Nz)
            'depths': self.depths,       # (Nz,)
            'frequencies': self.frequencies  # (Nf,)
        }
        
        return dataset
    
    def save_dataset(self, save_path: str):
        """
        生成并保存数据集
        
        Args:
            save_path: 保存路径
        """
        dataset = self.generate_dataset()
        np.savez(save_path, **dataset)
        print(f"Dataset saved to {save_path}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Input shape: {dataset['inputs'].shape}")
        print(f"  Label shape: {dataset['labels'].shape}")

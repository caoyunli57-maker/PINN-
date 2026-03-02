"""深度采样工具"""

import torch


class DepthSampler:
    """深度采样器"""
    
    def __init__(
        self,
        z_min: float,
        z_max: float,
        n_samples: int,
        device: str
    ):
        """
        Args:
            z_min: 最小深度
            z_max: 最大深度
            n_samples: 采样点数
            device: 设备
        """
        self.z_min = z_min
        self.z_max = z_max
        self.n_samples = n_samples
        self.device = device
    
    def sample(self) -> torch.Tensor:
        """
        均匀随机采样
        
        Returns:
            z: 深度张量 (n_samples, 1)
        """
        z = torch.rand(
            self.n_samples, 1,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        z = z * (self.z_max - self.z_min) + self.z_min
        
        return z
    
    def sample_grid(self) -> torch.Tensor:
        """
        均匀网格采样
        
        Returns:
            z: 深度张量 (n_samples, 1)
        """
        z = torch.linspace(
            self.z_min,
            self.z_max,
            self.n_samples,
            dtype=torch.float32,
            device=self.device
        ).reshape(-1, 1)
        z.requires_grad = True
        
        return z

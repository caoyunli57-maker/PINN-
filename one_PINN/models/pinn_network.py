"""PINN 网络 - 1D MT 反演"""

import torch
import torch.nn as nn
from typing import Optional


class PINNNetwork(nn.Module):
    """PINN 网络 - 连续函数 log_resistivity(z)"""
    
    def __init__(
        self,
        n_layers: int = 8,
        hidden_width: int = 128,
        activation: str = 'tanh'
    ):
        """
        Args:
            n_layers: 隐藏层数量
            hidden_width: 隐藏层宽度
            activation: 激活函数类型
        """
        super(PINNNetwork, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_width = hidden_width
        
        # 激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 输入层
        self.input_layer = nn.Linear(1, hidden_width)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_width, hidden_width)
            for _ in range(n_layers)
        ])
        
        # 输出层（线性）
        self.output_layer = nn.Linear(hidden_width, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 深度坐标 (batch, 1)
            
        Returns:
            log_resistivity: 对数电阻率 (batch, 1)
        """
        # 输入层
        x = self.activation(self.input_layer(z))
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # 输出层（线性）
        log_resistivity = self.output_layer(x)
        
        return log_resistivity

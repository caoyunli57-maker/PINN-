"""神经网络架构定义"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = 'tanh'
    ):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数类型
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # 激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建网络层
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation)
        
        # 输出层
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, input_dim)
            
        Returns:
            output: 输出张量 (batch, output_dim)
        """
        return self.network(x)


class ResistivityNet(nn.Module):
    """电阻率网络 - 连续函数 log_resistivity(z)"""
    
    def __init__(
        self,
        hidden_dims: List[int] = [128, 128, 128, 128, 128, 128],
        activation: str = 'tanh'
    ):
        """
        Args:
            hidden_dims: 隐藏层维度列表
            activation: 激活函数类型
        """
        super(ResistivityNet, self).__init__()
        
        self.mlp = MLP(
            input_dim=1,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 深度坐标 (batch, 1)
            
        Returns:
            log_resistivity: 对数电阻率 (batch, 1)
        """
        return self.mlp(z)

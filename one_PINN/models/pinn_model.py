"""PINN 模型封装"""

import torch
import torch.nn as nn
from typing import Dict, List
from .network import ResistivityNet


def initialize_as_halfspace(model: nn.Module, log_rho0: float = 2.0):
    """
    初始化网络为均匀半空间
    
    Args:
        model: PINN 模型
        log_rho0: 初始对数电阻率
    """
    layers = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            layers.append(module)
    
    # 初始化所有层
    for layer in layers[:-1]:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    
    # 初始化输出层
    if len(layers) > 0:
        output_layer = layers[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, log_rho0)


class PINNModel(nn.Module):
    """PINN 模型封装类"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        super(PINNModel, self).__init__()
        
        self.config = config
        
        # 从配置读取网络参数
        hidden_layers = config['model']['hidden_layers']
        activation = config['model']['activation']
        
        # 创建电阻率网络
        self.resistivity_net = ResistivityNet(
            hidden_dims=hidden_layers,
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
        return self.resistivity_net(z)
    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        预测模式
        
        Args:
            z: 深度坐标 (batch, 1)
            
        Returns:
            log_resistivity: 对数电阻率 (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(z)

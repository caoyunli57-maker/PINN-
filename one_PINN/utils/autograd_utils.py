"""自动微分工具函数"""

import torch
from typing import Optional


def gradient(
    y: torch.Tensor,
    x: torch.Tensor,
    grad_outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算 dy/dx
    
    Args:
        y: 输出张量 (batch, 1)
        x: 输入张量 (batch, 1)
        grad_outputs: 梯度输出权重
        
    Returns:
        dy_dx: 梯度张量 (batch, 1)
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    
    dy_dx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return dy_dx


def second_derivative(
    y: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    计算 d²y/dx²
    
    Args:
        y: 输出张量 (batch, 1)
        x: 输入张量 (batch, 1)
        
    Returns:
        d2y_dx2: 二阶导数张量 (batch, 1)
    """
    dy_dx = gradient(y, x)
    d2y_dx2 = gradient(dy_dx, x)
    
    return d2y_dx2

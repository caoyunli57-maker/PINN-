"""测试流程管理"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def forward_mt_1d(
    resistivity: np.ndarray,
    thickness: np.ndarray,
    frequencies: np.ndarray,
    mu: float = 4 * np.pi * 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D MT 正演计算
    
    Args:
        resistivity: 电阻率 (Nz,) [Ωm]
        thickness: 层厚度 (Nz-1,) [m]
        frequencies: 频率数组 (Nf,) [Hz]
        mu: 磁导率 [H/m]
        
    Returns:
        apparent_resistivity: 视电阻率 (Nf,) [Ωm]
        phase: 相位 (Nf,) [度]
    """
    n_layers = len(resistivity)
    n_freq = len(frequencies)
    
    app_res = np.zeros(n_freq)
    phase = np.zeros(n_freq)
    
    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq
        
        Z = np.sqrt(1j * omega * mu * resistivity[-1])
        
        for j in range(n_layers - 2, -1, -1):
            k = np.sqrt(1j * omega * mu / resistivity[j])
            Z_j = np.sqrt(1j * omega * mu * resistivity[j])
            
            if j < n_layers - 1:
                r = (Z - Z_j) / (Z + Z_j)
                Z = Z_j * (1 - r * np.exp(-2 * k * thickness[j])) / \
                    (1 + r * np.exp(-2 * k * thickness[j]))
        
        app_res[i] = (np.abs(Z) ** 2) / (omega * mu)
        phase[i] = np.angle(Z, deg=True)
    
    return app_res, phase


class MTTester:
    """MT 测试器"""
    
    def __init__(
        self,
        model: nn.Module,
        sampler,
        thickness: np.ndarray,
        frequencies: np.ndarray,
        device: str
    ):
        """
        Args:
            model: PINN 模型
            sampler: 深度采样器
            thickness: 层厚度 (Nz-1,) [m]
            frequencies: 频率数组 (Nf,) [Hz]
            device: 设备
        """
        self.model = model
        self.sampler = sampler
        self.thickness = thickness
        self.frequencies = frequencies
        self.device = device
    
    def predict_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测电阻率剖面
        
        Returns:
            z: 深度 (Nz,)
            resistivity: 电阻率 (Nz,)
        """
        self.model.eval()
        
        with torch.no_grad():
            z = self.sampler.sample_grid()
            log_resistivity = self.model(z)
            resistivity = 10 ** log_resistivity
        
        z_np = z.detach().cpu().numpy().flatten()
        resistivity_np = resistivity.detach().cpu().numpy().flatten()
        
        return z_np, resistivity_np
    
    def predict_mt_response(
        self,
        resistivity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测 MT 响应
        
        Args:
            resistivity: 电阻率 (Nz,)
            
        Returns:
            rhoa_pred: 视电阻率 (Nf,)
            phase_pred: 相位 (Nf,)
        """
        rhoa_pred, phase_pred = forward_mt_1d(
            resistivity,
            self.thickness,
            self.frequencies
        )
        
        return rhoa_pred, phase_pred
    
    def plot_results(
        self,
        observed_rhoa: np.ndarray,
        observed_phase: np.ndarray,
        true_resistivity: np.ndarray = None
    ):
        """
        绘制结果
        
        Args:
            observed_rhoa: 观测视电阻率 (Nf,)
            observed_phase: 观测相位 (Nf,)
            true_resistivity: 真实电阻率剖面 (Nz,)，可选
        """
        z, resistivity = self.predict_profile()
        rhoa_pred, phase_pred = self.predict_mt_response(resistivity)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 电阻率剖面
        axes[0].plot(resistivity, z, 'b-', linewidth=2, label='PINN Prediction')
        if true_resistivity is not None:
            axes[0].plot(true_resistivity, z, 'g--', linewidth=2, label='True Model', alpha=0.7)
        axes[0].set_xlabel('Resistivity (Ωm)', fontsize=12)
        axes[0].set_ylabel('Depth (m)', fontsize=12)
        axes[0].set_xscale('log')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Resistivity Profile', fontsize=14, fontweight='bold')
        if true_resistivity is not None:
            axes[0].legend(fontsize=10)
        
        # 计算拟合误差
        rhoa_rmse = np.sqrt(np.mean((np.log10(rhoa_pred) - np.log10(observed_rhoa))**2))
        phase_rmse = np.sqrt(np.mean((phase_pred - observed_phase)**2))
        
        # 视电阻率对比
        axes[1].loglog(self.frequencies, observed_rhoa, 'ko-', label='Observed', markersize=6, linewidth=1.5)
        axes[1].loglog(self.frequencies, rhoa_pred, 'r^-', label='Predicted', markersize=6, linewidth=1.5)
        axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[1].set_ylabel('Apparent Resistivity (Ωm)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].set_title(f'Apparent Resistivity\nRMSE(log): {rhoa_rmse:.4f}', 
                         fontsize=14, fontweight='bold')
        
        # 相位对比
        axes[2].semilogx(self.frequencies, observed_phase, 'ko-', label='Observed', markersize=6, linewidth=1.5)
        axes[2].semilogx(self.frequencies, phase_pred, 'r^-', label='Predicted', markersize=6, linewidth=1.5)
        axes[2].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[2].set_ylabel('Phase (degree)', fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title(f'Phase\nRMSE: {phase_rmse:.4f}°', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

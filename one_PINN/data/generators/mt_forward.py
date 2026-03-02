"""1D MT 正演计算模块"""

import numpy as np
from typing import Tuple


def mt_1d_forward(
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
        
        # 从底层向上递推计算阻抗
        Z = np.sqrt(1j * omega * mu * resistivity[-1])
        
        for j in range(n_layers - 2, -1, -1):
            k = np.sqrt(1j * omega * mu / resistivity[j])
            Z_j = np.sqrt(1j * omega * mu * resistivity[j])
            
            if j < n_layers - 1:
                r = (Z - Z_j) / (Z + Z_j)
                Z = Z_j * (1 - r * np.exp(-2 * k * thickness[j])) / \
                    (1 + r * np.exp(-2 * k * thickness[j]))
        
        # 计算视电阻率和相位
        app_res[i] = (np.abs(Z) ** 2) / (omega * mu)
        phase[i] = np.angle(Z, deg=True)
    
    return app_res, phase

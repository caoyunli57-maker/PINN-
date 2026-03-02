"""物理方程残差损失"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def forward_mt_1d_numpy(
    resistivity: np.ndarray,
    thickness: np.ndarray,
    frequencies: np.ndarray,
    mu: float = 4 * np.pi * 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D MT 正演计算 (numpy 版本，不可微分，仅供参考/验证)

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


def forward_mt_1d_torch(
    resistivity: torch.Tensor,
    thickness: torch.Tensor,
    frequencies: torch.Tensor,
    mu: float = 4 * np.pi * 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1D MT 正演计算 (PyTorch 可微分版本)

    用实部和虚部分别表示复数阻抗，保持 autograd 计算图。

    Args:
        resistivity: 电阻率 (Nz,) [Ωm]，需要 grad
        thickness: 层厚度 (Nz-1,) [m]
        frequencies: 频率数组 (Nf,) [Hz]
        mu: 磁导率 [H/m]

    Returns:
        apparent_resistivity: 视电阻率 (Nf,) [Ωm]
        phase: 相位 (Nf,) [度]
    """
    n_layers = resistivity.shape[0]
    n_freq = frequencies.shape[0]

    app_res_list = []
    phase_list = []

    for i in range(n_freq):
        omega = 2 * np.pi * frequencies[i].item()

        # 底层阻抗: Z = sqrt(j * omega * mu * rho[-1])
        # sqrt(j) = (1+j)/sqrt(2)
        # sqrt(j * omega * mu * rho) = sqrt(omega * mu * rho) * (1+j)/sqrt(2)
        sqrt_val = torch.sqrt(omega * mu * resistivity[-1])
        Z_re = sqrt_val / np.sqrt(2)
        Z_im = sqrt_val / np.sqrt(2)

        for j in range(n_layers - 2, -1, -1):
            # k = sqrt(j * omega * mu / rho[j])
            # k = sqrt(omega * mu / rho[j]) * (1+j)/sqrt(2)
            sqrt_k = torch.sqrt(omega * mu / resistivity[j])
            k_re = sqrt_k / np.sqrt(2)
            k_im = sqrt_k / np.sqrt(2)

            # Z_j = sqrt(j * omega * mu * rho[j])
            sqrt_zj = torch.sqrt(omega * mu * resistivity[j])
            Zj_re = sqrt_zj / np.sqrt(2)
            Zj_im = sqrt_zj / np.sqrt(2)

            # r = (Z - Z_j) / (Z + Z_j)  复数除法
            num_re = Z_re - Zj_re
            num_im = Z_im - Zj_im
            den_re = Z_re + Zj_re
            den_im = Z_im + Zj_im
            den_norm_sq = den_re ** 2 + den_im ** 2
            r_re = (num_re * den_re + num_im * den_im) / den_norm_sq
            r_im = (num_im * den_re - num_re * den_im) / den_norm_sq

            # exp(-2 * k * thickness[j])
            # -2*k = -2*(k_re + j*k_im) => real part: -2*k_re, imag part: -2*k_im
            # exp(a+jb) = exp(a) * (cos(b) + j*sin(b))
            t = thickness[j]
            exp_real_part = -2 * k_re * t
            exp_imag_part = -2 * k_im * t
            exp_mag = torch.exp(exp_real_part)
            exp_cos = torch.cos(exp_imag_part)
            exp_sin = torch.sin(exp_imag_part)
            e_re = exp_mag * exp_cos
            e_im = exp_mag * exp_sin

            # r * exp = (r_re + j*r_im) * (e_re + j*e_im)
            re_re = r_re * e_re - r_im * e_im
            re_im = r_re * e_im + r_im * e_re

            # numerator: 1 - r*exp
            num2_re = 1.0 - re_re
            num2_im = -re_im

            # denominator: 1 + r*exp
            den2_re = 1.0 + re_re
            den2_im = re_im

            # Z = Z_j * num2 / den2
            # 先算 num2/den2
            den2_norm_sq = den2_re ** 2 + den2_im ** 2
            frac_re = (num2_re * den2_re + num2_im * den2_im) / den2_norm_sq
            frac_im = (num2_im * den2_re - num2_re * den2_im) / den2_norm_sq

            # Z = Z_j * frac
            Z_re = Zj_re * frac_re - Zj_im * frac_im
            Z_im = Zj_re * frac_im + Zj_im * frac_re

        # |Z|^2 = Z_re^2 + Z_im^2
        Z_abs_sq = Z_re ** 2 + Z_im ** 2
        app_res_list.append(Z_abs_sq / (omega * mu))

        # phase = atan2(Z_im, Z_re) in degrees
        phase_rad = torch.atan2(Z_im, Z_re)
        phase_list.append(phase_rad * (180.0 / np.pi))

    app_res = torch.stack(app_res_list)
    phase = torch.stack(phase_list)

    return app_res, phase


class MTPhysicsLoss(nn.Module):
    """MT 物理一致性损失"""

    def __init__(
        self,
        thickness: np.ndarray,
        frequencies: np.ndarray,
        lambda_rhoa: float = 1.0,
        lambda_phase: float = 10.0,
        adaptive_weighting: bool = True
    ):
        """
        Args:
            thickness: 层厚度 (Nz-1,) [m]
            frequencies: 频率数组 (Nf,) [Hz]
            lambda_rhoa: 视电阻率损失权重
            lambda_phase: 相位损失权重
            adaptive_weighting: 是否使用自适应权重
        """
        super(MTPhysicsLoss, self).__init__()

        self.register_buffer(
            'thickness',
            torch.tensor(thickness, dtype=torch.float32)
        )
        self.register_buffer(
            'frequencies',
            torch.tensor(frequencies, dtype=torch.float32)
        )
        self.lambda_rhoa = lambda_rhoa
        self.lambda_phase = lambda_phase
        self.adaptive_weighting = adaptive_weighting
        self.eps = 1e-8
        self.frequency_mask = None
        self.mse = nn.MSELoss()

    def set_frequency_subset(self, freq_subset: np.ndarray):
        """
        设置频率子集

        Args:
            freq_subset: 频率子集
        """
        self.frequencies = torch.tensor(freq_subset, dtype=torch.float32,
                                         device=self.thickness.device)

    def update_weights(
        self,
        loss_rhoa: torch.Tensor,
        loss_phase: torch.Tensor,
        model: nn.Module
    ):
        """
        更新自适应权重

        Args:
            loss_rhoa: 视电阻率损失
            loss_phase: 相位损失
            model: PINN 模型
        """
        grads_rhoa = torch.autograd.grad(
            loss_rhoa,
            model.parameters(),
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        grads_phase = torch.autograd.grad(
            loss_phase,
            model.parameters(),
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        norm_rhoa = sum(g.norm() for g in grads_rhoa if g is not None)
        norm_phase = sum(g.norm() for g in grads_phase if g is not None)

        self.lambda_rhoa = norm_phase / (norm_rhoa + self.eps)
        self.lambda_phase = norm_rhoa / (norm_phase + self.eps)

    def forward(
        self,
        model: nn.Module,
        sampler,
        observed_rhoa: torch.Tensor,
        observed_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        计算物理一致性损失

        Args:
            model: PINN 网络
            sampler: 深度采样器
            observed_rhoa: 观测视电阻率 (Nf,)
            observed_phase: 观测相位 (Nf,)

        Returns:
            total_loss: 总损失
        """
        # 使用固定网格采样（而不是随机采样）
        # 这样可以确保每次采样的深度点位置一致，与thickness数组匹配
        z = sampler.sample_grid()

        # 网络预测 log_resistivity
        log_resistivity = model(z)

        # 转换为电阻率
        resistivity = 10 ** log_resistivity.squeeze()

        # 可微分的 MT 正演
        rhoa_pred, phase_pred = forward_mt_1d_torch(
            resistivity,
            self.thickness,
            self.frequencies
        )

        # 在 log-domain 计算视电阻率误差
        log_rhoa_pred = torch.log10(rhoa_pred)
        log_rhoa_obs = torch.log10(observed_rhoa)
        loss_rhoa = self.mse(log_rhoa_pred, log_rhoa_obs)

        # 计算相位误差
        loss_phase = self.mse(phase_pred, observed_phase)

        # 自适应权重调整（在计算总损失前）
        if self.adaptive_weighting:
            self.update_weights(loss_rhoa, loss_phase, model)

        # 总损失
        total_loss = self.lambda_rhoa * loss_rhoa + self.lambda_phase * loss_phase

        return {
            "total_loss": total_loss,
            "loss_rhoa": loss_rhoa.detach(),
            "loss_phase": loss_phase.detach()
        }

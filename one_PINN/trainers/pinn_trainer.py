"""PINN 训练器"""

import torch
import torch.nn as nn
from typing import List, Dict


class PINNTrainer:
    """PINN 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        sampler,
        learning_rate: float,
        device: str
    ):
        """
        Args:
            model: PINN 网络
            loss_fn: 损失函数
            sampler: 深度采样器
            learning_rate: 学习率
            device: 设备
        """
        self.model = model
        self.loss_fn = loss_fn
        self.sampler = sampler
        self.device = device
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 初始化 Adam 优化器
        self.optimizer_adam = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # 初始化 LBFGS 优化器
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
    
    def train_adam(
        self,
        num_epochs: int,
        observed_rhoa: torch.Tensor,
        observed_phase: torch.Tensor,
        verbose: bool = True
    ) -> List[float]:
        """
        Adam 训练
        
        Args:
            num_epochs: 训练轮数
            observed_rhoa: 观测视电阻率 (Nf,)
            observed_phase: 观测相位 (Nf,)
            verbose: 是否打印信息
            
        Returns:
            loss_history: 损失历史
        """
        loss_history = []
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.optimizer_adam.zero_grad()
            
            losses = self.loss_fn(
                self.model,
                self.sampler,
                observed_rhoa,
                observed_phase
            )
            
            loss = losses["total_loss"]
            loss.backward()
            self.optimizer_adam.step()
            
            loss_history.append(loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Adam Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        return loss_history
    
    def train_lbfgs(
        self,
        observed_rhoa: torch.Tensor,
        observed_phase: torch.Tensor,
        num_steps: int = 100,
        verbose: bool = True
    ) -> List[float]:
        """
        LBFGS 训练
        
        Args:
            observed_rhoa: 观测视电阻率 (Nf,)
            observed_phase: 观测相位 (Nf,)
            num_steps: 训练步数
            verbose: 是否打印信息
            
        Returns:
            loss_history: 损失历史
        """
        loss_history = []
        
        self.model.train()
        
        for step in range(num_steps):
            def closure():
                self.optimizer_lbfgs.zero_grad()
                losses = self.loss_fn(
                    self.model,
                    self.sampler,
                    observed_rhoa,
                    observed_phase
                )
                loss = losses["total_loss"]
                loss.backward()
                return loss
            
            loss = self.optimizer_lbfgs.step(closure)
            loss_history.append(loss.item())
            
            if verbose and (step + 1) % 10 == 0:
                print(f"LBFGS Step [{step + 1}/{num_steps}], Loss: {loss.item():.6f}")
        
        return loss_history
    
    def train_full(
        self,
        adam_epochs: int,
        lbfgs_steps: int,
        observed_rhoa: torch.Tensor,
        observed_phase: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            adam_epochs: Adam 训练轮数
            lbfgs_steps: LBFGS 训练步数
            observed_rhoa: 观测视电阻率 (Nf,)
            observed_phase: 观测相位 (Nf,)
            verbose: 是否打印信息
            
        Returns:
            history: 包含 'adam' 和 'lbfgs' 损失历史的字典
        """
        if verbose:
            print("=" * 50)
            print("Phase 1: Adam Optimization")
            print("=" * 50)
        
        adam_history = self.train_adam(
            adam_epochs,
            observed_rhoa,
            observed_phase,
            verbose
        )
        
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 2: LBFGS Optimization")
            print("=" * 50)
        
        lbfgs_history = self.train_lbfgs(
            observed_rhoa,
            observed_phase,
            lbfgs_steps,
            verbose
        )
        
        return {
            'adam': adam_history,
            'lbfgs': lbfgs_history
        }
    
    def train_curriculum(
        self,
        observed_rhoa: torch.Tensor,
        observed_phase: torch.Tensor,
        frequencies: torch.Tensor,
        adam_epochs_stage1: int,
        adam_epochs_stage2: int,
        adam_epochs_stage3: int,
        lbfgs_steps: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        课程学习训练
        
        Args:
            observed_rhoa: 观测视电阻率 (Nf,)
            observed_phase: 观测相位 (Nf,)
            frequencies: 频率数组 (Nf,)
            adam_epochs_stage1: 阶段1训练轮数
            adam_epochs_stage2: 阶段2训练轮数
            adam_epochs_stage3: 阶段3训练轮数
            lbfgs_steps: LBFGS训练步数
            verbose: 是否打印信息
            
        Returns:
            history: 包含各阶段损失历史的字典
        """
        # 按频率排序
        freq_sorted, indices = torch.sort(frequencies)
        rhoa_sorted = observed_rhoa[indices]
        phase_sorted = observed_phase[indices]
        
        n_freq = len(frequencies)
        n_low = int(n_freq * 0.3)
        n_mid = int(n_freq * 0.6)
        
        # 保存原始频率
        original_frequencies = self.loss_fn.frequencies.clone()

        # Stage 1: 低频
        if verbose:
            print("=" * 50)
            print("Stage 1: Low Frequency Training (30%)")
            print("=" * 50)

        self.loss_fn.frequencies = freq_sorted[:n_low]
        stage1_history = self.train_adam(
            adam_epochs_stage1,
            rhoa_sorted[:n_low],
            phase_sorted[:n_low],
            verbose
        )
        
        # Stage 2: 中频
        if verbose:
            print("\n" + "=" * 50)
            print("Stage 2: Mid Frequency Training (60%)")
            print("=" * 50)
        
        self.loss_fn.frequencies = freq_sorted[:n_mid]
        stage2_history = self.train_adam(
            adam_epochs_stage2,
            rhoa_sorted[:n_mid],
            phase_sorted[:n_mid],
            verbose
        )
        
        # Stage 3: 全频
        if verbose:
            print("\n" + "=" * 50)
            print("Stage 3: Full Frequency Training (100%)")
            print("=" * 50)
        
        self.loss_fn.frequencies = freq_sorted
        stage3_history = self.train_adam(
            adam_epochs_stage3,
            rhoa_sorted,
            phase_sorted,
            verbose
        )
        
        # LBFGS 精细优化
        if verbose:
            print("\n" + "=" * 50)
            print("Stage 4: LBFGS Optimization")
            print("=" * 50)
        
        lbfgs_history = self.train_lbfgs(
            rhoa_sorted,
            phase_sorted,
            lbfgs_steps,
            verbose
        )
        
        # 恢复原始频率
        self.loss_fn.frequencies = original_frequencies
        
        return {
            'stage1': stage1_history,
            'stage2': stage2_history,
            'stage3': stage3_history,
            'lbfgs': lbfgs_history
        }

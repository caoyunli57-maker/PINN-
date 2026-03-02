"""PINN 数据加载器"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple
from .preprocessor import prepare_dataset


class MT1DDataset(Dataset):
    """MT 1D 数据集类"""
    
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Args:
            inputs: 输入数据 (N, Nf, 2)
            labels: 标签数据 (N, Nz)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


class MT1DDataLoader:
    """MT 1D 数据加载器管理类"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.train_ratio = config['data']['train_ratio']
        self.val_ratio = config['data']['val_ratio']
        self.test_ratio = config['data']['test_ratio']
        self.seed = config.get('seed', 42)
    
    def load_data(
        self,
        dataset_path: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """
        加载并准备数据
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            metadata: 元数据字典 (depths, frequencies, preprocessor)
        """
        # 加载和预处理数据
        prepared_data = prepare_dataset(
            dataset_path,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
            self.seed
        )
        
        # 创建数据集
        train_dataset = MT1DDataset(
            prepared_data['train_inputs'],
            prepared_data['train_labels']
        )
        val_dataset = MT1DDataset(
            prepared_data['val_inputs'],
            prepared_data['val_labels']
        )
        test_dataset = MT1DDataset(
            prepared_data['test_inputs'],
            prepared_data['test_labels']
        )
        
        # 创建数据加载器
        train_loader = self.create_dataloader(train_dataset, shuffle=True)
        val_loader = self.create_dataloader(val_dataset, shuffle=False)
        test_loader = self.create_dataloader(test_dataset, shuffle=False)
        
        # 元数据
        metadata = {
            'depths': prepared_data['depths'],
            'frequencies': prepared_data['frequencies'],
            'preprocessor': prepared_data['preprocessor']
        }
        
        return train_loader, val_loader, test_loader, metadata
    
    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True
    ) -> DataLoader:
        """
        创建 DataLoader
        
        Args:
            dataset: PyTorch Dataset
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )

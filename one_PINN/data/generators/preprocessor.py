"""数据预处理模块"""

import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler


class MT1DPreprocessor:
    """1D MT 数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.input_scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        拟合归一化参数
        
        Args:
            inputs: 输入数据 (N, Nf, 2)
            labels: 标签数据 (N, Nz)
        """
        # 展平输入数据进行拟合
        N, Nf, C = inputs.shape
        inputs_flat = inputs.reshape(-1, C)
        
        # 拟合输入归一化器
        self.input_scaler.fit(inputs_flat)
        
        # 拟合标签归一化器
        self.label_scaler.fit(labels)
        
        self.is_fitted = True
    
    def transform(
        self,
        inputs: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用归一化变换
        
        Args:
            inputs: 输入数据 (N, Nf, 2)
            labels: 标签数据 (N, Nz)
            
        Returns:
            normalized_inputs: 归一化输入 (N, Nf, 2)
            normalized_labels: 归一化标签 (N, Nz)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # 归一化输入
        N, Nf, C = inputs.shape
        inputs_flat = inputs.reshape(-1, C)
        normalized_inputs_flat = self.input_scaler.transform(inputs_flat)
        normalized_inputs = normalized_inputs_flat.reshape(N, Nf, C)
        
        # 归一化标签
        normalized_labels = self.label_scaler.transform(labels)
        
        return normalized_inputs, normalized_labels
    
    def fit_transform(
        self,
        inputs: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并应用归一化变换
        
        Args:
            inputs: 输入数据 (N, Nf, 2)
            labels: 标签数据 (N, Nz)
            
        Returns:
            normalized_inputs: 归一化输入 (N, Nf, 2)
            normalized_labels: 归一化标签 (N, Nz)
        """
        self.fit(inputs, labels)
        return self.transform(inputs, labels)
    
    def inverse_transform_labels(self, normalized_labels: np.ndarray) -> np.ndarray:
        """
        反归一化标签
        
        Args:
            normalized_labels: 归一化标签 (N, Nz)
            
        Returns:
            labels: 原始标签 (N, Nz)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        return self.label_scaler.inverse_transform(normalized_labels)


def split_dataset(
    inputs: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    划分数据集
    
    Args:
        inputs: 输入数据 (N, Nf, 2)
        labels: 标签数据 (N, Nz)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        splits: 包含 'train', 'val', 'test' 的字典
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    
    n_samples = len(inputs)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    splits = {
        'train': (inputs[train_indices], labels[train_indices]),
        'val': (inputs[val_indices], labels[val_indices]),
        'test': (inputs[test_indices], labels[test_indices])
    }
    
    return splits


def prepare_dataset(
    dataset_path: str,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    加载、预处理和划分数据集
    
    Args:
        dataset_path: 数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        prepared_data: 包含以下键的字典
            - 'train_inputs': (N_train, Nf, 2)
            - 'train_labels': (N_train, Nz)
            - 'val_inputs': (N_val, Nf, 2)
            - 'val_labels': (N_val, Nz)
            - 'test_inputs': (N_test, Nf, 2)
            - 'test_labels': (N_test, Nz)
            - 'depths': (Nz,)
            - 'frequencies': (Nf,)
            - 'preprocessor': MT1DPreprocessor 对象
    """
    # 加载数据集
    data = np.load(dataset_path)
    inputs = data['inputs']
    labels = data['labels']
    depths = data['depths']
    frequencies = data['frequencies']
    
    # 划分数据集
    splits = split_dataset(
        inputs, labels,
        train_ratio, val_ratio, test_ratio,
        seed
    )
    
    # 初始化预处理器并在训练集上拟合
    preprocessor = MT1DPreprocessor()
    train_inputs_norm, train_labels_norm = preprocessor.fit_transform(
        splits['train'][0],
        splits['train'][1]
    )
    
    # 变换验证集和测试集
    val_inputs_norm, val_labels_norm = preprocessor.transform(
        splits['val'][0],
        splits['val'][1]
    )
    test_inputs_norm, test_labels_norm = preprocessor.transform(
        splits['test'][0],
        splits['test'][1]
    )
    
    prepared_data = {
        'train_inputs': train_inputs_norm,
        'train_labels': train_labels_norm,
        'val_inputs': val_inputs_norm,
        'val_labels': val_labels_norm,
        'test_inputs': test_inputs_norm,
        'test_labels': test_labels_norm,
        'depths': depths,
        'frequencies': frequencies,
        'preprocessor': preprocessor
    }
    
    return prepared_data

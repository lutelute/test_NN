"""
モジュラー加算タスクのデータセット生成
(a + b + c) mod p のタスク（3値版）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ModularAdditionDataset(Dataset):
    """
    モジュラー加算データセット（3値版）
    入力: (a, b, c) where a, b, c ∈ {0, 1, ..., p-1}
    出力: (a + b + c) mod p
    """

    def __init__(self, p: int, indices: np.ndarray = None, n_tokens: int = 3):
        """
        Args:
            p: 素数（モジュロの基数）
            indices: 使用するサンプルのインデックス（None の場合は全サンプル）
            n_tokens: 入力トークン数（デフォルト3）
        """
        self.p = p
        self.n_tokens = n_tokens

        if n_tokens == 2:
            # 2値版（後方互換性のため）
            all_a = np.arange(p)
            all_b = np.arange(p)
            aa, bb = np.meshgrid(all_a, all_b)
            self.all_inputs = np.stack([aa.flatten(), bb.flatten()], axis=1)
            self.all_labels = (self.all_inputs[:, 0] + self.all_inputs[:, 1]) % p
        elif n_tokens == 3:
            # 3値版
            all_a = np.arange(p)
            all_b = np.arange(p)
            all_c = np.arange(p)
            aa, bb, cc = np.meshgrid(all_a, all_b, all_c, indexing='ij')
            self.all_inputs = np.stack([aa.flatten(), bb.flatten(), cc.flatten()], axis=1)
            self.all_labels = (self.all_inputs[:, 0] + self.all_inputs[:, 1] + self.all_inputs[:, 2]) % p
        else:
            raise ValueError(f"n_tokens must be 2 or 3, got {n_tokens}")

        # インデックスでフィルタリング
        if indices is not None:
            self.inputs = self.all_inputs[indices]
            self.labels = self.all_labels[indices]
        else:
            self.inputs = self.all_inputs
            self.labels = self.all_labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs, label


def create_train_test_split(p: int, train_ratio: float = 0.5, seed: int = 42, n_tokens: int = 3):
    """
    Train/Test 分割を作成

    Args:
        p: 素数
        train_ratio: 訓練データの割合
        seed: 乱数シード
        n_tokens: 入力トークン数

    Returns:
        train_dataset, test_dataset
    """
    np.random.seed(seed)

    if n_tokens == 2:
        total_samples = p * p
    elif n_tokens == 3:
        total_samples = p * p * p
    else:
        raise ValueError(f"n_tokens must be 2 or 3, got {n_tokens}")

    indices = np.random.permutation(total_samples)

    train_size = int(total_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = ModularAdditionDataset(p, train_indices, n_tokens)
    test_dataset = ModularAdditionDataset(p, test_indices, n_tokens)

    return train_dataset, test_dataset


def get_dataloaders(p: int = 113, train_ratio: float = 0.5, batch_size: int = None,
                    seed: int = 42, n_tokens: int = 3, num_workers: int = 0):
    """
    DataLoader を作成

    Args:
        p: 素数
        train_ratio: 訓練データの割合
        batch_size: バッチサイズ（None の場合は全データ）
        seed: 乱数シード
        n_tokens: 入力トークン数
        num_workers: データ読み込みの並列ワーカー数

    Returns:
        train_loader, test_loader, p
    """
    train_dataset, test_dataset = create_train_test_split(p, train_ratio, seed, n_tokens)

    if batch_size is None:
        batch_size = len(train_dataset)  # full batch

    # M4 Mac向け最適化: persistent_workers で高速化
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, test_loader, p


if __name__ == "__main__":
    # テスト（3値版）
    p = 113
    n_tokens = 3
    train_dataset, test_dataset = create_train_test_split(p, n_tokens=n_tokens)

    print(f"p = {p}")
    print(f"n_tokens = {n_tokens}")
    print(f"Total samples: {p ** n_tokens}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # サンプル表示
    inputs, label = train_dataset[0]
    print(f"\nSample: ({inputs[0].item()}, {inputs[1].item()}, {inputs[2].item()}) -> {label.item()}")
    expected = (inputs[0].item() + inputs[1].item() + inputs[2].item()) % p
    print(f"Verification: ({inputs[0].item()} + {inputs[1].item()} + {inputs[2].item()}) mod {p} = {expected}")

#!/usr/bin/env python3
"""
Grokking Demo - 軽量学習スクリプト
10エポック間隔でチェックポイントを保存し、Grokkingの過程を詳細に観察

Usage:
    python train_grokking_demo.py --linear  # 並列化なし（安定）
    python train_grokking_demo.py           # 通常モード
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

from model import ModularAdditionTransformer


class GrokingDemoTrainer:
    """Grokking デモ用軽量トレーナー（2トークン版）"""

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1.0,
        train_ratio: float = 0.3,
        checkpoint_dir: str = "checkpoints_demo",
        linear_mode: bool = False,
    ):
        self.p = p
        self.linear_mode = linear_mode
        self.checkpoint_dir = checkpoint_dir

        # リニアモードではCPUを使用（安定性重視）
        if linear_mode:
            self.device = "cpu"
            print("Linear mode: Using CPU for stable sequential computation")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # データ生成（2トークン: a + b mod p）
        self.train_data, self.test_data = self._create_data(p, train_ratio)
        print(f"Train samples: {len(self.train_data[0])}, Test samples: {len(self.test_data[0])}")

        # モデル（2トークン版）
        self.model = ModularAdditionTransformer(
            p=p,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_tokens=2,  # 2トークン
        ).to(self.device)

        # オプティマイザ
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.config = {
            "p": p,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "n_tokens": 2,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_ratio": train_ratio,
            "linear_mode": linear_mode,
        }

    def _create_data(self, p: int, train_ratio: float):
        """2トークンデータを生成（a + b mod p）"""
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        np.random.seed(42)
        np.random.shuffle(all_pairs)

        split = int(len(all_pairs) * train_ratio)
        train_pairs = all_pairs[:split]
        test_pairs = all_pairs[split:]

        def pairs_to_tensors(pairs):
            x = torch.tensor([[a, b] for a, b in pairs])
            y = torch.tensor([(a + b) % p for a, b in pairs])
            return x, y

        return pairs_to_tensors(train_pairs), pairs_to_tensors(test_pairs)

    def train_epoch(self):
        """1エポックの訓練（リニアモード対応）"""
        self.model.train()

        x, y = self.train_data
        x, y = x.to(self.device), y.to(self.device)

        if self.linear_mode:
            # リニアモード: バッチを小分けにして順次処理
            batch_size = 256
            total_loss = 0
            total_correct = 0

            indices = torch.randperm(len(x))
            for i in range(0, len(x), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_x, batch_y = x[batch_idx], y[batch_idx]

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(batch_x)
                total_correct += (logits.argmax(dim=-1) == batch_y).sum().item()

            return total_loss / len(x), total_correct / len(x)
        else:
            # 通常モード: 全データ一括
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            return loss.item(), acc

    @torch.no_grad()
    def evaluate(self):
        """テストデータで評価"""
        self.model.eval()
        x, y = self.test_data
        x, y = x.to(self.device), y.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()

        return loss.item(), acc

    def train(self, epochs: int = 5000, save_interval: int = 10, log_interval: int = 100):
        """
        学習を実行

        Args:
            epochs: エポック数
            save_interval: チェックポイント保存間隔（デフォルト10）
            log_interval: ログ出力間隔
        """
        print(f"Starting training for {epochs} epochs (save every {save_interval} epochs)...")
        print(f"Config: {self.config}")

        best_test_acc = 0
        grokking_epoch = None

        pbar = tqdm(range(1, epochs + 1), desc="Training")

        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)

            # Grokking検出
            if grokking_epoch is None and train_acc > 0.99 and test_acc > 0.9:
                grokking_epoch = epoch
                print(f"\n*** Grokking detected at epoch {epoch}! ***")

            pbar.set_postfix({
                "train_acc": f"{train_acc:.1%}",
                "test_acc": f"{test_acc:.1%}",
            })

            if epoch % log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch:5d} | "
                    f"Train: {train_acc:.1%} | Test: {test_acc:.1%}"
                )

            # チェックポイント保存（10エポック間隔）
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)

            # ベストモデル保存
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_checkpoint(epoch, is_best=True)

        self.save_checkpoint(epochs, final=True)
        self.save_history()

        print(f"\nTraining completed!")
        print(f"Best test accuracy: {best_test_acc:.1%}")
        if grokking_epoch:
            print(f"Grokking occurred at epoch: {grokking_epoch}")

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }

        if final:
            path = os.path.join(self.checkpoint_dir, "final.pt")
        elif is_best:
            path = os.path.join(self.checkpoint_dir, "best.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")

        torch.save(checkpoint, path)

    def save_history(self):
        """学習履歴をJSONで保存"""
        history_path = os.path.join(self.checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grokking Demo Training")
    parser.add_argument("--p", type=int, default=97, help="Prime number")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--linear", action="store_true", help="Use linear (sequential) mode for stable training")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Training data ratio")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_demo", help="Checkpoint directory")

    args = parser.parse_args()

    trainer = GrokingDemoTrainer(
        p=args.p,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        checkpoint_dir=args.checkpoint_dir,
        linear_mode=args.linear,
    )

    trainer.train(
        epochs=args.epochs,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()

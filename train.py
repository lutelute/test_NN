"""
Grokking学習スクリプト
モジュラー加算タスクでTransformerを学習し、Grokking現象を観察
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

from data import get_dataloaders
from model import ModularAdditionTransformer


class Trainer:
    """Grokking学習用トレーナー"""

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1.0,
        train_ratio: float = 0.5,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.p = p
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir

        print(f"Using device: {self.device}")

        # データローダー
        self.train_loader, self.test_loader, _ = get_dataloaders(p, train_ratio)

        # モデル
        self.model = ModularAdditionTransformer(
            p=p,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(self.device)

        # オプティマイザ（Weight decayが重要！）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # 損失関数
        self.criterion = nn.CrossEntropyLoss()

        # ログ
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        # チェックポイントディレクトリ
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 設定を保存
        self.config = {
            "p": p,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_ratio": train_ratio,
        }

    def train_epoch(self):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self):
        """テストデータで評価"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def train(self, epochs: int = 50000, log_interval: int = 100, save_interval: int = 5000, start_epoch: int = 1):
        """
        学習を実行

        Args:
            epochs: エポック数
            log_interval: ログ出力間隔
            save_interval: チェックポイント保存間隔
            start_epoch: 開始エポック（再開時に使用）
        """
        if start_epoch > 1:
            print(f"Resuming training from epoch {start_epoch} to {epochs}...")
        else:
            print(f"Starting training for {epochs} epochs...")
        print(f"Config: {self.config}")

        best_test_acc = max(self.history["test_acc"]) if self.history["test_acc"] else 0
        grokking_epoch = None

        pbar = tqdm(range(start_epoch, epochs + 1), desc="Training")

        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)

            # Grokking検出（train acc > 99% かつ test acc が急上昇）
            if grokking_epoch is None and train_acc > 0.99 and test_acc > 0.9:
                grokking_epoch = epoch
                print(f"\n*** Grokking detected at epoch {epoch}! ***")

            # 進捗表示
            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2%}",
                "test_acc": f"{test_acc:.2%}",
            })

            # 詳細ログ
            if epoch % log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch:5d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}"
                )

            # チェックポイント保存
            if epoch % save_interval == 0 or test_acc > best_test_acc:
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch)

        # 最終保存
        self.save_checkpoint(epochs, final=True)
        self.save_history()

        print(f"\nTraining completed!")
        print(f"Best test accuracy: {best_test_acc:.2%}")
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

    def load_checkpoint(self, path: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        return checkpoint["epoch"]


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Grokking Training")
    parser.add_argument("--p", type=int, default=97, help="Prime number for modular arithmetic")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of epochs")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Training data ratio")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=5000, help="Checkpoint save interval")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    trainer = Trainer(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        checkpoint_dir=args.checkpoint_dir,
    )

    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resuming from epoch {start_epoch}")

    trainer.train(
        epochs=args.epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()

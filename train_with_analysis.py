"""
Grokking学習スクリプト（フーリエ解析付き）
学習中に定期的にフーリエ解析を実行し、結果を保存
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
from analyze import FourierAnalyzer


class TrainerWithAnalysis:
    """Grokking学習用トレーナー（フーリエ解析付き）"""

    def __init__(
        self,
        p: int = 113,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1.0,
        train_ratio: float = 0.5,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
        n_tokens: int = 3,
        batch_size: int = None,
    ):
        self.p = p
        self.n_tokens = n_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir

        print(f"Using device: {self.device}")

        # データローダー（並列ワーカー有効化）
        self.train_loader, self.test_loader, _ = get_dataloaders(
            p, train_ratio, batch_size=batch_size, n_tokens=n_tokens, num_workers=4
        )

        # モデル
        self.model = ModularAdditionTransformer(
            p=p,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_tokens=n_tokens,
        ).to(self.device)

        # torch.compile() で高速化（PyTorch 2.0+）
        if hasattr(torch, 'compile') and self.device != "mps":
            # MPSはcompileサポートが限定的なので除外
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile() not available: {e}")

        # オプティマイザ
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

        # フーリエ解析結果
        self.fourier_history = {
            "epochs": [],
            "best_correlations": [],
            "dominant_frequencies": [],
            "circularities": [],
            "angle_correlations": [],
            "spectrum_concentrations": [],
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
            "n_tokens": n_tokens,
        }

    def train_epoch(self):
        """1エポックの訓練（Mixed Precision対応）"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # M4 Mac向けMixed Precision
        use_amp = self.device in ["mps", "cuda"]
        dtype = torch.float16 if use_amp else torch.float32

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision forward pass
            with torch.autocast(device_type=self.device if self.device != "mps" else "cpu", dtype=dtype, enabled=use_amp and self.device == "cuda"):
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            # MPS用: 通常のbackward（MPSはautocastが限定的）
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

    def analyze_fourier(self) -> dict:
        """現在のモデルのフーリエ解析を実行"""
        self.model.eval()

        # CPU上でモデルのコピーを作成して解析
        model_cpu = ModularAdditionTransformer(
            p=self.config["p"],
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            n_layers=self.config["n_layers"],
            n_tokens=self.config.get("n_tokens", 3),
        )
        model_cpu.load_state_dict({k: v.cpu() for k, v in self.model.state_dict().items()})
        model_cpu.eval()

        analyzer = FourierAnalyzer(model_cpu)

        # フーリエ解析
        fourier_result = analyzer.verify_fourier_representation()
        circular_result = analyzer.analyze_circular_structure()

        return {
            "best_correlation": fourier_result["best_correlation"],
            "dominant_frequencies": fourier_result["dominant_frequencies"],
            "spectrum_concentration": fourier_result["spectrum_concentration"],
            "circularity": circular_result["circularity"],
            "angle_correlation": circular_result["angle_correlation"],
            "is_fourier": fourier_result["is_fourier_representation"],
            "is_circular": circular_result["is_circular"],
        }

    def train(
        self,
        epochs: int = 50000,
        log_interval: int = 100,
        save_interval: int = 1000,
        analysis_interval: int = 100,
        start_epoch: int = 1,
    ):
        """
        学習を実行

        Args:
            epochs: エポック数
            log_interval: ログ出力間隔
            save_interval: チェックポイント保存間隔
            analysis_interval: フーリエ解析間隔
            start_epoch: 開始エポック（再開時に使用）
        """
        if start_epoch > 1:
            print(f"Resuming training from epoch {start_epoch} to {epochs}...")
        else:
            print(f"Starting training for {epochs} epochs...")
        print(f"Config: {self.config}")

        best_test_acc = max(self.history["test_acc"]) if self.history["test_acc"] else 0
        grokking_epoch = None
        fourier_match_epoch = None

        pbar = tqdm(range(start_epoch, epochs + 1), desc="Training")

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

            # フーリエ解析
            if epoch % analysis_interval == 0 or epoch == 1:
                analysis = self.analyze_fourier()

                self.fourier_history["epochs"].append(epoch)
                self.fourier_history["best_correlations"].append(analysis["best_correlation"])
                self.fourier_history["dominant_frequencies"].append(analysis["dominant_frequencies"])
                self.fourier_history["circularities"].append(analysis["circularity"])
                self.fourier_history["angle_correlations"].append(analysis["angle_correlation"])
                self.fourier_history["spectrum_concentrations"].append(analysis["spectrum_concentration"])

                # フーリエ表現の完成を検出
                if fourier_match_epoch is None and analysis["is_fourier"]:
                    fourier_match_epoch = epoch
                    print(f"\n*** Fourier representation formed at epoch {epoch}! ***")

            # 進捗表示
            pbar.set_postfix({
                "train_acc": f"{train_acc:.2%}",
                "test_acc": f"{test_acc:.2%}",
            })

            # 詳細ログ
            if epoch % log_interval == 0:
                fourier_corr = self.fourier_history["best_correlations"][-1] if self.fourier_history["best_correlations"] else 0
                tqdm.write(
                    f"Epoch {epoch:5d} | "
                    f"Train: {train_acc:.2%} | Test: {test_acc:.2%} | "
                    f"Fourier Corr: {fourier_corr:.3f}"
                )

            # チェックポイント保存
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_checkpoint(epoch, is_best=True)

        # 最終保存
        self.save_checkpoint(epochs, final=True)
        self.save_history()
        self.save_fourier_history()

        print(f"\nTraining completed!")
        print(f"Best test accuracy: {best_test_acc:.2%}")
        if grokking_epoch:
            print(f"Grokking occurred at epoch: {grokking_epoch}")
        if fourier_match_epoch:
            print(f"Fourier representation formed at epoch: {fourier_match_epoch}")

        return self.history, self.fourier_history

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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

    def save_fourier_history(self):
        """フーリエ解析履歴をJSONで保存"""
        fourier_path = os.path.join(self.checkpoint_dir, "fourier_history.json")

        # numpy float32 を Python float に変換
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        serializable_history = convert_to_serializable(self.fourier_history)

        with open(fourier_path, "w") as f:
            json.dump(serializable_history, f)

    def load_checkpoint(self, path: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 履歴も読み込む
        history_path = os.path.join(self.checkpoint_dir, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    self.history = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load history.json: {e}")

        fourier_path = os.path.join(self.checkpoint_dir, "fourier_history.json")
        if os.path.exists(fourier_path):
            try:
                with open(fourier_path, "r") as f:
                    self.fourier_history = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load fourier_history.json: {e}")
                # 既存のフーリエ履歴が壊れている場合は、新しく開始
                self.fourier_history = {
                    "epochs": [],
                    "best_correlations": [],
                    "dominant_frequencies": [],
                    "circularities": [],
                    "angle_correlations": [],
                    "spectrum_concentrations": [],
                }

        return checkpoint["epoch"]


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Grokking Training with Fourier Analysis")
    parser.add_argument("--p", type=int, default=97, help="Prime number")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Training data ratio")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--analysis_interval", type=int, default=50, help="Fourier analysis interval")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--n_tokens", type=int, default=3, help="Number of input tokens (2 or 3)")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size (default 4096)")

    args = parser.parse_args()

    trainer = TrainerWithAnalysis(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        checkpoint_dir=args.checkpoint_dir,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
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
        analysis_interval=args.analysis_interval,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()

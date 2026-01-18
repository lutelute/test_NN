"""
Grokking モジュラー加算 - 統合実行スクリプト

このスクリプトは以下を実行します:
1. モジュラー加算タスクでTransformerを学習
2. Grokking現象を観察
3. フーリエ解析で学習した表現が数式と一致することを検証
"""

import argparse
import os
import sys

from train import Trainer
from analyze import analyze_checkpoint
from visualize import plot_all


def run_experiment(
    p: int = 97,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    epochs: int = 50000,
    train_ratio: float = 0.5,
    checkpoint_dir: str = "checkpoints",
    figures_dir: str = "figures",
    log_interval: int = 100,
    save_interval: int = 5000,
    skip_training: bool = False,
):
    """
    実験を実行

    Args:
        p: 素数（モジュロの基数）
        d_model: モデル次元
        n_heads: アテンションヘッド数
        n_layers: Transformer層数
        lr: 学習率
        weight_decay: Weight decay（Grokkingに重要）
        epochs: エポック数
        train_ratio: 訓練データ割合
        checkpoint_dir: チェックポイント保存先
        figures_dir: 図の保存先
        log_interval: ログ出力間隔
        save_interval: チェックポイント保存間隔
        skip_training: 学習をスキップするか
    """

    print("=" * 70)
    print("GROKKING EXPERIMENT: Modular Addition")
    print("=" * 70)
    print(f"\nTask: (a + b) mod {p}")
    print(f"Model: {n_layers}-layer Transformer, d_model={d_model}, n_heads={n_heads}")
    print(f"Training: lr={lr}, weight_decay={weight_decay}, epochs={epochs}")
    print()

    # ============================
    # Phase 1: Training
    # ============================
    if not skip_training:
        print("-" * 70)
        print("PHASE 1: Training")
        print("-" * 70)

        trainer = Trainer(
            p=p,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            lr=lr,
            weight_decay=weight_decay,
            train_ratio=train_ratio,
            checkpoint_dir=checkpoint_dir,
        )

        trainer.train(
            epochs=epochs,
            log_interval=log_interval,
            save_interval=save_interval,
        )

    # ============================
    # Phase 2: Fourier Analysis
    # ============================
    print("\n" + "-" * 70)
    print("PHASE 2: Fourier Analysis")
    print("-" * 70)

    checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, "final.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found in {checkpoint_dir}")
        return

    result = analyze_checkpoint(checkpoint_path, p)

    fourier = result["fourier_analysis"]
    circular = result["circular_analysis"]

    print(f"\n*** VERIFICATION RESULTS ***")
    print(f"Is Fourier Representation: {fourier['is_fourier_representation']}")
    print(f"Best Correlation with Theory: {fourier['best_correlation']:.4f}")

    if fourier['is_fourier_representation']:
        print("\n[SUCCESS] NN has learned a Fourier representation!")
        print("The internal weights correspond to the theoretical formula:")
        print("  Embedding(n) ~ [cos(2*pi*k*n/p), sin(2*pi*k*n/p)]")
    else:
        print("\n[INFO] Fourier representation not yet fully formed.")
        print("Try training for more epochs or adjusting hyperparameters.")

    print(f"\nDominant Frequencies:")
    for freq, power in fourier["dominant_frequencies"][:5]:
        print(f"  k={freq}: power={power:.2f}")

    print(f"\nCircular Structure:")
    print(f"  Is Circular: {circular['is_circular']}")
    print(f"  Angle Correlation: {circular['angle_correlation']:.4f}")

    # ============================
    # Phase 3: Visualization
    # ============================
    print("\n" + "-" * 70)
    print("PHASE 3: Visualization")
    print("-" * 70)

    history_path = os.path.join(checkpoint_dir, "history.json")
    plot_all(checkpoint_path, history_path, figures_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {checkpoint_dir}/")
    print(f"Figures saved to: {figures_dir}/")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Grokking Experiment: Verify that NN learns Fourier representation for modular addition"
    )

    # Model parameters
    parser.add_argument("--p", type=int, default=97, help="Prime number for modular arithmetic")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of epochs")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Training data ratio")

    # Output parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--figures_dir", type=str, default="figures", help="Figures directory")

    # Logging parameters
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=5000, help="Checkpoint save interval")

    # Options
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only analyze")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer epochs")

    args = parser.parse_args()

    # Quick test mode
    if args.quick:
        args.epochs = 5000
        args.log_interval = 50
        args.save_interval = 1000

    run_experiment(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        checkpoint_dir=args.checkpoint_dir,
        figures_dir=args.figures_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()

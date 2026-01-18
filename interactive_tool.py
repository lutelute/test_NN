"""
Grokking インタラクティブ解析ツール

エポックスライダーで任意のエポックを選択し、
そのエポックでの各層のフーリエ相関を可視化するツール

図2: 各層の2D射影グリッド（5行×6列）+ 左側波形パネル
図3: エポックスライダー + 学習曲線
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import json
import os
from typing import Dict, List, Tuple
import glob

from model import ModularAdditionTransformer
from analyze import FourierAnalyzer


class InteractiveGrokkingTool:
    """Grokkingの進行をインタラクティブに観察するツール"""

    def __init__(self, checkpoint_dir: str = "checkpoints", p: int = 113):
        self.checkpoint_dir = checkpoint_dir
        self.p = p
        self.checkpoints = []
        self.epochs = []
        self.history = None
        self.fourier_history = None
        self.current_analyzer = None

        # チェックポイントを読み込み
        self._load_data()

    def _load_data(self):
        """データを読み込み"""
        # history.json を読み込み
        history_path = os.path.join(self.checkpoint_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.history = json.load(f)

        # fourier_history.json を読み込み
        fourier_path = os.path.join(self.checkpoint_dir, "fourier_history.json")
        if os.path.exists(fourier_path):
            try:
                with open(fourier_path, "r") as f:
                    self.fourier_history = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load fourier_history.json: {e}")

        # チェックポイントファイルを検索
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "epoch_*.pt"))
        checkpoint_files.extend(glob.glob(os.path.join(self.checkpoint_dir, "best.pt")))
        checkpoint_files.extend(glob.glob(os.path.join(self.checkpoint_dir, "final.pt")))

        for cp_file in checkpoint_files:
            try:
                checkpoint = torch.load(cp_file, map_location="cpu")
                epoch = checkpoint.get("epoch", 0)
                self.checkpoints.append((epoch, cp_file, checkpoint))
            except Exception as e:
                print(f"Warning: Could not load {cp_file}: {e}")

        # エポック順にソート
        self.checkpoints.sort(key=lambda x: x[0])
        self.epochs = [cp[0] for cp in self.checkpoints]

        print(f"Loaded {len(self.checkpoints)} checkpoints")
        print(f"Epochs: {self.epochs}")

    def _load_model(self, checkpoint) -> ModularAdditionTransformer:
        """チェックポイントからモデルを読み込み"""
        config = checkpoint.get("config", {"p": self.p, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})

        model = ModularAdditionTransformer(
            p=config["p"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            n_tokens=config.get("n_tokens", 3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    def run_interactive(self):
        """インタラクティブツールを起動"""
        if len(self.checkpoints) == 0:
            print("No checkpoints found!")
            return

        # 黒背景のスタイル
        plt.style.use('dark_background')

        # 図の設定
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        fig.suptitle('Grokking Analysis Tool - Layer Correlations', color='white', fontsize=14, y=0.98)

        # メインのグリッドレイアウト
        main_gs = fig.add_gridspec(1, 2, width_ratios=[1, 5], left=0.02, right=0.98, top=0.93, bottom=0.12)

        # 左側: 波形パネル（6つ）
        left_gs = main_gs[0].subgridspec(6, 1, hspace=0.4)
        wave_axes = []
        for i in range(6):
            ax = fig.add_subplot(left_gs[i], facecolor='black')
            wave_axes.append(ax)

        # 右側: 5行×6列のグリッド + 学習曲線
        right_gs = main_gs[1].subgridspec(6, 6, height_ratios=[1, 1, 1, 1, 1, 0.8], hspace=0.35, wspace=0.2)

        # 上部5行: 各層の2D射影グリッド
        layer_axes = []
        for row in range(5):
            for col in range(6):
                ax = fig.add_subplot(right_gs[row, col], facecolor='black')
                layer_axes.append(ax)

        # 下部: 学習曲線
        ax_curve = fig.add_subplot(right_gs[5, :], facecolor='black')

        # スライダー用のスペース
        slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02], facecolor='#333333')

        # 初期エポック
        initial_epoch_idx = len(self.epochs) - 1

        # スライダー
        epoch_slider = Slider(
            slider_ax, 'Epoch', 0, len(self.epochs) - 1,
            valinit=initial_epoch_idx, valstep=1,
            color='cyan'
        )

        def update(val):
            """スライダー更新時の処理"""
            idx = int(epoch_slider.val)
            epoch, _, checkpoint = self.checkpoints[idx]
            model = self._load_model(checkpoint)
            self._update_plots(model, epoch, wave_axes, layer_axes, ax_curve, idx)
            fig.canvas.draw_idle()

        epoch_slider.on_changed(update)

        # 初期描画
        epoch, _, checkpoint = self.checkpoints[initial_epoch_idx]
        model = self._load_model(checkpoint)
        self._update_plots(model, epoch, wave_axes, layer_axes, ax_curve, initial_epoch_idx)

        plt.show()

    def _update_plots(self, model: ModularAdditionTransformer, epoch: int,
                     wave_axes: List, layer_axes: List, ax_curve, epoch_idx: int):
        """プロットを更新"""
        analyzer = FourierAnalyzer(model)

        # 各層の出力を取得
        layer_outputs = analyzer.get_layer_outputs()

        # 解析する層のリスト（重要な層のみ選択）- 30個分
        important_layers = []
        # 3トークン分の各位置の出力を追加
        for pos in range(3):
            important_layers.extend([
                f"embed_pos{pos}",
                f"embed_pos_pos{pos}",
                f"block_0_attn_out_pos{pos}",
                f"block_0_post_attn_pos{pos}",
                f"block_0_ff_out_pos{pos}",
                f"block_0_output_pos{pos}",
            ])
        important_layers.extend(["pooled", "logits"])

        # 利用可能な層のみフィルタ
        available_layers = [l for l in important_layers if l in layer_outputs]

        # ========== 左側: 波形パネル ==========
        dominant = analyzer.find_dominant_frequencies(top_k=6)
        n_points = analyzer.p
        n_range = np.arange(n_points)
        wave_colors = ['#9966FF', '#3366FF', '#00CCCC', '#66CC66', '#FFCC00', '#FF6666']

        for i, ax in enumerate(wave_axes):
            ax.clear()
            ax.set_facecolor('black')

            if i < len(dominant):
                freq, power = dominant[i]
                color = wave_colors[i % len(wave_colors)]

                # cos波形を描画
                angles = 2 * np.pi * freq * n_range / n_points
                wave = np.cos(angles)

                ax.plot(n_range, wave, color=color, linewidth=1)
                ax.set_xlim(0, n_points)
                ax.set_ylim(-1.5, 1.5)
                ax.set_ylabel(f'k={freq}', color=color, fontsize=8)
                ax.tick_params(colors='white', labelsize=5)

                for spine in ax.spines.values():
                    spine.set_color('gray')
            else:
                ax.set_visible(False)

        # ========== 右側: 各層の2D射影 ==========
        colors = plt.cm.hsv(np.linspace(0, 1, self.p))

        for i, ax in enumerate(layer_axes):
            ax.clear()
            ax.set_facecolor('black')

            if i < len(available_layers):
                layer_name = available_layers[i]
                weights = layer_outputs[layer_name]

                # 2D射影
                proj_2d = self._compute_2d_projection(weights)

                # 散布図
                ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=8, alpha=0.8)

                # 相関を計算
                result = analyzer._analyze_circular_for_weights(weights)
                angle_corr = result["angle_correlation"]

                # タイトル（短縮）
                short_name = self._shorten_layer_name(layer_name)
                ax.set_title(f'{short_name}\n{angle_corr:.2f}', color='white', fontsize=7)

                ax.set_aspect('equal')
                ax.tick_params(colors='white', labelsize=4)

                # 相関が高い場合は枠を強調
                if angle_corr > 0.8:
                    for spine in ax.spines.values():
                        spine.set_color('lime')
                        spine.set_linewidth(2)
                else:
                    for spine in ax.spines.values():
                        spine.set_color('gray')
                        spine.set_linewidth(0.5)
            else:
                ax.set_visible(False)

        # ========== 下部: 学習曲線 ==========
        ax_curve.clear()
        ax_curve.set_facecolor('black')
        ax_curve.set_title(f'Training Curve (Epoch: {epoch})', color='white', fontsize=11)

        if self.history:
            epochs_hist = range(1, len(self.history["train_acc"]) + 1)
            train_acc = [a * 100 for a in self.history["train_acc"]]
            test_acc = [a * 100 for a in self.history["test_acc"]]

            ax_curve.plot(epochs_hist, train_acc, 'cyan', label='TRAINING', alpha=0.8, linewidth=1.5)
            ax_curve.plot(epochs_hist, test_acc, 'orange', label='TESTING', alpha=0.8, linewidth=1.5)

            # 現在のエポックをハイライト
            ax_curve.axvline(x=epoch, color='lime', linestyle='-', linewidth=2)

            ax_curve.set_xlabel('Epoch', color='white')
            ax_curve.set_ylabel('ACCURACY', color='white')
            ax_curve.tick_params(colors='white')
            ax_curve.legend(loc='lower right', facecolor='black', labelcolor='white', fontsize=9)
            ax_curve.set_ylim(0, 105)
            ax_curve.set_xlim(0, len(self.history["train_acc"]))

            for spine in ax_curve.spines.values():
                spine.set_color('gray')

    def _compute_2d_projection(self, weights: np.ndarray) -> np.ndarray:
        """重みの2D射影を計算"""
        # 最も分散が大きい2次元を選択
        variances = weights.var(axis=0)
        top_2_dims = np.argsort(variances)[-2:]
        return weights[:, top_2_dims]

    def _shorten_layer_name(self, name: str) -> str:
        """層名を短縮"""
        replacements = {
            "embed_pos_pos": "EmbP",
            "embed_pos": "Emb",
            "block_0_attn_out_pos": "Attn",
            "block_0_post_attn_pos": "PAttn",
            "block_0_ff_out_pos": "FF",
            "block_0_output_pos": "Out",
            "pooled": "Pool",
            "logits": "Logit",
        }
        for old, new in replacements.items():
            if name.startswith(old):
                suffix = name[len(old):] if len(name) > len(old) else ""
                return f"{new}{suffix}"
        return name[:8]


def run_detailed_analysis(checkpoint_path: str, p: int = 113, output_dir: str = "detailed_analysis"):
    """
    詳細な解析を実行し、各層の2D射影グリッドを保存

    Args:
        checkpoint_path: チェックポイントのパス
        p: 素数
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {"p": p, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})
    epoch = checkpoint.get("epoch", "unknown")

    model = ModularAdditionTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_tokens=config.get("n_tokens", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    analyzer = FourierAnalyzer(model)

    # 全層の解析
    all_layers = analyzer.analyze_all_layers()

    # 結果をプロット
    plt.style.use('dark_background')
    n_layers = len(all_layers)
    n_cols = 6
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows), facecolor='black')
    fig.suptitle(f'Layer Analysis - Epoch {epoch}', color='white', fontsize=14)

    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, (layer_name, result) in enumerate(all_layers.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.set_facecolor('black')

        # 2D射影
        proj_2d = np.array(result["circular"]["projection_2d"])
        colors = plt.cm.hsv(np.linspace(0, 1, p))

        ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=15, alpha=0.8)

        corr = result["best_correlation"]
        angle_corr = result["circular"]["angle_correlation"]

        ax.set_title(f'{layer_name}\nF:{corr:.2f}, A:{angle_corr:.2f}',
                    color='white', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(colors='white', labelsize=6)

        # 相関が高い場合は枠を強調
        if angle_corr > 0.8:
            for spine in ax.spines.values():
                spine.set_color('lime')
                spine.set_linewidth(2)
        else:
            for spine in ax.spines.values():
                spine.set_color('gray')

    # 余分なaxesを非表示
    for i in range(len(all_layers), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"layer_analysis_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return all_layers


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Grokking Analysis Tool")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--p", type=int, default=113, help="Prime number")
    parser.add_argument("--detailed", type=str, default=None, help="Run detailed analysis on specific checkpoint")

    args = parser.parse_args()

    if args.detailed:
        run_detailed_analysis(args.detailed, args.p)
    else:
        tool = InteractiveGrokkingTool(args.checkpoint_dir, args.p)
        tool.run_interactive()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
プレゼンテーション用 高品質可視化スクリプト
- 白背景・高解像度
- 日本語フォント対応
- Plotlyによるインタラクティブ出力
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # バックエンド設定
import json
import os
from typing import Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from model import ModularAdditionTransformer
from analyze import FourierAnalyzer


def set_presentation_style():
    """プレゼンテーション用スタイル設定"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })


def load_model_from_checkpoint(checkpoint_path: str):
    """チェックポイントからモデルをロード"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {"p": 113, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})

    model = ModularAdditionTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_tokens=config.get("n_tokens", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint.get("epoch", 0)


def create_grokking_figure(history: Dict, save_path: str):
    """Grokking現象の美しい可視化"""
    set_presentation_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    train_acc = [a * 100 for a in history["train_acc"]]
    test_acc = [a * 100 for a in history["test_acc"]]

    # 精度プロット
    ax = axes[0]
    ax.fill_between(epochs, train_acc, test_acc, alpha=0.2, color='purple', label='Generalization Gap')
    ax.plot(epochs, train_acc, color='#2196F3', linewidth=2.5, label='Training Accuracy')
    ax.plot(epochs, test_acc, color='#F44336', linewidth=2.5, label='Test Accuracy')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Grokkingポイントを検出してマーク
    grokking_epoch = None
    for i, (tr, te) in enumerate(zip(history["train_acc"], history["test_acc"])):
        if tr > 0.99 and te > 0.9 and grokking_epoch is None:
            grokking_epoch = i + 1

    if grokking_epoch:
        ax.axvline(x=grokking_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax.annotate(f'Grokking!\n(epoch {grokking_epoch})',
                   xy=(grokking_epoch, 50), fontsize=12, color='green',
                   ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green'))

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Grokking: Delayed Generalization', fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, len(epochs))

    # ロスプロット
    ax = axes[1]
    ax.semilogy(epochs, history["train_loss"], color='#2196F3', linewidth=2.5, label='Training Loss')
    ax.semilogy(epochs, history["test_loss"], color='#F44336', linewidth=2.5, label='Test Loss')

    if grokking_epoch:
        ax.axvline(x=grokking_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('Loss Evolution', fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_fourier_figure(analyzer: FourierAnalyzer, save_path: str):
    """フーリエ解析の美しい可視化"""
    set_presentation_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    p = analyzer.p
    spectrum = analyzer.compute_fourier_spectrum()
    dominant = analyzer.find_dominant_frequencies(top_k=5)

    # 1. フーリエスペクトル
    ax = axes[0]
    freqs = np.arange(p // 2 + 1)
    bars = ax.bar(freqs, spectrum[:len(freqs)], color='#3F51B5', alpha=0.7, width=0.8)

    # 支配的周波数をハイライト
    colors = ['#FF5722', '#FFC107', '#4CAF50', '#9C27B0', '#00BCD4']
    for i, (freq, power) in enumerate(dominant[:3]):
        if freq < len(freqs):
            bars[freq].set_color(colors[i])
            bars[freq].set_alpha(1.0)
            ax.annotate(f'k={freq}', xy=(freq, power), xytext=(freq, power * 1.15),
                       fontsize=12, fontweight='bold', color=colors[i], ha='center')

    ax.set_xlabel('Frequency k', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power', fontsize=14, fontweight='bold')
    ax.set_title('Fourier Spectrum', fontsize=18, fontweight='bold', pad=15)

    # 2. 埋め込みの2D射影（円周構造）
    ax = axes[1]
    circular_result = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular_result["projection_2d"])

    colors_hsv = plt.cm.hsv(np.linspace(0, 1, p))
    scatter = ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=np.arange(p),
                        cmap='hsv', s=80, alpha=0.9, edgecolors='white', linewidth=0.5)

    # 連続する点を線で結ぶ
    for i in range(p - 1):
        ax.plot([proj_2d[i, 0], proj_2d[i+1, 0]],
               [proj_2d[i, 1], proj_2d[i+1, 1]],
               color='gray', alpha=0.2, linewidth=0.5)
    ax.plot([proj_2d[-1, 0], proj_2d[0, 0]],
           [proj_2d[-1, 1], proj_2d[0, 1]],
           color='gray', alpha=0.2, linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel(f'Dimension {circular_result["top_2_dims"][0]}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Dimension {circular_result["top_2_dims"][1]}', fontsize=14, fontweight='bold')
    ax.set_title(f'Circular Embedding Structure\n(Angle Correlation: {circular_result["angle_correlation"]:.3f})',
                fontsize=16, fontweight='bold', pad=15)
    plt.colorbar(scatter, ax=ax, label='Token Value (0 to p-1)')

    # 3. 理論との比較
    ax = axes[2]
    if dominant:
        k = dominant[0][0]
        n = np.arange(p)
        cos_theory = np.cos(2 * np.pi * k * n / p)
        sin_theory = np.sin(2 * np.pi * k * n / p)

        # 埋め込みの最も相関の高い次元
        weights = analyzer.get_embedding_weights()
        best_dim = np.argmax(np.var(weights, axis=0))
        embed_dim = weights[:, best_dim]
        embed_norm = (embed_dim - embed_dim.mean()) / (embed_dim.std() + 1e-8)

        ax.plot(n, cos_theory, color='#2196F3', linewidth=2.5, alpha=0.7,
               label=f'cos(2πk{k}n/p)')
        ax.plot(n, sin_theory, color='#F44336', linewidth=2.5, alpha=0.7,
               label=f'sin(2πk{k}n/p)')
        ax.plot(n, embed_norm, color='#4CAF50', linewidth=2.5, linestyle='--',
               label='Learned Embedding')

        ax.set_xlabel('Token n', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Fourier Basis Comparison (k={k})', fontsize=18, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_evolution_figure(fourier_history: Dict, history: Dict, save_path: str):
    """学習過程でのフーリエ表現の発展"""
    set_presentation_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = fourier_history["epochs"]

    # 1. フーリエ相関の発展
    ax = axes[0, 0]
    ax.plot(epochs, fourier_history["best_correlations"], color='#9C27B0', linewidth=2.5)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Threshold (0.9)')
    ax.fill_between(epochs, 0, fourier_history["best_correlations"], alpha=0.2, color='#9C27B0')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fourier Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Fourier Representation Development', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)

    # 2. 円環構造の発展
    ax = axes[0, 1]
    ax.plot(epochs, fourier_history["angle_correlations"], color='#00BCD4', linewidth=2.5)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.fill_between(epochs, 0, fourier_history["angle_correlations"], alpha=0.2, color='#00BCD4')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Angle Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Circular Structure Development', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)

    # 3. フーリエ相関 vs テスト精度
    ax = axes[1, 0]
    test_acc_at_fourier = [history["test_acc"][min(e-1, len(history["test_acc"])-1)] * 100
                          for e in epochs]

    ax.scatter(fourier_history["best_correlations"], test_acc_at_fourier,
              c=epochs, cmap='viridis', s=60, alpha=0.8)
    ax.set_xlabel('Fourier Correlation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Fourier Representation → Generalization', fontsize=16, fontweight='bold', pad=15)
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5)

    # カラーバー
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(epochs), vmax=max(epochs)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Epoch', fontsize=12)

    # 4. スペクトル集中度
    ax = axes[1, 1]
    ax.plot(epochs, fourier_history["spectrum_concentrations"], color='#FF5722', linewidth=2.5)
    ax.fill_between(epochs, 0, fourier_history["spectrum_concentrations"], alpha=0.2, color='#FF5722')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectrum Concentration', fontsize=14, fontweight='bold')
    ax.set_title('Frequency Selectivity Development', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_model_diagram(p: int, n_tokens: int, save_path: str):
    """モデルアーキテクチャ図"""
    set_presentation_style()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    ax.set_aspect('equal')

    # 色定義
    embed_color = '#E3F2FD'
    attn_color = '#FFF3E0'
    mlp_color = '#F3E5F5'
    border_color = '#1976D2'
    arrow_color = '#424242'

    def draw_box(ax, x, y, w, h, color, title, subtitle=''):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                             fill=True, facecolor=color, edgecolor=border_color,
                             linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.1, title, ha='center', va='center', fontsize=13, fontweight='bold', zorder=3)
        if subtitle:
            ax.text(x, y - 0.3, subtitle, ha='center', va='center', fontsize=10, color='gray', zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))

    # 入力
    ax.text(0.5, 2, f'Input\n({n_tokens} tokens)', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    ax.text(0.5, 0.8, f'a, b, c ∈ [0, {p-1}]', ha='center', va='center', fontsize=10, color='gray')

    draw_arrow(ax, 1.2, 2, 1.8, 2)

    # Token Embedding
    draw_box(ax, 2.5, 2, 1.2, 1.5, embed_color, 'Token\nEmbed', f'{p} → 128')

    draw_arrow(ax, 3.2, 2, 3.8, 2)

    # Position Encoding
    draw_box(ax, 4.5, 2, 1.2, 1.5, embed_color, 'Position\nEncode', f'{n_tokens} pos')

    draw_arrow(ax, 5.2, 2, 5.8, 2)

    # Attention
    draw_box(ax, 6.5, 2, 1.2, 1.5, attn_color, 'Self\nAttention', '4 heads')

    draw_arrow(ax, 7.2, 2, 7.8, 2)

    # MLP
    draw_box(ax, 8.5, 2, 1.2, 1.5, mlp_color, 'MLP', '128 → 512 → 128')

    draw_arrow(ax, 9.2, 2, 9.8, 2)

    # 出力
    ax.text(10.3, 2, f'Output\n(class)', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#F44336'))
    ax.text(10.3, 0.8, f'(a+b+c) mod {p}', ha='center', va='center', fontsize=10, color='gray')

    # タイトル
    ax.text(5.5, 3.6, f'Transformer for Modular Addition (p={p}, {n_tokens}-token)',
           ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_interactive_html(checkpoint_path: str, history_path: str, output_path: str):
    """Plotlyによるインタラクティブな可視化HTML"""

    # データロード
    model, config, epoch = load_model_from_checkpoint(checkpoint_path)
    p = config["p"]
    n_tokens = config.get("n_tokens", 3)

    with open(history_path, "r") as f:
        history = json.load(f)

    analyzer = FourierAnalyzer(model)

    # サブプロット作成
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Progress (Grokking)',
            'Fourier Spectrum',
            'Embedding Circular Structure',
            'Fourier Basis Comparison'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # 1. 学習曲線
    fig.add_trace(
        go.Scatter(x=epochs, y=[a*100 for a in history["train_acc"]],
                  name='Train Accuracy', line=dict(color='#2196F3', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=[a*100 for a in history["test_acc"]],
                  name='Test Accuracy', line=dict(color='#F44336', width=2)),
        row=1, col=1
    )

    # 2. フーリエスペクトル
    spectrum = analyzer.compute_fourier_spectrum()
    half_p = p // 2 + 1
    dominant = analyzer.find_dominant_frequencies(top_k=3)
    dominant_freqs = [f[0] for f in dominant]

    colors = ['#3F51B5' if i not in dominant_freqs else '#FF5722' for i in range(half_p)]

    fig.add_trace(
        go.Bar(x=list(range(half_p)), y=spectrum[:half_p].tolist(),
              marker_color=colors, name='Spectrum'),
        row=1, col=2
    )

    # 3. 円周構造
    circular_result = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular_result["projection_2d"])

    fig.add_trace(
        go.Scatter(x=proj_2d[:, 0].tolist(), y=proj_2d[:, 1].tolist(),
                  mode='markers+lines',
                  marker=dict(color=list(range(p)), colorscale='HSV', size=8),
                  line=dict(color='gray', width=0.5),
                  name='Embedding',
                  text=[f'Token {i}' for i in range(p)],
                  hovertemplate='Token %{text}<br>x: %{x:.3f}<br>y: %{y:.3f}'),
        row=2, col=1
    )

    # 4. フーリエ基底比較
    if dominant:
        k = dominant[0][0]
        n = np.arange(p)
        cos_theory = np.cos(2 * np.pi * k * n / p)
        sin_theory = np.sin(2 * np.pi * k * n / p)

        weights = analyzer.get_embedding_weights()
        best_dim = np.argmax(np.var(weights, axis=0))
        embed_dim = weights[:, best_dim]
        embed_norm = (embed_dim - embed_dim.mean()) / (embed_dim.std() + 1e-8)

        fig.add_trace(
            go.Scatter(x=n.tolist(), y=cos_theory.tolist(), name=f'cos(2πk{k}n/p)',
                      line=dict(color='#2196F3', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=n.tolist(), y=sin_theory.tolist(), name=f'sin(2πk{k}n/p)',
                      line=dict(color='#F44336', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=n.tolist(), y=embed_norm.tolist(), name='Learned',
                      line=dict(color='#4CAF50', width=2, dash='dash')),
            row=2, col=2
        )

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text=f'Grokking Analysis: {n_tokens}-token Modular Addition (p={p})',
            font=dict(size=20)
        ),
        height=800,
        showlegend=True,
        template='plotly_white'
    )

    # 軸ラベル
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency k", row=1, col=2)
    fig.update_yaxes(title_text="Power", row=1, col=2)
    fig.update_xaxes(title_text=f"Dim {circular_result['top_2_dims'][0]}", row=2, col=1)
    fig.update_yaxes(title_text=f"Dim {circular_result['top_2_dims'][1]}", row=2, col=1)
    fig.update_xaxes(title_text="Token n", row=2, col=2)
    fig.update_yaxes(title_text="Normalized Value", row=2, col=2)

    # HTML保存
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def create_all_presentation_figures(checkpoint_dir: str, output_dir: str):
    """プレゼン用の全図を生成"""

    os.makedirs(output_dir, exist_ok=True)

    # パス設定
    best_checkpoint = os.path.join(checkpoint_dir, "best.pt")
    final_checkpoint = os.path.join(checkpoint_dir, "final.pt")
    history_path = os.path.join(checkpoint_dir, "history.json")
    fourier_history_path = os.path.join(checkpoint_dir, "fourier_history.json")

    # チェックポイント選択
    checkpoint_path = best_checkpoint if os.path.exists(best_checkpoint) else final_checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found in {checkpoint_dir}")
        return

    # モデルロード
    model, config, epoch = load_model_from_checkpoint(checkpoint_path)
    p = config["p"]
    n_tokens = config.get("n_tokens", 3)
    analyzer = FourierAnalyzer(model)

    print(f"Loaded model: p={p}, n_tokens={n_tokens}, epoch={epoch}")

    # 1. モデルアーキテクチャ図
    create_model_diagram(p, n_tokens, os.path.join(output_dir, "01_model_architecture.png"))

    # 2. Grokking現象
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        create_grokking_figure(history, os.path.join(output_dir, "02_grokking.png"))

    # 3. フーリエ解析
    create_fourier_figure(analyzer, os.path.join(output_dir, "03_fourier_analysis.png"))

    # 4. 学習過程の発展
    if os.path.exists(fourier_history_path) and os.path.exists(history_path):
        with open(fourier_history_path, "r") as f:
            fourier_history = json.load(f)
        with open(history_path, "r") as f:
            history = json.load(f)
        create_evolution_figure(fourier_history, history, os.path.join(output_dir, "04_evolution.png"))

    # 5. インタラクティブHTML
    if os.path.exists(history_path):
        create_interactive_html(
            checkpoint_path,
            history_path,
            os.path.join(output_dir, "05_interactive_analysis.html")
        )

    print(f"\nAll presentation figures saved to: {output_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Presentation Visualization")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_3token_p67",
                       help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="figures_presentation",
                       help="Output directory")

    args = parser.parse_args()

    create_all_presentation_figures(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()

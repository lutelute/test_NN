"""
可視化スクリプト
学習曲線、フーリエスペクトル、埋め込みの可視化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Optional

from model import ModularAdditionTransformer
from analyze import FourierAnalyzer


def set_style():
    """プロットスタイルの設定"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    学習曲線をプロット

    Args:
        history: 学習履歴（train_loss, train_acc, test_loss, test_acc）
        save_path: 保存先パス
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.semilogy(epochs, history["train_loss"], label="Train Loss", alpha=0.8)
    ax.semilogy(epochs, history["test_loss"], label="Test Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training and Test Loss")
    ax.legend()

    # Accuracy
    ax = axes[1]
    train_acc = [a * 100 for a in history["train_acc"]]
    test_acc = [a * 100 for a in history["test_acc"]]
    ax.plot(epochs, train_acc, label="Train Accuracy", alpha=0.8)
    ax.plot(epochs, test_acc, label="Test Accuracy", alpha=0.8)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training and Test Accuracy (Grokking)")
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_fourier_spectrum(analyzer: FourierAnalyzer, save_path: Optional[str] = None):
    """
    フーリエスペクトルをプロット

    Args:
        analyzer: FourierAnalyzer インスタンス
        save_path: 保存先パス
    """
    set_style()

    spectrum = analyzer.compute_fourier_spectrum()
    p = analyzer.p

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 全スペクトル
    ax = axes[0]
    freqs = np.arange(p)
    ax.bar(freqs, spectrum, alpha=0.7, width=0.8)
    ax.set_xlabel("Frequency k")
    ax.set_ylabel("Power")
    ax.set_title(f"Fourier Spectrum of Embedding Weights (p={p})")

    # 支配的周波数をハイライト
    dominant = analyzer.find_dominant_frequencies(top_k=5)
    for freq, power in dominant:
        ax.axvline(x=freq, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 前半のみ（対称性のため）
    ax = axes[1]
    half_p = p // 2 + 1
    ax.bar(np.arange(half_p), spectrum[:half_p], alpha=0.7, width=0.8)
    ax.set_xlabel("Frequency k")
    ax.set_ylabel("Power")
    ax.set_title("Fourier Spectrum (First Half)")

    for freq, power in dominant:
        if freq < half_p:
            ax.axvline(x=freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.annotate(f'k={freq}', xy=(freq, power), xytext=(freq+2, power),
                       fontsize=10, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_embedding_circular(analyzer: FourierAnalyzer, save_path: Optional[str] = None):
    """
    埋め込みの2D射影をプロット（円周構造の確認）

    Args:
        analyzer: FourierAnalyzer インスタンス
        save_path: 保存先パス
    """
    set_style()

    circular_result = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular_result["projection_2d"])
    p = analyzer.p

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 散布図（数値で色分け）
    ax = axes[0]
    scatter = ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=np.arange(p),
                        cmap='hsv', s=50, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label='Number (0 to p-1)')
    ax.set_xlabel(f'Dimension {circular_result["top_2_dims"][0]}')
    ax.set_ylabel(f'Dimension {circular_result["top_2_dims"][1]}')
    ax.set_title('Embedding 2D Projection (Color = Number)')
    ax.set_aspect('equal')

    # 数値ラベル付き
    ax = axes[1]
    ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c='blue', s=20, alpha=0.5)

    # いくつかの数値のみラベル表示（見やすさのため）
    step = max(1, p // 20)
    for i in range(0, p, step):
        ax.annotate(str(i), (proj_2d[i, 0], proj_2d[i, 1]),
                   fontsize=8, alpha=0.8)

    ax.set_xlabel(f'Dimension {circular_result["top_2_dims"][0]}')
    ax.set_ylabel(f'Dimension {circular_result["top_2_dims"][1]}')
    ax.set_title(f'Embedding 2D Projection with Labels\n'
                f'Circularity: {circular_result["circularity"]:.3f}, '
                f'Angle Corr: {circular_result["angle_correlation"]:.3f}')
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_theoretical_comparison(analyzer: FourierAnalyzer, k: int, save_path: Optional[str] = None):
    """
    埋め込みと理論的フーリエ基底の比較

    Args:
        analyzer: FourierAnalyzer インスタンス
        k: 比較する周波数
        save_path: 保存先パス
    """
    set_style()

    weights = analyzer.get_embedding_weights()
    theory = analyzer.compute_theoretical_fourier_basis(k)
    p = analyzer.p

    # 最も相関の高い次元を見つける
    corr_result = analyzer.compute_correlation_with_theory(k)
    correlations = corr_result["correlation_per_dim"]
    best_dim = np.argmax(correlations)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n = np.arange(p)

    # 理論的なcos基底
    ax = axes[0, 0]
    ax.plot(n, theory[:, 0], 'b-', linewidth=2, label=f'cos(2π·{k}·n/{p})')
    ax.set_xlabel('n')
    ax.set_ylabel('Value')
    ax.set_title(f'Theoretical Cosine Basis (k={k})')
    ax.legend()

    # 理論的なsin基底
    ax = axes[0, 1]
    ax.plot(n, theory[:, 1], 'r-', linewidth=2, label=f'sin(2π·{k}·n/{p})')
    ax.set_xlabel('n')
    ax.set_ylabel('Value')
    ax.set_title(f'Theoretical Sine Basis (k={k})')
    ax.legend()

    # 最も相関の高い埋め込み次元
    ax = axes[1, 0]
    embed_dim = weights[:, best_dim]
    embed_normalized = (embed_dim - embed_dim.mean()) / embed_dim.std()
    ax.plot(n, embed_normalized, 'g-', linewidth=2,
            label=f'Embedding dim {best_dim} (normalized)')
    ax.set_xlabel('n')
    ax.set_ylabel('Value (normalized)')
    ax.set_title(f'Best Matching Embedding Dimension\n'
                f'(Corr with k={k}: {correlations[best_dim]:.4f})')
    ax.legend()

    # 比較プロット
    ax = axes[1, 1]
    cos_norm = (theory[:, 0] - theory[:, 0].mean()) / theory[:, 0].std()
    sin_norm = (theory[:, 1] - theory[:, 1].mean()) / theory[:, 1].std()

    ax.plot(n, embed_normalized, 'g-', linewidth=2, alpha=0.8, label='Learned Embedding')
    ax.plot(n, cos_norm, 'b--', linewidth=1.5, alpha=0.6, label='Theoretical cos')
    ax.plot(n, sin_norm, 'r--', linewidth=1.5, alpha=0.6, label='Theoretical sin')
    ax.set_xlabel('n')
    ax.set_ylabel('Value (normalized)')
    ax.set_title('Comparison: Learned vs Theoretical')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_fourier_evolution(fourier_history: Dict, save_path: Optional[str] = None):
    """
    フーリエ相関の時間発展をプロット

    Args:
        fourier_history: フーリエ解析履歴
        save_path: 保存先パス
    """
    set_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = fourier_history["epochs"]

    # 1. Best Correlation の時間発展
    ax = axes[0, 0]
    ax.plot(epochs, fourier_history["best_correlations"], 'b-', linewidth=2)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Threshold (0.9)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best Correlation")
    ax.set_title("Fourier Correlation Evolution")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # 2. Circularity の時間発展
    ax = axes[0, 1]
    ax.plot(epochs, fourier_history["circularities"], 'g-', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Circularity")
    ax.set_title("Embedding Circularity Evolution")

    # 3. Angle Correlation の時間発展
    ax = axes[1, 0]
    ax.plot(epochs, fourier_history["angle_correlations"], 'm-', linewidth=2)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Threshold (0.9)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Angle Correlation")
    ax.set_title("Circular Angle Correlation Evolution")
    ax.legend()

    # 4. Spectrum Concentration の時間発展
    ax = axes[1, 1]
    ax.plot(epochs, fourier_history["spectrum_concentrations"], 'c-', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spectrum Concentration")
    ax.set_title("Fourier Spectrum Concentration Evolution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_grokking_analysis(history: Dict, fourier_history: Dict = None, save_path: Optional[str] = None):
    """
    Grokking現象の詳細分析図

    Args:
        history: 学習履歴
        fourier_history: フーリエ解析履歴（オプション）
        save_path: 保存先パス
    """
    set_style()

    has_fourier = fourier_history is not None and len(fourier_history.get("epochs", [])) > 0

    if has_fourier:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = np.array([axes])

    epochs = range(1, len(history["train_loss"]) + 1)
    train_acc = [a * 100 for a in history["train_acc"]]
    test_acc = [a * 100 for a in history["test_acc"]]

    # 1. Accuracy のギャップ分析
    ax = axes.flat[0]
    ax.fill_between(epochs, train_acc, test_acc, alpha=0.3, color='orange', label='Gap')
    ax.plot(epochs, train_acc, 'b-', linewidth=1.5, label='Train Acc', alpha=0.8)
    ax.plot(epochs, test_acc, 'r-', linewidth=1.5, label='Test Acc', alpha=0.8)

    # Grokking ポイントを検出
    grokking_epoch = None
    for i, (tr, te) in enumerate(zip(history["train_acc"], history["test_acc"])):
        if tr > 0.99 and te > 0.9:
            grokking_epoch = i + 1
            break

    if grokking_epoch:
        ax.axvline(x=grokking_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax.annotate(f'Grokking\n(epoch {grokking_epoch})', xy=(grokking_epoch, 50),
                   fontsize=10, color='green', ha='center')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Grokking Analysis: Train-Test Gap")
    ax.legend()
    ax.set_ylim(0, 105)

    # 2. Loss のログスケール分析
    ax = axes.flat[1]
    ax.semilogy(epochs, history["train_loss"], 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax.semilogy(epochs, history["test_loss"], 'r-', linewidth=1.5, label='Test Loss', alpha=0.8)

    if grokking_epoch:
        ax.axvline(x=grokking_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Loss Evolution")
    ax.legend()

    if has_fourier:
        fourier_epochs = fourier_history["epochs"]

        # 3. フーリエ相関 vs Test Accuracy
        ax = axes.flat[2]
        ax.plot(fourier_epochs, fourier_history["best_correlations"], 'g-', linewidth=2, label='Fourier Corr')

        # 対応するtest_accを取得
        test_acc_at_fourier = [test_acc[min(e-1, len(test_acc)-1)] for e in fourier_epochs]
        ax2 = ax.twinx()
        ax2.plot(fourier_epochs, test_acc_at_fourier, 'r--', linewidth=1.5, alpha=0.7, label='Test Acc')
        ax2.set_ylabel("Test Accuracy (%)", color='red')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fourier Correlation", color='green')
        ax.set_title("Fourier Representation vs Generalization")
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)

        # 4. 支配的周波数の変化
        ax = axes.flat[3]
        if fourier_history.get("dominant_frequencies"):
            # 各エポックでの最も強い周波数を抽出
            top_freqs = []
            for freqs in fourier_history["dominant_frequencies"]:
                if freqs:
                    top_freqs.append(freqs[0][0])  # 最も強い周波数
                else:
                    top_freqs.append(0)

            ax.scatter(fourier_epochs, top_freqs, c=fourier_history["best_correlations"],
                      cmap='viridis', s=30, alpha=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dominant Frequency k")
            ax.set_title("Dominant Frequency Evolution\n(Color = Fourier Correlation)")
            plt.colorbar(ax.collections[0], ax=ax, label='Correlation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_layer_analysis(analyzer: FourierAnalyzer, save_path: Optional[str] = None):
    """
    各層のフーリエ解析結果をプロット

    Args:
        analyzer: FourierAnalyzer インスタンス
        save_path: 保存先パス
    """
    set_style()

    layer_results = analyzer.analyze_all_layers()
    n_layers = len(layer_results)

    if n_layers == 0:
        print("No layers to analyze")
        return

    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 10))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for i, (layer_name, result) in enumerate(layer_results.items()):
        # 上段: フーリエスペクトル
        ax = axes[0, i]
        spectrum = result["spectrum"]
        half_p = len(spectrum) // 2 + 1
        ax.bar(np.arange(half_p), spectrum[:half_p], alpha=0.7)
        ax.set_xlabel("Frequency k")
        ax.set_ylabel("Power")
        ax.set_title(f"{layer_name}\nBest Corr: {result['best_correlation']:.3f}")

        # 支配的周波数をマーク
        for freq, power in result["dominant_frequencies"][:3]:
            if freq < half_p:
                ax.axvline(x=freq, color='red', linestyle='--', alpha=0.5)

        # 下段: 2D射影
        ax = axes[1, i]
        proj = np.array(result["circular"]["projection_2d"])
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)),
                           cmap='hsv', s=20, alpha=0.8)
        ax.set_xlabel(f"Dim {result['circular']['top_2_dims'][0]}")
        ax.set_ylabel(f"Dim {result['circular']['top_2_dims'][1]}")
        ax.set_title(f"Circularity: {result['circular']['circularity']:.3f}")
        ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_attention_weights(checkpoint_path: str, p: int = 113, save_path: Optional[str] = None):
    """
    アテンション重みをヒートマップで可視化

    Args:
        checkpoint_path: チェックポイントのパス
        p: 素数
        save_path: 保存先パス
    """
    set_style()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {"p": p, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})

    n_tokens = config.get("n_tokens", 3)

    model = ModularAdditionTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_tokens=n_tokens,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # サンプル入力で attention weights を取得
    sample_inputs = torch.zeros(config["p"], n_tokens, dtype=torch.long)
    for i in range(n_tokens):
        sample_inputs[:, i] = torch.arange(config["p"])

    with torch.no_grad():
        _, intermediates = model.forward_with_intermediates(sample_inputs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Attention weights (最初のサンプルのみ表示)
    attn_weights = intermediates.get("block_0_attn_weights")
    if attn_weights is not None:
        # 平均 attention pattern
        avg_attn = attn_weights.mean(dim=0).numpy()

        ax = axes[0]
        im = ax.imshow(avg_attn, cmap='Blues', aspect='auto')
        ax.set_title('Average Attention Pattern')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        if n_tokens == 3:
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['a', 'b', 'c'])
            ax.set_yticklabels(['a', 'b', 'c'])
        else:
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['a', 'b'])
            ax.set_yticklabels(['a', 'b'])
        plt.colorbar(im, ax=ax)

    # 埋め込み重みのヒートマップ
    embed_weights = model.get_embedding_weights()

    ax = axes[1]
    # 最初の32次元のみ表示
    im = ax.imshow(embed_weights[:, :32].T, cmap='RdBu', aspect='auto')
    ax.set_title('Embedding Weights (first 32 dims)')
    ax.set_xlabel('Token (0 to p-1)')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)

    # 埋め込みの相関行列
    ax = axes[2]
    corr_matrix = np.corrcoef(embed_weights)
    im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_title('Embedding Correlation Matrix')
    ax.set_xlabel('Token')
    ax.set_ylabel('Token')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_embedding_3d(analyzer: FourierAnalyzer, save_path: Optional[str] = None):
    """
    埋め込みの3D可視化

    Args:
        analyzer: FourierAnalyzer インスタンス
        save_path: 保存先パス
    """
    from mpl_toolkits.mplot3d import Axes3D

    set_style()

    weights = analyzer.get_embedding_weights()
    p = analyzer.p

    # 最も分散が大きい3次元を選択
    variances = weights.var(axis=0)
    top_3_dims = np.argsort(variances)[-3:]

    proj_3d = weights[:, top_3_dims]

    fig = plt.figure(figsize=(12, 5))

    # 3D散布図
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.hsv(np.linspace(0, 1, p))
    ax1.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], c=colors, s=30, alpha=0.8)

    # 順序を線で結ぶ
    for i in range(p - 1):
        ax1.plot([proj_3d[i, 0], proj_3d[i+1, 0]],
                [proj_3d[i, 1], proj_3d[i+1, 1]],
                [proj_3d[i, 2], proj_3d[i+1, 2]], 'gray', alpha=0.3, linewidth=0.5)
    # 最後と最初を結ぶ
    ax1.plot([proj_3d[-1, 0], proj_3d[0, 0]],
            [proj_3d[-1, 1], proj_3d[0, 1]],
            [proj_3d[-1, 2], proj_3d[0, 2]], 'gray', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel(f'Dim {top_3_dims[0]}')
    ax1.set_ylabel(f'Dim {top_3_dims[1]}')
    ax1.set_zlabel(f'Dim {top_3_dims[2]}')
    ax1.set_title('Embedding 3D Projection')

    # 別角度
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], c=colors, s=30, alpha=0.8)
    ax2.view_init(elev=30, azim=120)
    ax2.set_xlabel(f'Dim {top_3_dims[0]}')
    ax2.set_ylabel(f'Dim {top_3_dims[1]}')
    ax2.set_zlabel(f'Dim {top_3_dims[2]}')
    ax2.set_title('Embedding 3D (Different Angle)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_checkpoint_comparison(checkpoint_paths: List[str], labels: List[str] = None,
                               save_path: Optional[str] = None):
    """
    複数チェックポイントの比較

    Args:
        checkpoint_paths: チェックポイントファイルのリスト
        labels: 各チェックポイントのラベル
        save_path: 保存先パス
    """
    set_style()

    if labels is None:
        labels = [f"Checkpoint {i+1}" for i in range(len(checkpoint_paths))]

    n_checkpoints = len(checkpoint_paths)
    fig, axes = plt.subplots(2, n_checkpoints, figsize=(5*n_checkpoints, 10))

    for i, (cp_path, label) in enumerate(zip(checkpoint_paths, labels)):
        checkpoint = torch.load(cp_path, map_location="cpu")
        config = checkpoint.get("config", {"p": 113, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})
        epoch = checkpoint.get("epoch", "?")

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

        # 上段: 2D射影
        ax = axes[0, i] if n_checkpoints > 1 else axes[0]
        circular_result = analyzer.analyze_circular_structure()
        proj_2d = np.array(circular_result["projection_2d"])
        colors = plt.cm.hsv(np.linspace(0, 1, config["p"]))
        ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=30, alpha=0.8)
        ax.set_aspect('equal')
        ax.set_title(f'{label} (Epoch {epoch})\nAngle Corr: {circular_result["angle_correlation"]:.3f}')

        # 下段: フーリエスペクトル
        ax = axes[1, i] if n_checkpoints > 1 else axes[1]
        spectrum = analyzer.compute_fourier_spectrum()
        half_p = config["p"] // 2 + 1
        ax.bar(np.arange(half_p), spectrum[:half_p], alpha=0.7)
        ax.set_xlabel("Frequency k")
        ax.set_ylabel("Power")

        fourier_result = analyzer.verify_fourier_representation()
        ax.set_title(f'Fourier Corr: {fourier_result["best_correlation"]:.3f}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def create_embedding_animation(checkpoint_dir: str, output_path: str = "embedding_evolution.gif",
                              fps: int = 5):
    """
    学習過程の埋め込み変化をGIFアニメーションで作成

    Args:
        checkpoint_dir: チェックポイントディレクトリ
        output_path: 出力ファイルパス
        fps: フレームレート
    """
    import glob
    from matplotlib.animation import FuncAnimation, PillowWriter

    # チェックポイントを読み込み
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))

    if len(checkpoint_files) == 0:
        print("No checkpoint files found!")
        return

    print(f"Found {len(checkpoint_files)} checkpoints")

    # 設定を最初のチェックポイントから取得
    first_cp = torch.load(checkpoint_files[0], map_location="cpu")
    config = first_cp.get("config", {"p": 113, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})
    p = config["p"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.hsv(np.linspace(0, 1, p))

    def update(frame_idx):
        cp_path = checkpoint_files[frame_idx]
        checkpoint = torch.load(cp_path, map_location="cpu")
        epoch = checkpoint.get("epoch", frame_idx)

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

        # 2D射影
        axes[0].clear()
        circular_result = analyzer.analyze_circular_structure()
        proj_2d = np.array(circular_result["projection_2d"])
        axes[0].scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=30, alpha=0.8)
        axes[0].set_aspect('equal')
        axes[0].set_title(f'Epoch {epoch} - Embedding 2D\nAngle Corr: {circular_result["angle_correlation"]:.3f}')

        # フーリエスペクトル
        axes[1].clear()
        spectrum = analyzer.compute_fourier_spectrum()
        half_p = p // 2 + 1
        axes[1].bar(np.arange(half_p), spectrum[:half_p], alpha=0.7, color='blue')
        axes[1].set_xlabel("Frequency k")
        axes[1].set_ylabel("Power")

        fourier_result = analyzer.verify_fourier_representation()
        axes[1].set_title(f'Fourier Spectrum\nBest Corr: {fourier_result["best_correlation"]:.3f}')

        return axes

    anim = FuncAnimation(fig, update, frames=len(checkpoint_files), interval=1000//fps)

    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    print(f"Saved: {output_path}")

    plt.close()


def plot_weight_histograms(checkpoint_path: str, save_path: Optional[str] = None):
    """
    モデル重みのヒストグラム

    Args:
        checkpoint_path: チェックポイントのパス
        save_path: 保存先パス
    """
    set_style()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # 重みを収集
    weight_names = []
    weight_values = []

    for name, param in state_dict.items():
        if 'weight' in name and param.dim() >= 1:
            weight_names.append(name.replace('.weight', '').replace('transformer_blocks.0.', 'block.'))
            weight_values.append(param.flatten().numpy())

    n_weights = len(weight_names)
    n_cols = 3
    n_rows = (n_weights + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, (name, values) in enumerate(zip(weight_names, weight_values)):
        if i < len(axes):
            ax = axes[i]
            ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\nmean={values.mean():.4f}, std={values.std():.4f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')

    # 余分なaxesを非表示
    for i in range(len(weight_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_fourier_surface(analyzer: FourierAnalyzer, save_path: Optional[str] = None):
    """
    フーリエ表現の3Dサーフェスプロット（xy面.png再現）
    左側: cos/sin波形群
    右側: 3Dサーフェスプロット

    Args:
        analyzer: FourierAnalyzer インスタンス
        save_path: 保存先パス
    """
    from mpl_toolkits.mplot3d import Axes3D

    plt.style.use('dark_background')

    p = analyzer.p
    weights = analyzer.get_embedding_weights()

    fig = plt.figure(figsize=(16, 8), facecolor='black')

    # 左側: 波形グリッド（2列 x 5行）
    left_gs = fig.add_gridspec(5, 2, left=0.03, right=0.35, top=0.95, bottom=0.05, wspace=0.1, hspace=0.3)

    n = np.arange(p)
    dominant = analyzer.find_dominant_frequencies(top_k=5)
    wave_colors = ['#9966FF', '#3366FF', '#00CCCC', '#66CC66', '#FFCC00']

    for i, ((freq, power), color) in enumerate(zip(dominant, wave_colors)):
        if i >= 5:
            break
        # cos波形
        ax_cos = fig.add_subplot(left_gs[i, 0], facecolor='black')
        angles = 2 * np.pi * freq * n / p
        cos_wave = np.cos(angles)
        ax_cos.plot(n, cos_wave, color=color, linewidth=1)
        ax_cos.set_xlim(0, p)
        ax_cos.set_ylim(-1.5, 1.5)
        ax_cos.tick_params(colors='white', labelsize=6)
        if i == 0:
            ax_cos.set_title(f'cos(k{freq})', color='white', fontsize=9)
        ax_cos.text(p*0.95, 1.0, f'k={freq}', color=color, fontsize=8, ha='right')
        for spine in ax_cos.spines.values():
            spine.set_color('gray')

        # sin波形
        ax_sin = fig.add_subplot(left_gs[i, 1], facecolor='black')
        sin_wave = np.sin(angles)
        ax_sin.plot(n, sin_wave, color=color, linewidth=1)
        ax_sin.set_xlim(0, p)
        ax_sin.set_ylim(-1.5, 1.5)
        ax_sin.tick_params(colors='white', labelsize=6)
        if i == 0:
            ax_sin.set_title(f'sin(k{freq})', color='white', fontsize=9)
        for spine in ax_sin.spines.values():
            spine.set_color('gray')

    # 中央: モデル概要（テキスト）
    ax_mid = fig.add_axes([0.37, 0.3, 0.15, 0.4], facecolor='black')
    ax_mid.axis('off')
    model_text = f"p = {p}\n114×3 → 128×3\n↓\nATTENTION\n↓\n114×1"
    ax_mid.text(0.5, 0.5, model_text, color='white', fontsize=12, ha='center', va='center',
               family='monospace', linespacing=1.5,
               bbox=dict(boxstyle='round', facecolor='#222222', edgecolor='gold', linewidth=2))

    # 右側: 3Dサーフェスプロット
    ax3d = fig.add_subplot(1, 2, 2, projection='3d', facecolor='black')

    # 複数の周波数の波を重ねてサーフェスを作成
    x = np.linspace(0, 2*np.pi, 50)
    y = np.linspace(0, 2*np.pi, 50)
    X, Y = np.meshgrid(x, y)

    # 各層の波面
    surface_colors = ['#00FFFF', '#FFFF00', '#FF00FF', '#00FF00']
    for layer_idx, (freq, power) in enumerate(dominant[:4]):
        Z = np.sin(freq * X + layer_idx * np.pi/4) * np.cos(freq * Y + layer_idx * np.pi/4)
        Z = Z + layer_idx * 2  # オフセット

        ax3d.plot_surface(X, Y, Z, alpha=0.7, cmap='plasma', edgecolor='none')

    ax3d.set_xlabel('X', color='white')
    ax3d.set_ylabel('Y', color='white')
    ax3d.set_zlabel('Z', color='white')
    ax3d.tick_params(colors='white')
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.set_title('Fourier Surface Representation', color='white', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_correlation_visualization(analyzer: FourierAnalyzer, save_path: Optional[str] = None):
    """
    相関可視化（正方グリッド版）
    各層の2D射影を正方グリッドで表示

    Args:
        analyzer: FourierAnalyzer インスタンス
        save_path: 保存先パス
    """
    plt.style.use('dark_background')

    p = analyzer.p
    layer_outputs = analyzer.get_layer_outputs()

    # 全ての層を取得
    all_layers = list(layer_outputs.keys())

    # グリッドサイズを決定（正方形に近くなるように）
    n_layers = len(all_layers)
    grid_size = int(np.ceil(np.sqrt(n_layers)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14), facecolor='black')
    fig.suptitle('Layer-wise 2D Projections (Circular Structure)', color='white', fontsize=14, y=0.98)

    colors = plt.cm.hsv(np.linspace(0, 1, p))

    for i, layer_name in enumerate(all_layers):
        row = i // grid_size
        col = i % grid_size
        ax = axes[row, col] if grid_size > 1 else axes

        ax.set_facecolor('black')

        weights = layer_outputs.get(layer_name)
        if weights is None:
            ax.axis('off')
            continue

        # 2D射影を計算
        result = analyzer._analyze_circular_for_weights(weights)
        proj_2d = np.array(result["projection_2d"])

        # 散布図をプロット
        ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=15, alpha=0.9)
        ax.set_aspect('equal')

        # ラベル（短縮名）
        short_name = layer_name.replace('block_0_', '').replace('_output', '').replace('_weights', '')
        short_name = short_name[:15]  # 最大15文字

        corr = result["angle_correlation"]
        ax.set_title(f'{short_name}\nr={corr:.2f}', color='white', fontsize=9, pad=3)

        # 軸の目盛りを非表示
        ax.set_xticks([])
        ax.set_yticks([])

        # 相関が高い場合は緑の枠、低い場合は灰色
        if corr > 0.9:
            border_color = '#00FF00'  # 緑
            border_width = 3
        elif corr > 0.7:
            border_color = '#FFFF00'  # 黄
            border_width = 2
        else:
            border_color = '#555555'  # 灰
            border_width = 1

        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)

    # 余分なサブプロットを非表示
    for i in range(len(all_layers), grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_model_architecture(model: ModularAdditionTransformer = None, p: int = 113,
                           save_path: Optional[str] = None):
    """
    モデルアーキテクチャ図（config.png再現）

    Args:
        model: ModularAdditionTransformerインスタンス（オプション）
        p: 素数
        save_path: 保存先パス
    """
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(16, 5), facecolor='black')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_facecolor('black')

    # 色設定
    box_color = '#1a1a1a'
    border_color = '#c9a227'  # ゴールド
    text_color = 'white'
    circle_color = '#c9a227'

    def draw_number_column(ax, x, y, title, n_show=8, p=113):
        """数字列を描画"""
        # ボックス
        rect = plt.Rectangle((x-0.3, y-1.3), 0.6, 2.6, fill=True,
                             facecolor=box_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)

        # 数字（上から）
        numbers = [0, 1, 2, 3, 4, 5, 6, 7, '...', p-1]
        y_positions = np.linspace(y+1.1, y-1.1, len(numbers))

        for num, ypos in zip(numbers, y_positions):
            if num == '...':
                ax.text(x, ypos, num, color=text_color, fontsize=9, ha='center', va='center')
            else:
                circle = plt.Circle((x, ypos), 0.08, fill=True, facecolor=circle_color, edgecolor='none')
                ax.add_patch(circle)
                ax.text(x + 0.2, ypos, str(num), color=text_color, fontsize=8, ha='left', va='center')

        ax.text(x, y+1.4, title, color=text_color, fontsize=10, ha='center', va='bottom', fontweight='bold')

    def draw_embed_block(ax, x, y):
        """EMBED ブロックを描画"""
        rect = plt.Rectangle((x-0.5, y-1.0), 1.0, 2.0, fill=True,
                             facecolor=box_color, edgecolor=border_color, linewidth=2, linestyle='--')
        ax.add_patch(rect)

        # 内部の点グリッド
        for i in range(5):
            for j in range(3):
                circle = plt.Circle((x-0.3+j*0.3, y+0.7-i*0.35), 0.05, fill=True,
                                   facecolor=circle_color, edgecolor='none', alpha=0.7)
                ax.add_patch(circle)

        ax.text(x, y-1.2, 'EMBED', color=text_color, fontsize=10, ha='center', va='top', fontweight='bold')
        ax.text(x, y+1.15, '128×3', color='gray', fontsize=8, ha='center', va='bottom')

    def draw_attention_block(ax, x, y):
        """ATTENTION ブロックを描画"""
        rect = plt.Rectangle((x-0.6, y-1.0), 1.2, 2.0, fill=True,
                             facecolor=box_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)

        # アテンションパターン（グリッド）
        for i in range(3):
            for j in range(3):
                intensity = 0.3 + 0.4 * (1 - abs(i-j)/2)
                rect_inner = plt.Rectangle((x-0.4+j*0.25, y+0.5-i*0.4), 0.2, 0.35,
                                          fill=True, facecolor=circle_color, alpha=intensity)
                ax.add_patch(rect_inner)

        ax.text(x, y-1.2, 'ATTENTION', color=text_color, fontsize=10, ha='center', va='top', fontweight='bold')

    def draw_mlp_block(ax, x, y):
        """MLP ブロックを描画"""
        rect = plt.Rectangle((x-0.6, y-1.0), 1.2, 2.0, fill=True,
                             facecolor=box_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)

        # ニューラルネットワーク風の線
        left_nodes = [(x-0.35, y+0.6-i*0.4) for i in range(4)]
        right_nodes = [(x+0.35, y+0.6-i*0.4) for i in range(4)]

        for ln in left_nodes:
            for rn in right_nodes:
                ax.plot([ln[0], rn[0]], [ln[1], rn[1]], color=circle_color, alpha=0.3, linewidth=0.5)

        for ln in left_nodes:
            circle = plt.Circle(ln, 0.06, fill=True, facecolor=circle_color, edgecolor='none')
            ax.add_patch(circle)
        for rn in right_nodes:
            circle = plt.Circle(rn, 0.06, fill=True, facecolor=circle_color, edgecolor='none')
            ax.add_patch(circle)

        ax.text(x, y-1.2, 'MULTILAYER\nPERCEPTRON', color=text_color, fontsize=9,
               ha='center', va='top', fontweight='bold')

    # 描画
    # 入力列
    draw_number_column(ax, 0.8, 1.5, f'114×3', p=p)

    # 矢印
    ax.annotate('', xy=(1.8, 1.5), xytext=(1.2, 1.5),
               arrowprops=dict(arrowstyle='->', color=border_color, lw=2))

    # EMBED
    draw_embed_block(ax, 2.5, 1.5)

    # 矢印
    ax.annotate('', xy=(3.8, 1.5), xytext=(3.1, 1.5),
               arrowprops=dict(arrowstyle='->', color=border_color, lw=2))

    # ATTENTION
    draw_attention_block(ax, 4.5, 1.5)

    # 矢印
    ax.annotate('', xy=(5.8, 1.5), xytext=(5.2, 1.5),
               arrowprops=dict(arrowstyle='->', color=border_color, lw=2))

    # MLP
    draw_mlp_block(ax, 6.5, 1.5)

    # 矢印 + UNEMBED
    ax.annotate('', xy=(8.0, 1.5), xytext=(7.2, 1.5),
               arrowprops=dict(arrowstyle='->', color=border_color, lw=2))
    ax.text(7.6, 1.8, 'UNEMBED', color=text_color, fontsize=9, ha='center', fontweight='bold')

    # 出力列
    draw_number_column(ax, 8.8, 1.5, f'114×1', p=p)

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_all(checkpoint_path: str, history_path: str = None, output_dir: str = "figures"):
    """
    全ての可視化を実行

    Args:
        checkpoint_path: チェックポイントファイルのパス
        history_path: 学習履歴JSONのパス
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {"p": 113, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 3})

    # モデル作成・重み読み込み
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

    # 1. 学習曲線
    history = checkpoint.get("history", None)
    if history is None and history_path and os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    if history:
        plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))

    # 2. フーリエスペクトル
    plot_fourier_spectrum(analyzer, os.path.join(output_dir, "fourier_spectrum.png"))

    # 3. 埋め込みの円周構造
    plot_embedding_circular(analyzer, os.path.join(output_dir, "embedding_circular.png"))

    # 4. 理論との比較（支配的周波数）
    dominant_freqs = analyzer.find_dominant_frequencies(top_k=3)
    if dominant_freqs:
        k = dominant_freqs[0][0]
        plot_theoretical_comparison(analyzer, k, os.path.join(output_dir, "theoretical_comparison.png"))

    # 5. フーリエ相関の時間発展
    fourier_history_path = os.path.join(os.path.dirname(checkpoint_path), "fourier_history.json")
    fourier_history = None
    if os.path.exists(fourier_history_path):
        try:
            with open(fourier_history_path, "r") as f:
                fourier_history = json.load(f)
            plot_fourier_evolution(fourier_history, os.path.join(output_dir, "fourier_evolution.png"))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not load fourier_history.json: {e}")

    # 6. Grokking分析
    if history:
        plot_grokking_analysis(history, fourier_history, os.path.join(output_dir, "grokking_analysis.png"))

    # 7. 層ごとの解析
    try:
        plot_layer_analysis(analyzer, os.path.join(output_dir, "layer_analysis.png"))
    except Exception as e:
        print(f"Layer analysis skipped: {e}")

    # 8. アテンション重みとヒートマップ
    try:
        plot_attention_weights(checkpoint_path, config["p"], os.path.join(output_dir, "attention_weights.png"))
    except Exception as e:
        print(f"Attention weights skipped: {e}")

    # 9. 3D可視化
    try:
        plot_embedding_3d(analyzer, os.path.join(output_dir, "embedding_3d.png"))
    except Exception as e:
        print(f"3D visualization skipped: {e}")

    # 10. 重みヒストグラム
    try:
        plot_weight_histograms(checkpoint_path, os.path.join(output_dir, "weight_histograms.png"))
    except Exception as e:
        print(f"Weight histograms skipped: {e}")

    # 11. フーリエサーフェス可視化（xy面.png再現）
    try:
        plot_fourier_surface(analyzer, os.path.join(output_dir, "fourier_surface.png"))
    except Exception as e:
        print(f"Fourier surface skipped: {e}")

    # 12. 相関可視化（相関.png再現）
    try:
        plot_correlation_visualization(analyzer, os.path.join(output_dir, "correlation_viz.png"))
    except Exception as e:
        print(f"Correlation visualization skipped: {e}")

    # 13. モデルアーキテクチャ図（config.png再現）
    try:
        plot_model_architecture(model, config["p"], os.path.join(output_dir, "model_architecture.png"))
    except Exception as e:
        print(f"Model architecture skipped: {e}")

    print(f"\nAll figures saved to: {output_dir}/")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Checkpoint path")
    parser.add_argument("--history", type=str, default="checkpoints/history.json", help="History JSON path")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    plot_all(args.checkpoint, args.history, args.output_dir)


if __name__ == "__main__":
    main()

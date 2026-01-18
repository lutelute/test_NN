#!/usr/bin/env python3
"""
Grokking インタラクティブ解析ツール
- DFT表示（支配的周波数をハイライト）
- MLP出力の対角表示
- 学習過程のスライダー
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec
import glob
import os

from model import ModularAdditionTransformer
from analyze import FourierAnalyzer


def load_model(checkpoint_path):
    """モデルをロード"""
    cp = torch.load(checkpoint_path, map_location='cpu')
    config = cp['config']
    model = ModularAdditionTransformer(
        p=config['p'], d_model=config['d_model'],
        n_heads=config['n_heads'], n_layers=config['n_layers'],
        n_tokens=config.get('n_tokens', 2)
    )
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    return model, config, cp.get('epoch', 0)


def compute_mlp_output_matrix(model, config):
    """
    MLP出力を行列として計算
    2-token: output[a][b] = model prediction for (a, b)
    3-token: output[a+b][c] = model prediction for (a, b, c) with fixed a+b
    """
    p = config['p']
    n_tokens = config.get('n_tokens', 2)
    
    if n_tokens == 2:
        # 2-token: p x p matrix
        matrix = np.zeros((p, p))
        with torch.no_grad():
            for a in range(p):
                for b in range(p):
                    x = torch.tensor([[a, b]])
                    pred = model(x).argmax(dim=-1).item()
                    matrix[a, b] = pred
        return matrix, "a", "b", "(a+b) mod p"
    else:
        # 3-token: 集約して表示（a+bを固定、cを変化）
        matrix = np.zeros((p, p))
        with torch.no_grad():
            for ab_sum in range(p):
                for c in range(p):
                    # a+b = ab_sum となる (a,b) を1つ選ぶ
                    a = ab_sum % p
                    b = 0
                    x = torch.tensor([[a, b, c]])
                    pred = model(x).argmax(dim=-1).item()
                    matrix[ab_sum, c] = pred
        return matrix, "a+b", "c", "(a+b+c) mod p"


def compute_dft_spectrum(model, p):
    """埋め込みのDFTスペクトラムを計算"""
    weights = model.token_embedding.weight.detach().numpy()  # (p, d_model)
    
    # 各次元でDFT
    spectra = []
    for dim in range(weights.shape[1]):
        fft = np.fft.fft(weights[:, dim])
        power = np.abs(fft) ** 2
        spectra.append(power)
    
    # 平均スペクトラム
    avg_spectrum = np.mean(spectra, axis=0)
    return avg_spectrum[:p//2+1]  # 半分だけ（対称性）


def run_interactive_analysis(checkpoint_dir, title_prefix=""):
    """インタラクティブ解析ツールを起動"""
    
    # チェックポイントを検索
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    if not checkpoint_files:
        checkpoint_files = [os.path.join(checkpoint_dir, "best.pt")]
    
    if not os.path.exists(checkpoint_files[0]):
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    # 初期モデルをロード
    model, config, epoch = load_model(checkpoint_files[-1])  # 最新
    p = config['p']
    n_tokens = config.get('n_tokens', 2)
    
    # Figure設定
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    fig.suptitle(f'{title_prefix}p={p} {n_tokens}-token Grokking Analysis', 
                 color='gold', fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. DFTスペクトラム
    ax_dft = fig.add_subplot(gs[0, 0])
    spectrum = compute_dft_spectrum(model, p)
    freqs = np.arange(len(spectrum))
    bars = ax_dft.bar(freqs, spectrum, color='cyan', alpha=0.7, width=0.8)
    
    # 支配的周波数をハイライト
    dominant_k = np.argmax(spectrum[1:]) + 1  # 0を除く
    bars[dominant_k].set_color('yellow')
    bars[dominant_k].set_alpha(1.0)
    
    ax_dft.axvline(x=dominant_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax_dft.set_xlabel('Frequency k', color='white')
    ax_dft.set_ylabel('Power', color='white')
    ax_dft.set_title(f'DFT Spectrum (dominant k={dominant_k})', color='white')
    ax_dft.tick_params(colors='white')
    
    # 周波数ラベル
    ax_dft.text(dominant_k, spectrum[dominant_k]*1.1, f'k={dominant_k}', 
                color='yellow', fontsize=12, ha='center', fontweight='bold')
    
    # 2. 埋め込みの2D射影（円周構造）
    ax_embed = fig.add_subplot(gs[0, 1])
    analyzer = FourierAnalyzer(model)
    circular = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular['projection_2d'])
    colors = plt.cm.hsv(np.linspace(0, 1, p))
    ax_embed.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=50, alpha=0.9)
    ax_embed.set_aspect('equal')
    ax_embed.set_title(f'Embedding Circle (r={circular["angle_correlation"]:.3f})', color='white')
    ax_embed.tick_params(colors='white')
    for spine in ax_embed.spines.values():
        spine.set_color('gold' if circular["angle_correlation"] > 0.9 else 'gray')
        spine.set_linewidth(2)
    
    # 3. 理論的cos/sin波形との比較
    ax_wave = fig.add_subplot(gs[0, 2])
    n = np.arange(p)
    cos_theory = np.cos(2 * np.pi * dominant_k * n / p)
    sin_theory = np.sin(2 * np.pi * dominant_k * n / p)
    
    # 最も相関の高い埋め込み次元
    weights = model.token_embedding.weight.detach().numpy()
    best_dim = np.argmax(np.var(weights, axis=0))
    embed_wave = weights[:, best_dim]
    embed_norm = (embed_wave - embed_wave.mean()) / (embed_wave.std() + 1e-8)
    
    ax_wave.plot(n, cos_theory, 'b-', linewidth=2, alpha=0.7, label=f'cos(2πk{dominant_k}n/{p})')
    ax_wave.plot(n, sin_theory, 'r-', linewidth=2, alpha=0.7, label=f'sin(2πk{dominant_k}n/{p})')
    ax_wave.plot(n, embed_norm, 'g--', linewidth=2, alpha=0.9, label='Learned embedding')
    ax_wave.set_xlabel('n', color='white')
    ax_wave.set_ylabel('Value', color='white')
    ax_wave.set_title('Fourier Basis Comparison', color='white')
    ax_wave.legend(fontsize=8, loc='upper right')
    ax_wave.tick_params(colors='white')
    
    # 4. MLP出力の対角行列表示
    ax_mlp = fig.add_subplot(gs[1, 0:2])
    matrix, xlabel, ylabel, title = compute_mlp_output_matrix(model, config)
    
    # 対角パターンを期待値と比較
    expected = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            expected[i, j] = (i + j) % p
    
    # 正解なら緑、間違いなら赤
    correct_mask = (matrix == expected)
    accuracy = correct_mask.mean() * 100
    
    # カラーマップで表示（対角パターンが見える）
    im = ax_mlp.imshow(matrix, cmap='hsv', aspect='equal', interpolation='nearest')
    ax_mlp.set_xlabel(f'{xlabel}', color='white', fontsize=12)
    ax_mlp.set_ylabel(f'{ylabel}', color='white', fontsize=12)
    ax_mlp.set_title(f'MLP Output: {title} (Accuracy: {accuracy:.1f}%)', color='white')
    ax_mlp.tick_params(colors='white')
    
    # 対角線を強調
    for i in range(min(p, 20)):  # 最初の20本の対角線
        diag_val = i
        xs = np.arange(p)
        ys = (diag_val - xs) % p
        ax_mlp.plot(xs, ys, 'w-', alpha=0.1, linewidth=0.5)
    
    plt.colorbar(im, ax=ax_mlp, label='Predicted value')
    
    # 5. 精度の説明
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis('off')
    
    info_text = f"""
    Model Configuration
    ─────────────────────
    p = {p}
    n_tokens = {n_tokens}
    d_model = {config['d_model']}
    
    Fourier Analysis
    ─────────────────────
    Dominant freq: k = {dominant_k}
    Circularity: {circular['circularity']:.3f}
    Angle corr: {circular['angle_correlation']:.3f}
    
    MLP Output
    ─────────────────────
    Accuracy: {accuracy:.1f}%
    
    Theory
    ─────────────────────
    cos(a)·cos(b) - sin(a)·sin(b)
      = cos(a+b)
    
    → Model learns Fourier
      representation!
    """
    
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=10, color='white', family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='gold'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_2token_final')
    parser.add_argument('--title', type=str, default='')
    args = parser.parse_args()
    
    run_interactive_analysis(args.checkpoint_dir, args.title)


if __name__ == "__main__":
    main()

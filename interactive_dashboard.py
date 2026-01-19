#!/usr/bin/env python3
"""
Grokking Interactive Dashboard
Streamlit-based interactive analysis tool

Usage:
    streamlit run interactive_dashboard.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import glob
from pathlib import Path

from model import ModularAdditionTransformer
from analyze import FourierAnalyzer


st.set_page_config(
    page_title="Grokking Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(checkpoint_path: str):
    """モデルをキャッシュしてロード"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # configをcheckpoint内または別ファイルから取得
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # config.jsonから読み込み
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # デフォルト値
            config = {"p": 97, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 2}

    model = ModularAdditionTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_tokens=config.get("n_tokens", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint.get("epoch", 0)


@st.cache_data
def load_history(history_path: str):
    """学習履歴をキャッシュしてロード"""
    with open(history_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_fourier_history(fourier_path: str):
    """フーリエ履歴をキャッシュしてロード"""
    with open(fourier_path, "r") as f:
        return json.load(f)


def get_checkpoint_dirs():
    """利用可能なチェックポイントディレクトリを取得"""
    dirs = []
    for d in os.listdir("."):
        if d.startswith("checkpoints") and os.path.isdir(d):
            # 必要なファイルが存在するか確認
            has_checkpoint = (
                os.path.exists(os.path.join(d, "best.pt")) or
                os.path.exists(os.path.join(d, "final.pt")) or
                any(f.startswith("checkpoint_epoch_") and f.endswith(".pt") for f in os.listdir(d))
            )
            if has_checkpoint:
                dirs.append(d)
    # mod番号でソート（checkpoints_mod2, checkpoints_mod3, ...）
    def sort_key(x):
        if "mod" in x:
            try:
                return (0, int(x.split("mod")[1].split("_")[0]))
            except:
                return (1, x)
        return (2, x)
    return sorted(dirs, key=sort_key)


def plot_training_curves(history):
    """学習曲線のインタラクティブプロット"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy", "Loss (log scale)"),
        horizontal_spacing=0.1
    )

    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_acc = [a * 100 for a in history["train_acc"]]
    test_acc = [a * 100 for a in history["test_acc"]]

    # 精度
    fig.add_trace(
        go.Scatter(x=epochs, y=train_acc, name="Train Accuracy",
                  line=dict(color="#2196F3", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=test_acc, name="Test Accuracy",
                  line=dict(color="#F44336", width=2)),
        row=1, col=1
    )

    # Grokkingポイント検出
    grokking_epoch = None
    for i, (tr, te) in enumerate(zip(history["train_acc"], history["test_acc"])):
        if tr > 0.99 and te > 0.9:
            grokking_epoch = i + 1
            break

    if grokking_epoch:
        fig.add_vline(x=grokking_epoch, line_dash="dash", line_color="green",
                     annotation_text=f"Grokking @ {grokking_epoch}", row=1, col=1)

    # ロス
    fig.add_trace(
        go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                  line=dict(color="#2196F3", width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history["test_loss"], name="Test Loss",
                  line=dict(color="#F44336", width=2)),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1, range=[0, 105])
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=2)

    fig.update_layout(height=400, showlegend=True, legend=dict(orientation="h", y=-0.15))

    return fig


def plot_fourier_spectrum(analyzer):
    """フーリエスペクトルのインタラクティブプロット"""
    spectrum = analyzer.compute_fourier_spectrum()
    p = analyzer.p
    half_p = p // 2 + 1
    dominant = analyzer.find_dominant_frequencies(top_k=5)
    dominant_freqs = [f[0] for f in dominant]

    colors = ["#FF5722" if i in dominant_freqs else "#3F51B5" for i in range(half_p)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(half_p)),
        y=spectrum[:half_p].tolist(),
        marker_color=colors,
        text=[f"k={i}" if i in dominant_freqs else "" for i in range(half_p)],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Fourier Spectrum (p={p})",
        xaxis_title="Frequency k",
        yaxis_title="Power",
        height=400
    )

    return fig, dominant


def plot_embedding_circle(analyzer):
    """埋め込みの円周構造をインタラクティブにプロット"""
    circular_result = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular_result["projection_2d"])
    p = analyzer.p

    fig = go.Figure()

    # 点を線で結ぶ
    x_line = proj_2d[:, 0].tolist() + [proj_2d[0, 0]]
    y_line = proj_2d[:, 1].tolist() + [proj_2d[0, 1]]
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="gray", width=0.5),
        showlegend=False
    ))

    # 散布図
    fig.add_trace(go.Scatter(
        x=proj_2d[:, 0].tolist(),
        y=proj_2d[:, 1].tolist(),
        mode="markers",
        marker=dict(
            color=list(range(p)),
            colorscale="HSV",
            size=10,
            colorbar=dict(title="Token")
        ),
        text=[f"Token {i}" for i in range(p)],
        hovertemplate="Token %{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Embedding Circular Structure<br>(Angle Correlation: {circular_result['angle_correlation']:.3f})",
        xaxis_title=f"Dimension {circular_result['top_2_dims'][0]}",
        yaxis_title=f"Dimension {circular_result['top_2_dims'][1]}",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )

    return fig, circular_result


def plot_fourier_basis_comparison(analyzer, dominant_freqs):
    """フーリエ基底との比較"""
    if not dominant_freqs:
        return None

    k = dominant_freqs[0][0]
    p = analyzer.p
    n = np.arange(p)

    cos_theory = np.cos(2 * np.pi * k * n / p)
    sin_theory = np.sin(2 * np.pi * k * n / p)

    weights = analyzer.get_embedding_weights()
    best_dim = np.argmax(np.var(weights, axis=0))
    embed_dim = weights[:, best_dim]
    embed_norm = (embed_dim - embed_dim.mean()) / (embed_dim.std() + 1e-8)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n.tolist(), y=cos_theory.tolist(),
        name=f"cos(2πk{k}n/p)",
        line=dict(color="#2196F3", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=n.tolist(), y=sin_theory.tolist(),
        name=f"sin(2πk{k}n/p)",
        line=dict(color="#F44336", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=n.tolist(), y=embed_norm.tolist(),
        name="Learned Embedding",
        line=dict(color="#4CAF50", width=2, dash="dash")
    ))

    fig.update_layout(
        title=f"Fourier Basis Comparison (k={k})",
        xaxis_title="Token n",
        yaxis_title="Normalized Value",
        height=400,
        legend=dict(orientation="h", y=-0.15)
    )

    return fig


def plot_fourier_evolution(fourier_history):
    """フーリエ相関の時間発展"""
    epochs = fourier_history["epochs"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Fourier Correlation",
            "Angle Correlation",
            "Spectrum Concentration",
            "Circularity"
        )
    )

    fig.add_trace(go.Scatter(
        x=epochs, y=fourier_history["best_correlations"],
        line=dict(color="#9C27B0", width=2),
        fill="tozeroy", name="Fourier Corr"
    ), row=1, col=1)
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=fourier_history["angle_correlations"],
        line=dict(color="#00BCD4", width=2),
        fill="tozeroy", name="Angle Corr"
    ), row=1, col=2)
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=2)

    fig.add_trace(go.Scatter(
        x=epochs, y=fourier_history["spectrum_concentrations"],
        line=dict(color="#FF5722", width=2),
        fill="tozeroy", name="Spectrum Conc"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=fourier_history["circularities"],
        line=dict(color="#4CAF50", width=2),
        fill="tozeroy", name="Circularity"
    ), row=2, col=2)

    fig.update_layout(height=600, showlegend=False)
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)

    return fig


def plot_neuron_correlation_matrix(model, config, grid_size=5, sample_size=500, fixed_dims=None):
    """
    ニューロン出力の相関行列（散布図グリッド）- 軽量版
    相関.pngのような可視化

    Args:
        fixed_dims: 固定の次元リスト（指定すると全エポックで同じ次元を使用）
    """
    p = config["p"]
    n_tokens = config.get("n_tokens", 3)

    # サンプリングして軽量化
    if n_tokens == 2:
        all_inputs = [[a, b] for a in range(p) for b in range(p)]
    else:
        all_inputs = [[a, b, 0] for a in range(p) for b in range(p)]

    # ランダムサンプリング（再現性のためseed固定）
    np.random.seed(42)
    if len(all_inputs) > sample_size:
        indices = np.random.choice(len(all_inputs), sample_size, replace=False)
        sampled_inputs = [all_inputs[i] for i in sorted(indices)]
    else:
        sampled_inputs = all_inputs

    with torch.no_grad():
        inputs = torch.tensor(sampled_inputs)
        _, intermediates = model.forward_with_intermediates(inputs)

    # pooled層の出力を使用（shape: [batch, d_model]）
    pooled = intermediates["pooled"].numpy()  # (batch, d_model)

    # 固定次元が指定されていればそれを使用、なければ分散で選択
    if fixed_dims is not None:
        top_dims = fixed_dims[:grid_size]
    else:
        variances = np.var(pooled, axis=0)
        top_dims = np.argsort(variances)[::-1][:grid_size]

    # サブプロット作成
    fig = make_subplots(
        rows=grid_size, cols=grid_size,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    # HSVカラーマップ
    colors = [f"hsl({int(i * 360 / len(pooled))}, 70%, 50%)" for i in range(len(pooled))]

    for i in range(grid_size):
        for j in range(grid_size):
            dim_i = top_dims[i]
            dim_j = top_dims[j]

            x_data = pooled[:, dim_j]
            y_data = pooled[:, dim_i]

            # 相関係数を計算
            corr = np.corrcoef(x_data, y_data)[0, 1] if i != j else 1.0

            fig.add_trace(
                go.Scatter(
                    x=x_data.tolist(),
                    y=y_data.tolist(),
                    mode="markers",
                    marker=dict(
                        color=list(range(len(pooled))),
                        colorscale="HSV",
                        size=3,
                        opacity=0.6
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=i+1, col=j+1
            )

            # 対角線上には次元番号を表示
            if i == j:
                fig.add_annotation(
                    text=f"d{dim_i}",
                    xref=f"x{i*grid_size+j+1}" if i*grid_size+j > 0 else "x",
                    yref=f"y{i*grid_size+j+1}" if i*grid_size+j > 0 else "y",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    showarrow=False,
                    font=dict(color="yellow", size=10)
                )

    # レイアウト調整
    fig.update_layout(
        height=600,
        width=600,
        title=f"Neuron Correlation Matrix (Top {grid_size} dims by variance)",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        showlegend=False
    )

    # 軸の目盛りを非表示
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    return fig


def get_epoch_path(checkpoint_dir, epoch):
    """エポックファイルのパスを取得（両方の形式に対応）"""
    # 新形式: checkpoint_epoch_XXXXX.pt
    path1 = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:05d}.pt")
    if os.path.exists(path1):
        return path1
    # 旧形式: epoch_XXXXX.pt
    path2 = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    if os.path.exists(path2):
        return path2
    return None


def plot_epoch_progress(checkpoint_dir, selected_epoch, history, config):
    """エポック進捗の可視化（埋め込み空間 + 学習曲線）"""
    p = config["p"]

    # 選択されたエポックのモデルをロード
    epoch_path = get_epoch_path(checkpoint_dir, selected_epoch)
    if epoch_path is None:
        return None, None, None

    model, _, _ = load_model(epoch_path)
    analyzer = FourierAnalyzer(model)

    # 埋め込みの円周構造を取得
    circular_result = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular_result["projection_2d"])

    # 2段構成のサブプロットを作成
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"Embedding Space (Epoch {selected_epoch}, Angle Corr: {circular_result['angle_correlation']:.3f})",
            "Training Progress"
        ),
        vertical_spacing=0.12
    )

    # 上段: 埋め込み空間の散布図
    colors = [f"hsl({int(i * 360 / p)}, 80%, 50%)" for i in range(p)]
    fig.add_trace(
        go.Scatter(
            x=proj_2d[:, 0].tolist(),
            y=proj_2d[:, 1].tolist(),
            mode="markers",
            marker=dict(
                color=list(range(p)),
                colorscale="HSV",
                size=8,
                colorbar=dict(title="Token", x=1.02)
            ),
            text=[f"Token {i}" for i in range(p)],
            hovertemplate="Token %{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            showlegend=False
        ),
        row=1, col=1
    )

    # 点を線で結ぶ
    x_line = proj_2d[:, 0].tolist() + [proj_2d[0, 0]]
    y_line = proj_2d[:, 1].tolist() + [proj_2d[0, 1]]
    fig.add_trace(
        go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="rgba(128,128,128,0.3)", width=1),
            showlegend=False
        ),
        row=1, col=1
    )

    # 下段: 学習曲線
    epochs = list(range(1, len(history["train_acc"]) + 1))
    train_acc = [a * 100 for a in history["train_acc"]]
    test_acc = [a * 100 for a in history["test_acc"]]

    fig.add_trace(
        go.Scatter(x=epochs, y=train_acc, name="Train",
                   line=dict(color="#2196F3", width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=test_acc, name="Test",
                   line=dict(color="#F44336", width=2)),
        row=2, col=1
    )

    # 現在のエポック位置を縦線で表示
    fig.add_vline(
        x=selected_epoch,
        line=dict(color="yellow", width=3, dash="solid"),
        row=2, col=1
    )

    # レイアウト調整
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 105], row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )

    # 上段の背景を黒に
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)", row=2, col=1)

    # ニューロン相関行列も生成（軽量版: 5×5, 500サンプル）
    # fixed_dimsが渡されていれば使用
    corr_fig = plot_neuron_correlation_matrix(model, config, grid_size=5, sample_size=500, fixed_dims=None)

    return fig, circular_result, corr_fig, model


def plot_mlp_output_matrix(model, config, use_logits=True):
    """MLP出力行列の可視化（滑らかな波面用にlogitsを使用）"""
    p = config["p"]
    n_tokens = config.get("n_tokens", 2)

    if n_tokens == 2:
        # バッチ処理で高速化
        all_inputs = torch.tensor([[a, b] for a in range(p) for b in range(p)])
        with torch.no_grad():
            all_logits = model(all_inputs)
        all_preds = all_logits.argmax(dim=-1).numpy().reshape(p, p)
        pred_matrix = all_preds

        # 正解クラスのlogitを取得（滑らかな波面用）
        expected = np.array([[(a + b) % p for b in range(p)] for a in range(p)])
        correct_indices = expected.flatten()
        logit_matrix = all_logits[np.arange(p * p), correct_indices].numpy().reshape(p, p)

        xlabel, ylabel = "b", "a"
        title = "(a+b) mod p"
    else:
        # 3トークンモデル用バッチ処理
        all_inputs = torch.tensor([[a, 0, c] for a in range(p) for c in range(p)])
        with torch.no_grad():
            all_logits = model(all_inputs)
        all_preds = all_logits.argmax(dim=-1).numpy().reshape(p, p)
        pred_matrix = all_preds

        expected = np.array([[(a + c) % p for c in range(p)] for a in range(p)])
        correct_indices = expected.flatten()
        logit_matrix = all_logits[np.arange(p * p), correct_indices].numpy().reshape(p, p)

        xlabel, ylabel = "c", "a+b"
        title = "(a+b+c) mod p"

    accuracy = (pred_matrix == expected).mean() * 100

    # 2Dヒートマップ（予測値）
    fig = px.imshow(
        pred_matrix,
        color_continuous_scale="Viridis",
        labels=dict(x=xlabel, y=ylabel, color="Prediction"),
        title=f"MLP Output: {title} (Accuracy: {accuracy:.1f}%)"
    )
    fig.update_layout(height=500)

    return fig, accuracy, pred_matrix, logit_matrix


def plot_mlp_output_3d(logit_matrix, config, interpolation_factor=2):
    """MLP出力の3Dサーフェスプロット（滑らかな波形表示）"""
    from scipy.ndimage import zoom

    p = config["p"]

    # 補間で滑らかにする
    if interpolation_factor > 1:
        smooth_matrix = zoom(logit_matrix, interpolation_factor, order=3)
    else:
        smooth_matrix = logit_matrix

    # x, y座標も補間に合わせる
    new_size = smooth_matrix.shape[0]
    x = np.linspace(0, p-1, new_size)
    y = np.linspace(0, p-1, new_size)

    # 3Dサーフェスプロット
    fig = go.Figure(data=[go.Surface(
        x=x,
        y=y,
        z=smooth_matrix,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Logit (confidence)"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        )
    )])

    fig.update_layout(
        title="3D Surface: Correct Class Logit (Wave Pattern)",
        scene=dict(
            xaxis_title="b",
            yaxis_title="a",
            zaxis_title="Logit",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            bgcolor="black"
        ),
        height=500,
        paper_bgcolor="black",
        font=dict(color="white")
    )

    return fig


def main():
    st.title("🧠 Grokking Analysis Dashboard")
    st.markdown("---")

    # サイドバー
    with st.sidebar:
        st.header("Settings")

        # チェックポイント選択
        checkpoint_dirs = get_checkpoint_dirs()
        if not checkpoint_dirs:
            st.error("No checkpoint directories found!")
            return

        # demo_5epをデフォルトに設定
        default_index = 0
        for i, d in enumerate(checkpoint_dirs):
            if "demo_5ep" in d:
                default_index = i
                break

        selected_dir = st.selectbox(
            "Select Checkpoint Directory",
            checkpoint_dirs,
            index=default_index
        )

        # ファイルパス（best.pt, final.pt, または最新のcheckpoint_epoch_*.pt）
        best_path = os.path.join(selected_dir, "best.pt")
        final_path = os.path.join(selected_dir, "final.pt")

        # 利用可能なチェックポイントをリストアップ
        available_checkpoints = []
        if os.path.exists(best_path):
            available_checkpoints.append(("best.pt (best)", best_path))
        if os.path.exists(final_path):
            available_checkpoints.append(("final.pt", final_path))

        # epoch checkpointsも追加
        epoch_files = sorted([f for f in os.listdir(selected_dir)
                             if f.startswith("checkpoint_epoch_") and f.endswith(".pt")])
        for ef in epoch_files:
            ep_num = ef.replace("checkpoint_epoch_", "").replace(".pt", "")
            available_checkpoints.append((f"epoch {int(ep_num)}", os.path.join(selected_dir, ef)))

        if not available_checkpoints:
            st.error("No checkpoint files found!")
            return

        # チェックポイント選択（複数ある場合）
        if len(available_checkpoints) > 1:
            cp_names = [cp[0] for cp in available_checkpoints]
            selected_cp_idx = st.selectbox(
                "Select Checkpoint",
                range(len(cp_names)),
                format_func=lambda i: cp_names[i],
                index=len(cp_names) - 1  # デフォルトは最新
            )
            checkpoint_path = available_checkpoints[selected_cp_idx][1]
        else:
            checkpoint_path = available_checkpoints[0][1]

        history_path = os.path.join(selected_dir, "history.json")
        fourier_path = os.path.join(selected_dir, "fourier_history.json")

        st.markdown("---")
        st.header("Model Info")

        # モデルロード
        try:
            model, config, epoch = load_model(checkpoint_path)
            analyzer = FourierAnalyzer(model)

            st.success(f"✅ Model loaded (epoch {epoch})")
            st.json({
                "p": config["p"],
                "n_tokens": config.get("n_tokens", 2),
                "d_model": config["d_model"],
                "n_heads": config["n_heads"],
                "n_layers": config["n_layers"]
            })
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    # メインコンテンツ
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Training Progress",
        "🔬 Fourier Analysis",
        "⏱️ Evolution",
        "🎯 Model Output",
        "🎬 Epoch Slider",
        "📐 Fourier Theory",
        "🔍 Attention",
        "🧠 Neurons"
    ])

    with tab1:
        st.header("📈 Training Progress")

        # 解説
        with st.expander("📚 Grokkingとは？", expanded=False):
            st.markdown("""
            **Grokking（グロッキング）** は、ニューラルネットワークが**過学習した後に突然汎化する**現象です。

            ### 典型的な学習パターン
            1. **Phase 1: 記憶（Memorization）**
               - 訓練精度が急速に100%に到達
               - テスト精度は低いまま（過学習状態）
               - モデルは訓練データを「暗記」している

            2. **Phase 2: 汎化（Generalization）**
               - 訓練精度は100%のまま
               - 突然テスト精度が急上昇 ← **これがGrokking!**
               - モデルが「真のアルゴリズム」を発見

            ### なぜ起こる？
            - **Weight Decay（重み減衰）** が鍵
            - 複雑な記憶解は徐々にペナルティを受ける
            - シンプルなフーリエ解が最終的に勝利する

            ### このグラフの見方
            - **青線**: 訓練精度（早期に100%到達）
            - **橙線**: テスト精度（遅れて急上昇 = Grokking）
            - 赤い縦線: Train/Test精度の差が最大の点（過学習ピーク）
            """)

        if os.path.exists(history_path):
            history = load_history(history_path)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                final_train_acc = history["train_acc"][-1] * 100
                st.metric("Final Train Accuracy", f"{final_train_acc:.1f}%")
            with col2:
                final_test_acc = history["test_acc"][-1] * 100
                st.metric("Final Test Accuracy", f"{final_test_acc:.1f}%")
            with col3:
                final_train_loss = history["train_loss"][-1]
                st.metric("Final Train Loss", f"{final_train_loss:.4f}")
            with col4:
                total_epochs = len(history["train_loss"])
                st.metric("Total Epochs", total_epochs)

            fig = plot_training_curves(history)
            st.plotly_chart(fig, use_container_width=True)

            # Grokking検出
            train_acc = np.array(history["train_acc"])
            test_acc = np.array(history["test_acc"])
            gap = train_acc - test_acc
            max_gap_epoch = np.argmax(gap)
            if gap[max_gap_epoch] > 0.3:
                st.info(f"🎯 Grokking検出: エポック{max_gap_epoch}で過学習ピーク（Train-Test差={gap[max_gap_epoch]*100:.1f}%）、その後汎化")
        else:
            st.warning("history.json not found")

    with tab2:
        st.header("🔬 Fourier Analysis")

        # 概要解説
        with st.expander("📚 フーリエ解析とは？", expanded=False):
            st.markdown("""
            **フーリエ解析** では、モデルが学習した内部表現を周波数成分に分解して分析します。

            ### なぜフーリエ表現が重要？
            Grokkingしたモデルは、入力を**フーリエ基底**（cos, sin波）で表現することを学習します。

            ### 見るべきポイント
            | グラフ | 意味 | 良い状態 |
            |--------|------|----------|
            | **Fourier Spectrum** | 埋め込みの周波数成分 | 特定のkにピークが立つ |
            | **Embedding Circle** | 埋め込みの2D射影 | きれいな円形になる |
            | **Dominant Frequencies** | 最も強い周波数 | k=1,2,3などの低周波が強い |

            ### 指標の解釈
            - **Fourier corr > 0.7**: フーリエ表現を学習済み
            - **Circular corr > 0.9**: 円環構造が形成されている
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fourier Spectrum")
            fig_spectrum, dominant = plot_fourier_spectrum(analyzer)
            st.plotly_chart(fig_spectrum, use_container_width=True)

            st.markdown("**Dominant Frequencies:**")
            for freq, power in dominant[:5]:
                st.markdown(f"- k={freq}: power={power:.4f}")

        with col2:
            st.subheader("Embedding Circle")
            fig_circle, circular_result = plot_embedding_circle(analyzer)
            st.plotly_chart(fig_circle, use_container_width=True)

            fourier_result = analyzer.verify_fourier_representation()
            is_fourier = "✅" if fourier_result["is_fourier_representation"] else "❌"
            is_circular = "✅" if circular_result["is_circular"] else "❌"

            st.markdown(f"""
            **Analysis Results:**
            - Fourier Representation: {is_fourier} (corr={fourier_result['best_correlation']:.3f})
            - Circular Structure: {is_circular} (corr={circular_result['angle_correlation']:.3f})
            """)

        st.markdown("---")

        # フーリエ学習セクション
        st.subheader("Interactive Fourier Learning")

        # 解説セクション
        with st.expander("📚 Why Fourier Representation Can Express Addition", expanded=False):
            st.markdown(r"""
### The Key: Angle Addition Formula

Fourier basis functions (cos, sin) have a special property called the **angle addition formula**:

$$\cos\left(\frac{2\pi k(a+b)}{p}\right) = \cos\left(\frac{2\pi ka}{p}\right)\cos\left(\frac{2\pi kb}{p}\right) - \sin\left(\frac{2\pi ka}{p}\right)\sin\left(\frac{2\pi kb}{p}\right)$$

$$\sin\left(\frac{2\pi k(a+b)}{p}\right) = \sin\left(\frac{2\pi ka}{p}\right)\cos\left(\frac{2\pi kb}{p}\right) + \cos\left(\frac{2\pi ka}{p}\right)\sin\left(\frac{2\pi kb}{p}\right)$$

### How the Neural Network Uses This

| Layer | Role |
|-------|------|
| **Embedding** | Encode each token as Fourier components: $a \to [\cos(2\pi ka/p), \sin(2\pi ka/p), ...]$ |
| **MLP** | Compute products using angle addition formula (multiplication + addition) |
| **Output** | Decode from Fourier space back to answer $(a+b) \mod p$ |

### Why Circular Structure Emerges

All pairs $(a, b)$ with the same sum $s = (a+b) \mod p$ have the **same Fourier representation** in the MLP output.

For $p=59$, there are 59 possible sum values ($s=0,1,2,...,58$), each corresponding to a different angle $2\pi s/p$ on a circle.

### Concrete Example (p=5, k=1)

For $a=2, b=3$, answer is $(2+3) \mod 5 = 0$

**Embeddings:**
- $\cos(2\pi \cdot 2/5) \approx -0.81$, $\sin(2\pi \cdot 2/5) \approx 0.59$
- $\cos(2\pi \cdot 3/5) \approx -0.81$, $\sin(2\pi \cdot 3/5) \approx -0.59$

**Angle Addition:**
$$\cos(2\pi \cdot 5/5) = (-0.81)(-0.81) - (0.59)(-0.59) = 0.66 + 0.35 = 1.0 = \cos(0)$$

→ This represents sum = 0! ✓

### Summary

**The angle addition formula transforms "addition" into "combination of multiplications"** — which MLPs can compute. This is why the network learns Fourier representations for modular arithmetic.
            """)

        p = config["p"]
        n = np.arange(p)
        weights = analyzer.get_embedding_weights()

        # 加法定理インタラクティブデモ
        with st.expander("🧮 Interactive Angle Addition Demo", expanded=False):
            st.markdown("**Try it yourself:** Select values of a, b, and k to see the angle addition formula in action.")

            col_demo1, col_demo2, col_demo3 = st.columns(3)
            with col_demo1:
                demo_a = st.slider("a", 0, max(1, p-1), min(2, p-1), key="demo_a")
            with col_demo2:
                demo_b = st.slider("b", 0, max(1, p-1), min(3, p-1), key="demo_b")
            with col_demo3:
                max_k = max(2, min(p//2, 10))  # 最低でも2にする
                demo_k = st.slider("k (frequency)", 1, max_k, 1, key="demo_k")

            demo_sum = (demo_a + demo_b) % p

            # 個別のcos/sin値
            cos_a = np.cos(2 * np.pi * demo_k * demo_a / p)
            sin_a = np.sin(2 * np.pi * demo_k * demo_a / p)
            cos_b = np.cos(2 * np.pi * demo_k * demo_b / p)
            sin_b = np.sin(2 * np.pi * demo_k * demo_b / p)

            # 加法定理による計算
            cos_sum_formula = cos_a * cos_b - sin_a * sin_b
            sin_sum_formula = sin_a * cos_b + cos_a * sin_b

            # 直接計算
            cos_sum_direct = np.cos(2 * np.pi * demo_k * demo_sum / p)
            sin_sum_direct = np.sin(2 * np.pi * demo_k * demo_sum / p)

            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.markdown(f"""
**Input Values:**
- $a = {demo_a}$, $b = {demo_b}$, $k = {demo_k}$
- $(a + b) \\mod {p} = {demo_sum}$

**Fourier Components of a:**
- $\\cos(2\\pi \\cdot {demo_k} \\cdot {demo_a}/{p}) = {cos_a:.4f}$
- $\\sin(2\\pi \\cdot {demo_k} \\cdot {demo_a}/{p}) = {sin_a:.4f}$

**Fourier Components of b:**
- $\\cos(2\\pi \\cdot {demo_k} \\cdot {demo_b}/{p}) = {cos_b:.4f}$
- $\\sin(2\\pi \\cdot {demo_k} \\cdot {demo_b}/{p}) = {sin_b:.4f}$
                """)
            with col_result2:
                st.markdown(f"""
**Angle Addition Formula:**
- $\\cos(a+b) = \\cos(a)\\cos(b) - \\sin(a)\\sin(b)$
- $= ({cos_a:.4f})({cos_b:.4f}) - ({sin_a:.4f})({sin_b:.4f})$
- $= {cos_sum_formula:.4f}$

**Direct Calculation:**
- $\\cos(2\\pi \\cdot {demo_k} \\cdot {demo_sum}/{p}) = {cos_sum_direct:.4f}$

**Match:** {"✅ Yes!" if abs(cos_sum_formula - cos_sum_direct) < 1e-10 else "❌ No"}
                """)

            # 円上での可視化
            fig_demo = go.Figure()

            # 単位円
            theta_circle = np.linspace(0, 2*np.pi, 100)
            fig_demo.add_trace(go.Scatter(
                x=np.cos(theta_circle).tolist(), y=np.sin(theta_circle).tolist(),
                mode="lines", line=dict(color="gray", width=1),
                name="Unit Circle", showlegend=False
            ))

            # 各点をプロット
            fig_demo.add_trace(go.Scatter(
                x=[cos_a], y=[sin_a], mode="markers+text",
                marker=dict(size=15, color="#2196F3"),
                text=[f"a={demo_a}"], textposition="top right",
                name=f"a={demo_a}"
            ))
            fig_demo.add_trace(go.Scatter(
                x=[cos_b], y=[sin_b], mode="markers+text",
                marker=dict(size=15, color="#F44336"),
                text=[f"b={demo_b}"], textposition="top right",
                name=f"b={demo_b}"
            ))
            fig_demo.add_trace(go.Scatter(
                x=[cos_sum_direct], y=[sin_sum_direct], mode="markers+text",
                marker=dict(size=15, color="#4CAF50", symbol="star"),
                text=[f"sum={demo_sum}"], textposition="top right",
                name=f"(a+b) mod {p} = {demo_sum}"
            ))

            fig_demo.update_layout(
                title=f"Fourier Representation on Unit Circle (k={demo_k})",
                xaxis=dict(title="cos", range=[-1.5, 1.5], scaleanchor="y"),
                yaxis=dict(title="sin", range=[-1.5, 1.5]),
                height=400, width=400,
                plot_bgcolor="black", paper_bgcolor="black",
                font=dict(color="white")
            )
            st.plotly_chart(fig_demo, use_container_width=False)

        # 周波数選択
        st.markdown("**Select frequencies to analyze (k values):**")
        available_k = list(range(1, min(p // 2, 25) + 1))
        # dominant_kをavailable_kに含まれる値のみにフィルタリング
        dominant_k = [d[0] for d in dominant[:5] if d[0] in available_k]
        default_k = dominant_k[:3] if dominant_k else [1, 2, 3]

        col_select1, col_select2 = st.columns(2)
        with col_select1:
            selected_k = st.multiselect(
                "Compare frequencies",
                options=available_k,
                default=default_k,
                help="Select multiple k values to compare"
            )
        with col_select2:
            show_superposition = st.checkbox("Show superposition", value=True)
            show_learned = st.checkbox("Show learned embedding", value=True)

        if selected_k:
            # 個別周波数の比較
            st.markdown("#### Individual Frequency Components")

            fig_compare = make_subplots(
                rows=1, cols=len(selected_k),
                subplot_titles=[f"k={k}" for k in selected_k]
            )

            for idx, k in enumerate(selected_k):
                cos_basis = np.cos(2 * np.pi * k * n / p)
                sin_basis = np.sin(2 * np.pi * k * n / p)

                fig_compare.add_trace(
                    go.Scatter(x=n.tolist(), y=cos_basis.tolist(),
                              name=f"cos(2πk{k}n/p)", line=dict(color="#2196F3")),
                    row=1, col=idx+1
                )
                fig_compare.add_trace(
                    go.Scatter(x=n.tolist(), y=sin_basis.tolist(),
                              name=f"sin(2πk{k}n/p)", line=dict(color="#F44336")),
                    row=1, col=idx+1
                )

            fig_compare.update_layout(height=300, showlegend=False,
                                     plot_bgcolor="black", paper_bgcolor="black",
                                     font=dict(color="white"))
            fig_compare.update_xaxes(title_text="n", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            fig_compare.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(fig_compare, use_container_width=True)

            # 重ね合わせと学習済み埋め込みの比較
            st.markdown("#### Superposition vs Learned Embedding")
            st.markdown("""
            <div style="background: rgba(78, 205, 196, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <b>見方:</b> 緑線（理論）とオレンジ線（学習済み）が<b>一致するほど良い</b>。<br>
            • <b>Superposition</b>: 選択した周波数kのcos/sinを重ね合わせた理論的な波形<br>
            • <b>Learned Embedding</b>: モデルが実際に学習した埋め込みの最大分散次元<br>
            → 一致 = モデルがフーリエ表現を正しく学習している証拠
            </div>
            """, unsafe_allow_html=True)

            # 最も分散が大きい次元を取得
            variances = np.var(weights, axis=0)
            top_dim = np.argsort(variances)[-1]
            learned_dim = weights[:, top_dim]
            learned_norm = (learned_dim - learned_dim.mean()) / (learned_dim.std() + 1e-8)

            fig_super = go.Figure()

            # 重ね合わせを計算
            if show_superposition and len(selected_k) > 0:
                superposition = np.zeros(p)
                for k in selected_k:
                    # 各周波数のcos/sinを学習済み埋め込みとの相関で重み付け
                    cos_basis = np.cos(2 * np.pi * k * n / p)
                    sin_basis = np.sin(2 * np.pi * k * n / p)

                    cos_corr = np.corrcoef(learned_dim, cos_basis)[0, 1]
                    sin_corr = np.corrcoef(learned_dim, sin_basis)[0, 1]

                    if not np.isnan(cos_corr):
                        superposition += cos_corr * cos_basis
                    if not np.isnan(sin_corr):
                        superposition += sin_corr * sin_basis

                # 正規化
                if superposition.std() > 0:
                    superposition = (superposition - superposition.mean()) / superposition.std()

                fig_super.add_trace(go.Scatter(
                    x=n.tolist(), y=superposition.tolist(),
                    name=f"Superposition (k={','.join(map(str, selected_k))})",
                    line=dict(color="#4CAF50", width=2)
                ))

            # 学習済み埋め込み
            if show_learned:
                fig_super.add_trace(go.Scatter(
                    x=n.tolist(), y=learned_norm.tolist(),
                    name=f"Learned (dim {top_dim})",
                    line=dict(color="#FF9800", width=2, dash="dash")
                ))

            fig_super.update_layout(
                title="Fourier Superposition vs Learned Embedding",
                xaxis_title="Token n",
                yaxis_title="Normalized Value",
                height=400,
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                legend=dict(orientation="h", y=-0.15)
            )
            fig_super.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            fig_super.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(fig_super, use_container_width=True)

            # 相関表
            st.markdown("#### Correlation with Fourier Bases")
            corr_data = []
            for k in selected_k:
                cos_basis = np.cos(2 * np.pi * k * n / p)
                sin_basis = np.sin(2 * np.pi * k * n / p)
                cos_corr = np.corrcoef(learned_dim, cos_basis)[0, 1]
                sin_corr = np.corrcoef(learned_dim, sin_basis)[0, 1]
                combined = np.sqrt(cos_corr**2 + sin_corr**2) if not (np.isnan(cos_corr) or np.isnan(sin_corr)) else 0
                corr_data.append({
                    "k": k,
                    "cos correlation": f"{cos_corr:.3f}" if not np.isnan(cos_corr) else "N/A",
                    "sin correlation": f"{sin_corr:.3f}" if not np.isnan(sin_corr) else "N/A",
                    "combined": f"{combined:.3f}"
                })
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)

        st.markdown("---")
        st.subheader("Single Frequency Comparison")
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <b>見方:</b> 主要周波数kに対するcos/sinと学習済み埋め込みの比較<br>
        • <b>青線 cos</b>と<b>赤線 sin</b>: 理論的なフーリエ基底<br>
        • <b>緑点線</b>: 学習済み埋め込み（最大分散次元）<br>
        → 緑線がcos/sinのどちらかに近い形 = その周波数を学習している
        </div>
        """, unsafe_allow_html=True)
        fig_comparison = plot_fourier_basis_comparison(analyzer, dominant)
        if fig_comparison:
            st.plotly_chart(fig_comparison, use_container_width=True)

    with tab3:
        st.header("⏱️ Training Evolution")

        # 解説
        with st.expander("📚 学習進化の見方", expanded=False):
            st.markdown("""
            **Training Evolution** では、学習中にモデルの内部表現がどう変化するかを追跡します。

            ### 円環構造の形成過程
            学習が進むにつれて、モデルの内部表現は以下のように変化します：

            1. **初期（ランダム）**: 点がバラバラに分布
            2. **記憶フェーズ**: 少しずつ構造が現れ始める
            3. **汎化フェーズ**: きれいな円環構造が形成される ← **Grokking完了!**

            ### グラフの見方
            - **左: 円環プロット** - 各点は (a+b) mod p の値を表す
            - **右: ニューロン相関** - フーリエ次元間の相関
            - **色**: レインボーカラーで 0→p-1 を表現

            ### 良い学習の指標
            - 点が円周上に等間隔で並ぶ
            - 角度相関（Angle Corr）が 0.9 以上
            - 色が虹の順序で並ぶ
            """)

        # フーリエ履歴があれば表示
        if os.path.exists(fourier_path):
            fourier_history = load_fourier_history(fourier_path)
            fig_evolution = plot_fourier_evolution(fourier_history)
            st.plotly_chart(fig_evolution, use_container_width=True)

        # エポックチェックポイントを取得
        epoch_checkpoints = glob.glob(os.path.join(selected_dir, "epoch_*.pt")) + glob.glob(os.path.join(selected_dir, "checkpoint_epoch_*.pt"))

        if epoch_checkpoints:
            epochs_available = sorted([
                int(os.path.basename(f).replace("checkpoint_epoch_", "").replace("epoch_", "").replace(".pt", ""))
                for f in epoch_checkpoints
            ])

            if epochs_available:
                st.subheader("Epoch Selector")

                # 範囲を絞るオプション
                col_range1, col_range2 = st.columns(2)
                with col_range1:
                    start_idx = st.number_input("Start epoch", min_value=epochs_available[0],
                                                max_value=epochs_available[-1], value=epochs_available[0], step=10)
                with col_range2:
                    end_idx = st.number_input("End epoch", min_value=epochs_available[0],
                                              max_value=epochs_available[-1], value=min(epochs_available[-1], 1000), step=10)

                # 範囲内のエポックをフィルタ
                filtered_epochs = [e for e in epochs_available if start_idx <= e <= end_idx]
                if not filtered_epochs:
                    filtered_epochs = epochs_available

                selected_epoch = st.select_slider(
                    "Select Epoch",
                    options=filtered_epochs,
                    value=filtered_epochs[-1]
                )

                epoch_path = get_epoch_path(selected_dir, selected_epoch)
                if epoch_path:
                    epoch_model, _, _ = load_model(epoch_path)
                    epoch_analyzer = FourierAnalyzer(epoch_model)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_circle_epoch, _ = plot_embedding_circle(epoch_analyzer)
                        st.plotly_chart(fig_circle_epoch, use_container_width=True, key=f"evolution_circle_{selected_epoch}")
                    with col2:
                        fig_spectrum_epoch, _ = plot_fourier_spectrum(epoch_analyzer)
                        st.plotly_chart(fig_spectrum_epoch, use_container_width=True, key=f"evolution_spectrum_{selected_epoch}")
        else:
            st.warning("No epoch checkpoints found in this directory.")

    with tab4:
        st.header("🎯 Model Output Analysis")

        # 解説
        with st.expander("📚 モデル出力の見方", expanded=False):
            st.markdown("""
            **Model Output Analysis** では、モデルの予測結果を2D/3Dで可視化します。

            ### 2つの表示モード
            | モード | 内容 | 特徴 |
            |--------|------|------|
            | **Predictions (Cyclic)** | 予測値（離散） | モジュラ演算の周期性が見える |
            | **Logits (Continuous)** | 出力ロジット（連続） | 滑らかな波面が見える |

            ### 2D Heatmap の見方
            - **横軸**: 入力 b
            - **縦軸**: 入力 a
            - **色**: 予測値 or ロジット値
            - **パターン**: 斜めの縞模様が正しい（(a+b) mod p の等高線）

            ### 3D Surface の見方
            - **波面の形状**: cos(ω(a+b)) のような波形が見えるはず
            - **滑らかさ**: Logitsモードで滑らかな表面が見えれば学習成功

            ### カラーマップの選択
            - **HSV/Phase**: 循環データに最適（0とp-1が同じ色）
            - **Twilight**: 周期性を強調
            """)

        fig_matrix, accuracy, pred_matrix, logit_matrix = plot_mlp_output_matrix(model, config)
        p = config["p"]

        st.metric("Model Accuracy", f"{accuracy:.1f}%")

        # カラーマップ選択
        col_mode, col_cmap = st.columns(2)
        with col_mode:
            view_mode = st.radio(
                "View Mode",
                ["Predictions (Cyclic)", "Logits (Continuous)"],
                horizontal=True
            )
        with col_cmap:
            # 循環型カラーマップオプション
            cyclical_cmaps = ["HSV", "Phase", "Edge", "IceFire", "Twilight"]
            selected_cmap = st.selectbox("Colormap (cyclical)", cyclical_cmaps, index=0)

        # 2Dと3Dを並べて表示
        col_2d, col_3d = st.columns(2)

        if view_mode == "Predictions (Cyclic)":
            with col_2d:
                st.subheader("2D Heatmap")
                # 循環型カラーマップで予測値を表示
                fig_2d_cyclic = go.Figure(data=go.Heatmap(
                    z=pred_matrix,
                    colorscale=selected_cmap,
                    zmin=0, zmax=p,  # 0〜p-1で循環
                    colorbar=dict(title=f"(a+b) mod {p}", tickvals=[0, p//4, p//2, 3*p//4, p-1])
                ))
                fig_2d_cyclic.update_layout(
                    xaxis_title="b", yaxis_title="a",
                    height=500, plot_bgcolor="black", paper_bgcolor="black",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_2d_cyclic, use_container_width=True)
            with col_3d:
                st.subheader("3D Surface")
                fig_3d_pred = go.Figure(data=[go.Surface(
                    z=pred_matrix,
                    colorscale=selected_cmap,
                    cmin=0, cmax=p,
                    showscale=True,
                    colorbar=dict(title=f"(a+b) mod {p}")
                )])
                fig_3d_pred.update_layout(
                    scene=dict(xaxis_title="b", yaxis_title="a", zaxis_title="Prediction", bgcolor="black"),
                    height=500, paper_bgcolor="black", font=dict(color="white")
                )
                st.plotly_chart(fig_3d_pred, use_container_width=True)
        else:
            # Logit表示（滑らかな波）
            from scipy.ndimage import zoom
            smooth_logit = zoom(logit_matrix, 2, order=3)
            x = np.linspace(0, p-1, smooth_logit.shape[1])
            y = np.linspace(0, p-1, smooth_logit.shape[0])

            with col_2d:
                st.subheader("2D Heatmap (Logits)")
                fig_2d_smooth = go.Figure(data=go.Heatmap(
                    z=smooth_logit,
                    x=x,
                    y=y,
                    colorscale="Viridis",
                    colorbar=dict(title="Logit (confidence)")
                ))
                fig_2d_smooth.update_layout(
                    xaxis_title="b", yaxis_title="a",
                    height=500, plot_bgcolor="black", paper_bgcolor="black",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_2d_smooth, use_container_width=True)

            with col_3d:
                st.subheader("3D Surface (Wave)")
                fig_3d = plot_mlp_output_3d(logit_matrix, config)
                st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("Test Predictions")

        col1, col2 = st.columns(2)
        n_tokens = config.get("n_tokens", 2)
        p = config["p"]

        with col1:
            a = st.number_input("a", min_value=0, max_value=p-1, value=0)
            b = st.number_input("b", min_value=0, max_value=p-1, value=0)
            if n_tokens == 3:
                c = st.number_input("c", min_value=0, max_value=p-1, value=0)

        with col2:
            if n_tokens == 2:
                x = torch.tensor([[a, b]])
                expected = (a + b) % p
            else:
                x = torch.tensor([[a, b, c]])
                expected = (a + b + c) % p

            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()

            is_correct = pred == expected
            color = "green" if is_correct else "red"

            st.markdown(f"""
            **Input:** {tuple(x.squeeze().tolist())}

            **Prediction:** <span style='color:{color};font-size:24px;'>{pred}</span>

            **Expected:** {expected}

            **Correct:** {'✅' if is_correct else '❌'}
            """, unsafe_allow_html=True)

            # 確率分布
            fig_probs = go.Figure()
            fig_probs.add_trace(go.Bar(
                x=list(range(p)),
                y=probs.tolist(),
                marker_color=["green" if i == expected else "blue" for i in range(p)]
            ))
            fig_probs.update_layout(
                title="Output Probability Distribution",
                xaxis_title="Class",
                yaxis_title="Probability",
                height=300
            )
            st.plotly_chart(fig_probs, use_container_width=True)

    with tab5:
        st.header("🎬 Epoch Slider - 学習進捗アニメーション")

        # 解説
        with st.expander("📚 アニメーションの見方", expanded=False):
            st.markdown("""
            **Epoch Slider** では、学習の各エポックでモデルの内部表現がどう変化するかをアニメーションで確認できます。

            ### 表示内容
            | パネル | 内容 |
            |--------|------|
            | **円環プロット** | 各 (a+b) mod p の値の内部表現を2Dに射影 |
            | **学習曲線** | Train/Test精度の推移（現在位置を表示） |
            | **ニューロン相関** | フーリエ次元間の相関行列 |

            ### アニメーションの操作
            - **▶️ Play**: アニメーション再生
            - **⏸️ Pause**: 一時停止
            - **スライダー**: 任意のエポックにジャンプ

            ### 見るべきポイント
            1. **初期**: 点がランダムに分布
            2. **記憶フェーズ**: Train精度↑、Test精度は低いまま
            3. **Grokking**: Test精度が急上昇、円環が形成される
            4. **最終状態**: きれいな円環 + 高い角度相関

            ### 検出次元について
            - **cos次元/sin次元**: 最終モデルで検出したフーリエペア
            - この次元ペアを全エポックで固定して追跡
            """)

        if os.path.exists(history_path):
            history = load_history(history_path)

            # 利用可能なエポックチェックポイントを取得
            epoch_checkpoints = glob.glob(os.path.join(selected_dir, "epoch_*.pt")) + glob.glob(os.path.join(selected_dir, "checkpoint_epoch_*.pt"))
            if epoch_checkpoints:
                epochs_available = sorted([
                    int(os.path.basename(f).replace("checkpoint_epoch_", "").replace("epoch_", "").replace(".pt", ""))
                    for f in epoch_checkpoints
                ])

                if epochs_available:
                    # キャッシュクリアボタン
                    if st.button("🔄 Clear Cache", key="clear_cache_btn"):
                        st.cache_data.clear()
                        st.rerun()

                    # フーリエ基底との相関が高い次元を計算
                    # キャッシュキーにチェックポイント数を含める
                    n_checkpoints = len(epochs_available)

                    @st.cache_data
                    def get_fourier_dims(_dir, _p, _n_tokens, _d_model, _n_cp, n_dims=10):
                        """フーリエ基底との相関が高い次元を取得（cos/sin次元ペアも検出）"""
                        p = _p
                        n_tokens = _n_tokens

                        # 最終モデルをロード
                        best_path = os.path.join(_dir, "best.pt")
                        final_path = os.path.join(_dir, "final.pt")
                        if os.path.exists(best_path):
                            ref_path = best_path
                        elif os.path.exists(final_path):
                            ref_path = final_path
                        else:
                            # checkpoint_epoch_*.pt から最新を取得
                            epoch_files = sorted([f for f in os.listdir(_dir)
                                                 if f.startswith("checkpoint_epoch_") and f.endswith(".pt")])
                            if epoch_files:
                                ref_path = os.path.join(_dir, epoch_files[-1])
                            else:
                                return None, None, None, None

                        ref_model, _, _ = load_model(ref_path)

                        # 各(a+b) mod p の値に対する埋め込みを計算
                        sum_embeddings = np.zeros((p, _d_model))
                        samples_per_sum = 5

                        all_pairs = []
                        sum_labels = []
                        for s in range(p):
                            for i in range(samples_per_sum):
                                a = (s + i * 17) % p
                                b = (s - a) % p
                                all_pairs.append([a, b])
                                sum_labels.append(s)

                        if n_tokens == 2:
                            inputs = torch.tensor(all_pairs)
                        else:
                            inputs = torch.tensor([[a, b, 0] for a, b in all_pairs])

                        with torch.no_grad():
                            _, intermediates = ref_model.forward_with_intermediates(inputs)
                        pooled = intermediates["pooled"].numpy()

                        # 各和値ごとに平均
                        sum_labels = np.array(sum_labels)
                        for s in range(p):
                            mask = sum_labels == s
                            sum_embeddings[s] = pooled[mask].mean(axis=0)

                        # 最良のcos/sin次元ペアを検出
                        s_values = np.arange(p)
                        best_k = 1
                        best_cos_dim = 0
                        best_sin_dim = 1
                        best_total_corr = 0
                        all_dim_info = []  # 各次元の情報を保存

                        for k in range(1, min(p // 4, 20) + 1):
                            cos_basis = np.cos(2 * np.pi * k * s_values / p)
                            sin_basis = np.sin(2 * np.pi * k * s_values / p)

                            cos_corrs = []
                            sin_corrs = []
                            for d in range(_d_model):
                                dim_vals = sum_embeddings[:, d]
                                if np.std(dim_vals) > 0.01:
                                    cc = np.corrcoef(dim_vals, cos_basis)[0, 1]
                                    sc = np.corrcoef(dim_vals, sin_basis)[0, 1]
                                    cos_corrs.append((d, cc if not np.isnan(cc) else 0))
                                    sin_corrs.append((d, sc if not np.isnan(sc) else 0))
                                else:
                                    cos_corrs.append((d, 0))
                                    sin_corrs.append((d, 0))

                            # 最もcos/sinに相関が高い次元を見つける
                            cos_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                            sin_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

                            cos_dim = cos_corrs[0][0]
                            cos_val = abs(cos_corrs[0][1])
                            # sin次元はcos次元と異なるものを選ぶ
                            sin_dim = sin_corrs[0][0] if sin_corrs[0][0] != cos_dim else sin_corrs[1][0]
                            sin_val = abs(sin_corrs[0][1]) if sin_corrs[0][0] != cos_dim else abs(sin_corrs[1][1])

                            total_corr = cos_val + sin_val
                            if total_corr > best_total_corr:
                                best_total_corr = total_corr
                                best_k = k
                                best_cos_dim = cos_dim
                                best_sin_dim = sin_dim

                        # 各次元のフーリエスコアを計算
                        fourier_scores = []
                        best_k_per_dim = []

                        for dim in range(_d_model):
                            dim_vals = sum_embeddings[:, dim]
                            best_corr = 0
                            best_dim_k = 1

                            for k in range(1, min(p // 4, 20) + 1):
                                cos_basis = np.cos(2 * np.pi * k * s_values / p)
                                sin_basis = np.sin(2 * np.pi * k * s_values / p)

                                if np.std(dim_vals) > 0.01:
                                    cos_corr = abs(np.corrcoef(dim_vals, cos_basis)[0, 1])
                                    sin_corr = abs(np.corrcoef(dim_vals, sin_basis)[0, 1])
                                    if not np.isnan(cos_corr) and cos_corr > best_corr:
                                        best_corr = cos_corr
                                        best_dim_k = k
                                    if not np.isnan(sin_corr) and sin_corr > best_corr:
                                        best_corr = sin_corr
                                        best_dim_k = k

                            fourier_scores.append(best_corr)
                            best_k_per_dim.append(best_dim_k)

                        # フーリエ相関が高い次元を選択
                        top_indices = np.argsort(fourier_scores)[::-1][:n_dims]
                        top_dims = top_indices.tolist()
                        top_k = [best_k_per_dim[i] for i in top_indices]
                        top_corrs = [fourier_scores[i] for i in top_indices]

                        # cos/sin次元ペア情報
                        best_pair = {
                            "k": best_k,
                            "cos_dim": best_cos_dim,
                            "sin_dim": best_sin_dim,
                            "total_corr": best_total_corr
                        }

                        return top_dims, top_k, top_corrs, best_pair

                    fixed_dims, fourier_k, fourier_corrs, best_pair = get_fourier_dims(
                        selected_dir, config["p"], config.get("n_tokens", 2), config.get("d_model", 128),
                        n_checkpoints, n_dims=10
                    )

                    if fixed_dims and best_pair:
                        # 表示用に整形
                        dim_info = ", ".join([f"d{d}(k={k})" for d, k in zip(fixed_dims[:5], fourier_k[:5])])
                        pair_info = f"最良ペア: cos=d{best_pair['cos_dim']}, sin=d{best_pair['sin_dim']} (k={best_pair['k']}, corr={best_pair['total_corr']:.3f})"
                        st.success(f"フーリエ検出次元（上位10）: {dim_info}...")
                        st.info(pair_info)
                    else:
                        st.warning("次元の検出に失敗しました")
                        best_pair = {"k": 1, "cos_dim": 0, "sin_dim": 1, "total_corr": 0}

                    # 事前キャッシュ: 全エポックの可視化データを読み込み
                    @st.cache_data(show_spinner="エポックデータをプリロード中...")
                    def preload_epoch_data(_dir, _p, _n_tokens, _n_cp, _epochs, _fixed_dims, _best_pair_tuple, sample_step=1):
                        """全エポックの可視化データを事前計算（固定次元ペアを使用）"""
                        p = _p
                        n_tokens = _n_tokens

                        # 最終モデルで検出した固定次元ペアを使用（タプルから展開）
                        fixed_k, fixed_cos_dim, fixed_sin_dim = _best_pair_tuple

                        # 円環用: 各和値に対して複数サンプルを用意して平均を取る（軽量化）
                        np.random.seed(42)
                        samples_per_sum = 3  # 5→3に削減
                        all_circle_pairs = []
                        sum_labels = []

                        for s in range(p):
                            for i in range(samples_per_sum):
                                a = (s + i * 17) % p
                                b = (s - a) % p
                                all_circle_pairs.append([a, b])
                                sum_labels.append(s)

                        sum_labels = np.array(sum_labels)

                        # 相関行列用サンプル（軽量化）
                        corr_pairs = [[np.random.randint(p), np.random.randint(p)] for _ in range(100)]

                        if n_tokens == 2:
                            inputs_circle = torch.tensor(all_circle_pairs)
                            inputs_corr = torch.tensor(corr_pairs)
                        else:
                            inputs_circle = torch.tensor([[a, b, 0] for a, b in all_circle_pairs])
                            inputs_corr = torch.tensor([[a, b, 0] for a, b in corr_pairs])

                        epoch_data = {}
                        sampled_epochs = _epochs[::sample_step]

                        for ep in sampled_epochs:
                            ep_path = get_epoch_path(_dir, ep)
                            if ep_path is None:
                                continue

                            try:
                                ep_model, _, _ = load_model(ep_path)

                                with torch.no_grad():
                                    _, inter_circle = ep_model.forward_with_intermediates(inputs_circle)
                                    _, inter_corr = ep_model.forward_with_intermediates(inputs_corr)

                                pooled_all = inter_circle["pooled"].numpy()
                                pooled_corr = inter_corr["pooled"].numpy()

                                # 各和値ごとに平均を取る
                                sum_embeddings = np.zeros((p, pooled_all.shape[1]))
                                for s in range(p):
                                    mask = sum_labels == s
                                    sum_embeddings[s] = pooled_all[mask].mean(axis=0)

                                # 固定の次元ペアを使用（最終モデルで検出したもの）
                                proj_2d = sum_embeddings[:, [fixed_cos_dim, fixed_sin_dim]]

                                # 円環性を計算（角度と理論角度の相関）
                                center = proj_2d.mean(axis=0)
                                centered = proj_2d - center
                                angles = np.arctan2(centered[:, 1], centered[:, 0])
                                expected_angles = 2 * np.pi * fixed_k * np.arange(p) / p - np.pi

                                best_corr = 0
                                for shift in range(p):
                                    shifted_expected = np.roll(expected_angles, shift)
                                    corr = np.corrcoef(angles, shifted_expected)[0, 1]
                                    if not np.isnan(corr):
                                        best_corr = max(best_corr, abs(corr))

                                # 相関行列用データ
                                pooled_sampled = pooled_corr[:, _fixed_dims[:10]] if _fixed_dims else pooled_corr[:, :10]

                                epoch_data[ep] = {
                                    "proj_2d": proj_2d,  # (p, 2) - 各(a+b) mod p の表現
                                    "angle_corr": best_corr,
                                    "pooled": pooled_sampled  # (200, 10)
                                }
                            except Exception as e:
                                continue

                        return epoch_data, sampled_epochs

                    # エポック刻み設定
                    col_step1, col_step2 = st.columns([1, 2])
                    with col_step1:
                        auto_step = st.checkbox("自動刻み", value=True, help="フレーム数が50以下になるよう自動調整")
                    with col_step2:
                        if auto_step:
                            sample_step = max(1, len(epochs_available) // 30)  # 50→30フレームに削減
                            st.info(f"自動設定: {sample_step}エポック間隔（約30フレーム）")
                        else:
                            sample_step = st.number_input(
                                "エポック刻み",
                                min_value=1,
                                max_value=max(1, len(epochs_available) // 5),
                                value=10,
                                step=5,
                                help="何エポックごとにフレームを作成するか"
                            )

                    # best_pairをタプルに変換（キャッシュ用）
                    best_pair_tuple = (best_pair["k"], best_pair["cos_dim"], best_pair["sin_dim"])

                    with st.spinner("データをプリロード中..."):
                        epoch_data, sampled_epochs = preload_epoch_data(
                            selected_dir, config["p"], config.get("n_tokens", 2),
                            n_checkpoints, epochs_available, fixed_dims, best_pair_tuple, sample_step
                        )

                    st.info(f"プリロード完了: {len(epoch_data)}フレーム（{sample_step}エポック間隔）")

                    # Plotlyアニメーション作成（学習曲線統合版）
                    if epoch_data:
                        grid_size = 7  # 7x7グリッド
                        first_ep = sampled_epochs[0]
                        first_data = epoch_data.get(first_ep, {})

                        # 全エポックから軸の範囲を計算（固定用）
                        all_proj_x, all_proj_y = [], []
                        all_pooled = [[] for _ in range(grid_size)]
                        for ep_data in epoch_data.values():
                            proj = ep_data["proj_2d"]
                            pooled = ep_data["pooled"]
                            all_proj_x.extend(proj[:, 0].tolist())
                            all_proj_y.extend(proj[:, 1].tolist())
                            for i in range(min(grid_size, pooled.shape[1])):
                                all_pooled[i].extend(pooled[:, i].tolist())

                        # 軸範囲を計算（10%マージン）
                        def calc_range(data):
                            if not data:
                                return [-1, 1]
                            mn, mx = min(data), max(data)
                            margin = (mx - mn) * 0.1 + 0.01
                            return [mn - margin, mx + margin]

                        proj_x_range = calc_range(all_proj_x)
                        proj_y_range = calc_range(all_proj_y)
                        pooled_ranges = [calc_range(d) for d in all_pooled]

                        # 学習曲線データ
                        epochs_list = list(range(1, len(history["train_acc"]) + 1))
                        train_acc = [a * 100 for a in history["train_acc"]]
                        test_acc = [a * 100 for a in history["test_acc"]]

                        # トレース数: 埋め込み(1) + 相関行列(25) + 学習曲線(2) + 縦線(1) = 29
                        n_corr_traces = grid_size * grid_size

                        # フレーム作成
                        frames = []
                        slider_steps = []

                        for ep in sampled_epochs:
                            if ep not in epoch_data:
                                continue
                            data = epoch_data[ep]
                            proj = data["proj_2d"]
                            pooled = data["pooled"]
                            angle_corr = data.get("angle_corr", 0)

                            frame_traces = []

                            # 埋め込み空間（(a+b) mod p の値で色分け - 円環上に配置されるはず）
                            # proj の各点は sum=0,1,...,p-1 の順
                            p_val = config["p"]

                            # 点を線で結ぶ（円を形成するはず）
                            x_line = proj[:, 0].tolist() + [proj[0, 0]]
                            y_line = proj[:, 1].tolist() + [proj[0, 1]]

                            frame_traces.append(go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                line=dict(color="rgba(128,128,128,0.3)", width=1),
                                showlegend=False
                            ))

                            frame_traces.append(go.Scatter(
                                x=proj[:, 0].tolist(),
                                y=proj[:, 1].tolist(),
                                mode="markers",
                                marker=dict(
                                    color=list(range(p_val)),
                                    colorscale="HSV",  # HSVで円環状の色を表現
                                    size=6,
                                    opacity=0.9
                                ),
                                showlegend=False
                            ))

                            # 5x5相関行列（入力値で色分け）
                            for i in range(grid_size):
                                for j in range(grid_size):
                                    frame_traces.append(go.Scatter(
                                        x=pooled[:, j].tolist(),
                                        y=pooled[:, i].tolist(),
                                        mode="markers",
                                        marker=dict(
                                            color=list(range(len(pooled))),
                                            colorscale="Plasma",
                                            size=3,
                                            opacity=0.6
                                        ),
                                        showlegend=False
                                    ))

                            # 学習曲線（固定）
                            frame_traces.append(go.Scatter(
                                x=epochs_list, y=train_acc,
                                mode="lines", line=dict(color="#2196F3", width=1.5),
                                showlegend=False
                            ))
                            frame_traces.append(go.Scatter(
                                x=epochs_list, y=test_acc,
                                mode="lines", line=dict(color="#F44336", width=1.5),
                                showlegend=False
                            ))

                            # 現在位置の縦線
                            frame_traces.append(go.Scatter(
                                x=[ep, ep], y=[0, 100],
                                mode="lines", line=dict(color="#FFFF00", width=2),
                                showlegend=False
                            ))

                            frames.append(go.Frame(
                                data=frame_traces,
                                name=str(ep),
                                layout=go.Layout(
                                    annotations=[dict(
                                        text=f"Epoch {ep} | Train: {train_acc[min(ep-1, len(train_acc)-1)]:.1f}% | Test: {test_acc[min(ep-1, len(test_acc)-1)]:.1f}% | Circle: {angle_corr:.2f}",
                                        xref="paper", yref="paper",
                                        x=0.5, y=1.02, showarrow=False,
                                        font=dict(size=14, color="white")
                                    )]
                                )
                            ))

                            slider_steps.append({
                                "args": [[str(ep)], {
                                    "frame": {"duration": 100, "redraw": False},
                                    "transition": {"duration": 50, "easing": "linear"},
                                    "mode": "immediate"
                                }],
                                "label": str(ep),
                                "method": "animate"
                            })

                        # メインfigure: 8行8列（7x7相関行列 + 1行学習曲線、左に埋め込み）
                        # グリッドに縦幅を多く割り当て
                        grid_row_height = 0.12  # 各グリッド行の高さ
                        curve_row_height = 0.10  # 学習曲線の高さ
                        fig = make_subplots(
                            rows=grid_size + 1, cols=grid_size + 1,
                            column_widths=[0.28] + [0.103] * grid_size,
                            row_heights=[grid_row_height] * grid_size + [curve_row_height],
                            specs=[[{"rowspan": grid_size}] + [{}] * grid_size] +
                                  [[None] + [{}] * grid_size for _ in range(grid_size - 1)] +
                                  [[{"colspan": grid_size + 1}] + [None] * grid_size],
                            horizontal_spacing=0.008,
                            vertical_spacing=0.015
                        )

                        # 初期データ
                        if first_data:
                            proj = first_data["proj_2d"]
                            pooled = first_data["pooled"]
                            p_val = config["p"]

                            # 埋め込み空間: 点を結ぶ線（円を形成）
                            x_line = proj[:, 0].tolist() + [proj[0, 0]]
                            y_line = proj[:, 1].tolist() + [proj[0, 1]]
                            fig.add_trace(go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                line=dict(color="rgba(128,128,128,0.3)", width=1),
                                showlegend=False
                            ), row=1, col=1)

                            # 埋め込み空間: 散布図（(a+b) mod p で色分け）
                            fig.add_trace(go.Scatter(
                                x=proj[:, 0].tolist(),
                                y=proj[:, 1].tolist(),
                                mode="markers",
                                marker=dict(
                                    color=list(range(p_val)),
                                    colorscale="HSV",
                                    size=6,
                                    opacity=0.9
                                ),
                                showlegend=False
                            ), row=1, col=1)

                            # 10x10相関行列
                            for i in range(grid_size):
                                for j in range(grid_size):
                                    fig.add_trace(go.Scatter(
                                        x=pooled[:, j].tolist(),
                                        y=pooled[:, i].tolist(),
                                        mode="markers",
                                        marker=dict(
                                            color=list(range(len(pooled))),
                                            colorscale="Plasma",
                                            size=3,
                                            opacity=0.6
                                        ),
                                        showlegend=False
                                    ), row=i+1, col=j+2)

                            # 学習曲線
                            fig.add_trace(go.Scatter(
                                x=epochs_list, y=train_acc,
                                mode="lines", line=dict(color="#2196F3", width=1.5),
                                name="Train", showlegend=True
                            ), row=grid_size+1, col=1)
                            fig.add_trace(go.Scatter(
                                x=epochs_list, y=test_acc,
                                mode="lines", line=dict(color="#F44336", width=1.5),
                                name="Test", showlegend=True
                            ), row=grid_size+1, col=1)

                            # 現在位置の縦線
                            fig.add_trace(go.Scatter(
                                x=[first_ep, first_ep], y=[0, 100],
                                mode="lines", line=dict(color="#FFFF00", width=2),
                                showlegend=False
                            ), row=grid_size+1, col=1)

                        # 初期angle correlation
                        first_angle_corr = first_data.get("angle_corr", 0) if first_data else 0

                        # レイアウト
                        fig.update_layout(
                            height=900,
                            plot_bgcolor="black",
                            paper_bgcolor="black",
                            font=dict(color="white"),
                            margin=dict(t=40, b=70, l=15, r=15),
                            annotations=[dict(
                                text=f"Epoch {first_ep} | Train: {train_acc[min(first_ep-1, len(train_acc)-1)]:.1f}% | Test: {test_acc[min(first_ep-1, len(test_acc)-1)]:.1f}% | Circle: {first_angle_corr:.2f}",
                                xref="paper", yref="paper",
                                x=0.5, y=1.02, showarrow=False,
                                font=dict(size=14, color="white")
                            )],
                            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                            updatemenus=[{
                                "type": "buttons",
                                "showactive": False,
                                "y": -0.15,
                                "x": 0.05,
                                "buttons": [
                                    {"label": "▶ 再生", "method": "animate", "args": [None, {
                                        "frame": {"duration": 100, "redraw": False},
                                        "transition": {"duration": 50, "easing": "linear"},
                                        "fromcurrent": True,
                                        "mode": "immediate"
                                    }]},
                                    {"label": "⏸ 停止", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                                ]
                            }],
                            sliders=[{
                                "active": 0,
                                "steps": slider_steps,
                                "x": 0.2,
                                "len": 0.75,
                                "y": -0.08,
                                "currentvalue": {"prefix": "Epoch: ", "visible": True, "xanchor": "center"},
                                "transition": {"duration": 50, "easing": "linear"}
                            }]
                        )

                        # 軸設定（固定範囲で滑らかなアニメーション）
                        fig.update_xaxes(showticklabels=False, showgrid=False)
                        fig.update_yaxes(showticklabels=False, showgrid=False)

                        # 円環プロットの軸を固定
                        fig.update_xaxes(range=proj_x_range, row=1, col=1)
                        fig.update_yaxes(range=proj_y_range, row=1, col=1)

                        # 相関行列の軸を固定
                        for i in range(grid_size):
                            for j in range(grid_size):
                                fig.update_xaxes(range=pooled_ranges[j], row=i+1, col=j+2)
                                fig.update_yaxes(range=pooled_ranges[i], row=i+1, col=j+2)

                        # 学習曲線の軸は表示
                        last_row_idx = grid_size + 1
                        fig.update_xaxes(showticklabels=True, showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                                        title_text="Epoch", row=last_row_idx, col=1)
                        fig.update_yaxes(showticklabels=True, showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                                        title_text="Acc%", range=[0, 105], row=last_row_idx, col=1)

                        fig.frames = frames

                        # グラフ表示
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("エポックデータが見つかりません")
                else:
                    st.warning("エポックチェックポイントが見つかりません")
            else:
                st.warning("エポックチェックポイントが見つかりません")
        else:
            st.warning("history.json が見つかりません")

    # ===== Tab 6: Fourier Theory =====
    with tab6:
        st.header("📐 Fourier Theory")
        st.markdown("""
        **Grokkingの核心**: Transformerはモジュラ加算をフーリエ基底で学習します。

        $$\\cos(\\omega(a+b)) = \\cos(\\omega a)\\cos(\\omega b) - \\sin(\\omega a)\\sin(\\omega b)$$
        """)

        p = config["p"]

        # 周波数選択
        col1, col2 = st.columns([1, 3])
        with col1:
            freq_k = st.slider("周波数 k", 1, min(p // 4, 20), 8, key="theory_freq_k")

        omega = 2 * np.pi * freq_k / p

        # --- セクション1: 計算フロー ---
        st.subheader("1️⃣ 計算フロー（Step by Step）")

        col_a, col_b = st.columns(2)
        with col_a:
            a_val = st.number_input("a", 0, p - 1, 15, key="theory_a")
        with col_b:
            b_val = st.number_input("b", 0, p - 1, 25, key="theory_b")

        # 計算
        cos_a = np.cos(omega * a_val)
        sin_a = np.sin(omega * a_val)
        cos_b = np.cos(omega * b_val)
        sin_b = np.sin(omega * b_val)
        cos_cos = cos_a * cos_b
        sin_sin = sin_a * sin_b
        result_fourier = cos_cos - sin_sin
        result_direct = np.cos(omega * (a_val + b_val))
        answer = (a_val + b_val) % p

        # フロー図
        flow_fig = go.Figure()

        # ステップ配置
        steps = [
            {"x": 0, "text": f"入力<br>a={a_val}, b={b_val}", "color": "#667EEA"},
            {"x": 1, "text": f"埋め込み<br>cos(ωa)={cos_a:.3f}<br>sin(ωa)={sin_a:.3f}<br>cos(ωb)={cos_b:.3f}<br>sin(ωb)={sin_b:.3f}", "color": "#FFD700"},
            {"x": 2, "text": f"Attention<br>cos·cos={cos_cos:.3f}<br>sin·sin={sin_sin:.3f}", "color": "#FFA500"},
            {"x": 3, "text": f"MLP<br>cos·cos - sin·sin<br>={result_fourier:.3f}", "color": "#4ECDC4"},
            {"x": 4, "text": f"出力<br>({a_val}+{b_val}) mod {p}<br>= {answer}", "color": "#FF6B6B"},
        ]

        for i, step in enumerate(steps):
            flow_fig.add_trace(go.Scatter(
                x=[step["x"]], y=[0],
                mode="markers+text",
                marker=dict(size=80, color=step["color"], symbol="square"),
                text=step["text"],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                showlegend=False
            ))
            if i < len(steps) - 1:
                flow_fig.add_annotation(
                    x=step["x"] + 0.5, y=0,
                    ax=step["x"] + 0.3, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="white"
                )

        flow_fig.update_layout(
            title=f"計算フロー: ({a_val} + {b_val}) mod {p} = {answer}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(flow_fig, use_container_width=True, key="flow_chart")

        # 検証
        st.success(f"✅ 加法定理の検証: cos(ω({a_val}+{b_val})) = {result_direct:.6f}, cos·cos - sin·sin = {result_fourier:.6f}, 差 = {abs(result_direct - result_fourier):.2e}")

        # --- セクション2: 3D表面比較 ---
        st.subheader("2️⃣ 3D表面: cos·cos, sin·sin, cos(x+y)")

        # 軽量化: グリッドサイズを制限
        grid_size = min(30, p)
        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        Z_coscos = np.cos(omega * X) * np.cos(omega * Y)
        Z_sinsin = np.sin(omega * X) * np.sin(omega * Y)
        Z_sum = np.cos(omega * (X + Y))

        surface_col1, surface_col2, surface_col3 = st.columns(3)

        with surface_col1:
            fig1 = go.Figure(data=[go.Surface(z=Z_coscos, x=X, y=Y, colorscale="YlOrBr", showscale=False)])
            fig1.update_layout(
                title="cos(ωx)·cos(ωy)",
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=""),
                height=300, margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig1, use_container_width=True, key="surface_coscos")

        with surface_col2:
            fig2 = go.Figure(data=[go.Surface(z=Z_sinsin, x=X, y=Y, colorscale="Oranges", showscale=False)])
            fig2.update_layout(
                title="sin(ωx)·sin(ωy)",
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=""),
                height=300, margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True, key="surface_sinsin")

        with surface_col3:
            fig3 = go.Figure(data=[go.Surface(z=Z_sum, x=X, y=Y, colorscale="Teal", showscale=False)])
            fig3.update_layout(
                title="cos(ω(x+y)) = cos·cos - sin·sin",
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=""),
                height=300, margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig3, use_container_width=True, key="surface_sum")

        # --- セクション3: ニューロン波形比較 ---
        st.subheader("3️⃣ ニューロン波形 vs 理論的cos/sin")

        x_vals = np.arange(p)

        # 実際のニューロン出力を取得
        try:
            n_tokens = config.get("n_tokens", 2)
            if n_tokens == 2:
                inputs = torch.tensor([[x, 0] for x in range(p)])
            else:
                inputs = torch.tensor([[x, 0, 0] for x in range(p)])

            with torch.no_grad():
                _, intermediates = model.forward_with_intermediates(inputs)
            pooled = intermediates["pooled"].numpy()

            # 各次元のcos/sin相関を計算（高速版）
            s_values = np.arange(p)
            cos_basis = np.cos(omega * s_values)
            sin_basis = np.sin(omega * s_values)

            best_cos_dim, best_cos_corr = 0, 0
            best_sin_dim, best_sin_corr = 0, 0

            for d in range(pooled.shape[1]):
                dim_vals = pooled[:, d]
                if np.std(dim_vals) > 0.01:
                    cc = abs(np.corrcoef(dim_vals, cos_basis)[0, 1])
                    sc = abs(np.corrcoef(dim_vals, sin_basis)[0, 1])
                    if not np.isnan(cc) and cc > best_cos_corr:
                        best_cos_corr = cc
                        best_cos_dim = d
                    if not np.isnan(sc) and sc > best_sin_corr:
                        best_sin_corr = sc
                        best_sin_dim = d

            # 正規化
            cos_neuron = pooled[:, best_cos_dim]
            sin_neuron = pooled[:, best_sin_dim]
            cos_neuron_norm = (cos_neuron - cos_neuron.mean()) / (cos_neuron.std() + 1e-8)
            sin_neuron_norm = (sin_neuron - sin_neuron.mean()) / (sin_neuron.std() + 1e-8)

            wave_fig = make_subplots(rows=1, cols=2, subplot_titles=[
                f"Cos次元 d{best_cos_dim} (corr={best_cos_corr:.3f})",
                f"Sin次元 d{best_sin_dim} (corr={best_sin_corr:.3f})"
            ])

            # Cos比較
            wave_fig.add_trace(go.Scatter(x=x_vals, y=cos_neuron_norm, mode="lines", name="ニューロン出力", line=dict(color="#FFD700")), row=1, col=1)
            wave_fig.add_trace(go.Scatter(x=x_vals, y=cos_basis, mode="lines", name=f"cos(2π·{freq_k}·x/{p})", line=dict(color="#4ECDC4", dash="dash")), row=1, col=1)

            # Sin比較
            wave_fig.add_trace(go.Scatter(x=x_vals, y=sin_neuron_norm, mode="lines", name="ニューロン出力", line=dict(color="#FF69B4")), row=1, col=2)
            wave_fig.add_trace(go.Scatter(x=x_vals, y=sin_basis, mode="lines", name=f"sin(2π·{freq_k}·x/{p})", line=dict(color="#4ECDC4", dash="dash")), row=1, col=2)

            wave_fig.update_layout(height=300, showlegend=True)
            st.plotly_chart(wave_fig, use_container_width=True, key="wave_comparison")

            # --- セクション4: リサージュ図形 ---
            st.subheader("4️⃣ リサージュ図形（cos vs sin ニューロン）")

            lissajous_fig = go.Figure()
            lissajous_fig.add_trace(go.Scatter(
                x=cos_neuron_norm, y=sin_neuron_norm,
                mode="markers",
                marker=dict(size=8, color=x_vals, colorscale="Rainbow", showscale=True, colorbar=dict(title="x")),
                text=[f"x={x}" for x in x_vals],
                hovertemplate="x=%{text}<br>cos_dim=%{x:.3f}<br>sin_dim=%{y:.3f}<extra></extra>"
            ))
            lissajous_fig.update_layout(
                title=f"リサージュ: d{best_cos_dim} vs d{best_sin_dim}（円形なら周波数{freq_k}を学習済み）",
                xaxis_title=f"次元 {best_cos_dim} (cos相関)",
                yaxis_title=f"次元 {best_sin_dim} (sin相関)",
                height=400,
                xaxis=dict(scaleanchor="y", scaleratio=1)
            )
            st.plotly_chart(lissajous_fig, use_container_width=True, key="lissajous")

        except Exception as e:
            st.error(f"ニューロン解析エラー: {e}")

        # --- セクション5: 加法定理検証バーチャート ---
        st.subheader("5️⃣ 加法定理の検証（バーチャート）")

        terms = ["cos(ωa)", "cos(ωb)", "sin(ωa)", "sin(ωb)", "cos·cos", "sin·sin", "LHS", "RHS"]
        values = [cos_a, cos_b, sin_a, sin_b, cos_cos, sin_sin, result_direct, result_fourier]
        colors = ["#667EEA", "#667EEA", "#F5576C", "#F5576C", "#FFD700", "#FFA500", "#4ECDC4", "#4ECDC4"]

        bar_fig = go.Figure(data=[go.Bar(x=terms, y=values, marker_color=colors, text=[f"{v:.3f}" for v in values], textposition="outside")])
        bar_fig.update_layout(
            title=f"加法定理: a={a_val}, b={b_val}, k={freq_k}",
            yaxis_title="値",
            height=350,
            yaxis=dict(range=[min(values) - 0.3, max(values) + 0.3])
        )
        st.plotly_chart(bar_fig, use_container_width=True, key="addition_theorem_bar")

    # ===== Tab 7: Attention Analysis =====
    with tab7:
        st.header("🔍 Attention Analysis")

        # 解説
        with st.expander("📚 Attention機構の役割", expanded=False):
            st.markdown("""
            **Attention機構** は、Transformerの中核であり、Grokkingにおいて重要な役割を果たします。

            ### Attentionの働き
            入力 `[a, b]` に対して、Attentionは以下を行います：

            1. **Query, Key, Value の計算**: 各トークンの埋め込みから Q, K, V を生成
            2. **Attention重み**: `softmax(Q·K^T / √d)` で計算
            3. **値の集約**: 重みで V を重み付け平均

            ### Grokkingでの役割
            - **埋め込みの掛け算**: cos(ωa)·cos(ωb) や sin(ωa)·sin(ωb) を計算
            - **情報の統合**: a と b の情報を統合
            - **フーリエ成分の混合**: 加法定理の前半部分を担当

            ### Attentionパターンの意味
            - **Attn[a→b]**: aがbの情報をどれだけ取り込むか
            - **Attn[b→a]**: bがaの情報をどれだけ取り込むか
            """)

        p = config["p"]
        n_tokens = config.get("n_tokens", 2)
        n_heads = config.get("n_heads", 4)

        try:
            # ===== 1. 全体Attentionパターンマップ =====
            st.subheader("1️⃣ 全体Attentionパターンマップ")
            st.markdown("全ての(a, b)ペアに対するAttention重みを可視化")

            # サンプリング数
            sample_step = max(1, p // 30)  # 最大30x30のグリッド

            @st.cache_data
            def compute_attention_maps(_model_id, _p, _n_tokens, _sample_step):
                """全(a,b)ペアのAttention重みを計算"""
                a_vals = list(range(0, p, _sample_step))
                b_vals = list(range(0, p, _sample_step))
                n_a, n_b = len(a_vals), len(b_vals)

                # バッチで計算
                inputs = []
                for a in a_vals:
                    for b in b_vals:
                        if _n_tokens == 2:
                            inputs.append([a, b])
                        else:
                            inputs.append([a, b, 0])

                inputs_tensor = torch.tensor(inputs)
                with torch.no_grad():
                    _, intermediates = model.forward_with_intermediates(inputs_tensor)

                attn_weights = intermediates["block_0_attn_weights"]  # (batch, heads, seq, seq)
                return attn_weights.numpy(), a_vals, b_vals

            # Attention重みを計算
            all_attn, a_vals, b_vals = compute_attention_maps(
                id(model), p, n_tokens, sample_step
            )
            n_a, n_b = len(a_vals), len(b_vals)

            # ヘッドとAttentionパターンの選択
            col_head, col_pattern = st.columns(2)
            with col_head:
                head_select = st.selectbox(
                    "ヘッド選択",
                    ["全ヘッド平均"] + [f"Head {i}" for i in range(all_attn.shape[1])],
                    key="attn_map_head"
                )
            with col_pattern:
                pattern_select = st.selectbox(
                    "Attentionパターン",
                    ["a→b (aがbを見る)", "b→a (bがaを見る)", "a→a (自己注意)", "b→b (自己注意)"],
                    key="attn_pattern"
                )

            # パターンに応じたインデックス
            pattern_map = {
                "a→b (aがbを見る)": (0, 1),
                "b→a (bがaを見る)": (1, 0),
                "a→a (自己注意)": (0, 0),
                "b→b (自己注意)": (1, 1),
            }
            qi, ki = pattern_map[pattern_select]

            # ヘッドの選択
            if head_select == "全ヘッド平均":
                attn_slice = all_attn.mean(axis=1)[:, qi, ki]
            else:
                head_idx = int(head_select.split()[-1])
                attn_slice = all_attn[:, head_idx, qi, ki]

            # 2Dマップに整形
            attn_map = attn_slice.reshape(n_a, n_b)

            fig_map = go.Figure(data=go.Heatmap(
                z=attn_map,
                x=b_vals,
                y=a_vals,
                colorscale="RdBu",
                zmid=0.5,
                colorbar=dict(title="Attention")
            ))
            fig_map.update_layout(
                title=f"{pattern_select} - {head_select}",
                xaxis_title="b",
                yaxis_title="a",
                height=500,
                width=600
            )
            st.plotly_chart(fig_map, use_container_width=True, key="attn_full_map")

            # ===== 2. ヘッド別Attentionパターン比較 =====
            st.subheader("2️⃣ 全ヘッドのAttentionパターン比較")
            st.markdown("各ヘッドがどのような役割を持っているかを比較")

            n_heads_actual = all_attn.shape[1]
            cols = st.columns(n_heads_actual)

            for h in range(n_heads_actual):
                with cols[h]:
                    # a→b パターンを表示
                    attn_h = all_attn[:, h, 0, 1].reshape(n_a, n_b)
                    fig_h = go.Figure(data=go.Heatmap(
                        z=attn_h,
                        colorscale="Viridis",
                        showscale=False
                    ))
                    fig_h.update_layout(
                        title=f"Head {h}",
                        height=200,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis=dict(showticklabels=False, title="b"),
                        yaxis=dict(showticklabels=False, title="a")
                    )
                    st.plotly_chart(fig_h, use_container_width=True, key=f"attn_head_map_{h}")

                    # 統計情報
                    mean_val = attn_h.mean()
                    std_val = attn_h.std()
                    st.caption(f"mean={mean_val:.3f}, std={std_val:.3f}")

            # ===== 3. Attention重み分布 =====
            st.subheader("3️⃣ Attention重み分布")

            col1, col2 = st.columns(2)

            with col1:
                # 全サンプルの分布
                fig_hist = go.Figure()
                for h in range(n_heads_actual):
                    attn_flat = all_attn[:, h, 0, 1].flatten()
                    fig_hist.add_trace(go.Histogram(
                        x=attn_flat,
                        name=f"Head {h}",
                        opacity=0.6,
                        nbinsx=30
                    ))
                fig_hist.update_layout(
                    title="Attention[a→b]の分布（全サンプル）",
                    xaxis_title="Attention Weight",
                    yaxis_title="Count",
                    barmode="overlay",
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True, key="attn_hist")

            with col2:
                # ヘッド間の相関
                head_patterns = []
                for h in range(n_heads_actual):
                    head_patterns.append(all_attn[:, h, 0, 1].flatten())
                head_patterns = np.array(head_patterns)

                corr_matrix = np.corrcoef(head_patterns)
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=[f"H{i}" for i in range(n_heads_actual)],
                    y=[f"H{i}" for i in range(n_heads_actual)],
                    colorscale="RdBu",
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
                    texttemplate="%{text}"
                ))
                fig_corr.update_layout(
                    title="ヘッド間の相関",
                    height=350
                )
                st.plotly_chart(fig_corr, use_container_width=True, key="head_corr")

            # ===== 4. 個別サンプル分析 =====
            st.subheader("4️⃣ 個別サンプル分析")

            col_a, col_b = st.columns(2)
            with col_a:
                a_att = st.slider("a", 0, p - 1, 10, key="attn_a")
            with col_b:
                b_att = st.slider("b", 0, p - 1, 20, key="attn_b")

            answer = (a_att + b_att) % p
            st.info(f"入力: ({a_att}, {b_att}) → 正解: ({a_att} + {b_att}) mod {p} = {answer}")

            # 単一サンプルのAttention
            if n_tokens == 2:
                test_input = torch.tensor([[a_att, b_att]])
            else:
                test_input = torch.tensor([[a_att, b_att, 0]])

            with torch.no_grad():
                logits, intermediates = model.forward_with_intermediates(test_input)

            pred = logits.argmax(dim=-1).item()
            pred_correct = "✅" if pred == answer else "❌"
            st.success(f"予測: {pred} {pred_correct}")

            attn_weights = intermediates["block_0_attn_weights"]
            n_heads_single = attn_weights.shape[1]

            # 全ヘッドを横に並べて表示
            cols = st.columns(n_heads_single + 1)

            if n_tokens == 2:
                labels = ["a", "b"]
            else:
                labels = ["a", "b", "="]

            for h in range(n_heads_single):
                with cols[h]:
                    attn_h = attn_weights[0, h].numpy()
                    if attn_h.ndim == 1:
                        s = int(np.sqrt(len(attn_h)))
                        attn_h = attn_h.reshape(s, s)

                    text_h = [[f"{attn_h[i, j]:.2f}" for j in range(attn_h.shape[1])]
                             for i in range(attn_h.shape[0])]

                    fig_h = go.Figure(data=go.Heatmap(
                        z=attn_h.tolist(),
                        x=labels,
                        y=labels,
                        colorscale="Viridis",
                        text=text_h,
                        texttemplate="%{text}",
                        showscale=False
                    ))
                    fig_h.update_layout(
                        title=f"Head {h}",
                        height=250,
                        margin=dict(l=30, r=10, t=40, b=30),
                    )
                    st.plotly_chart(fig_h, use_container_width=True, key=f"single_head_{h}")

            # 平均Attention
            with cols[-1]:
                avg_attn = attn_weights[0].mean(dim=0).numpy()
                if avg_attn.ndim == 1:
                    s = int(np.sqrt(len(avg_attn)))
                    avg_attn = avg_attn.reshape(s, s)

                avg_text = [[f"{avg_attn[i, j]:.2f}" for j in range(avg_attn.shape[1])]
                           for i in range(avg_attn.shape[0])]

                fig_avg = go.Figure(data=go.Heatmap(
                    z=avg_attn.tolist(),
                    x=labels,
                    y=labels,
                    colorscale="Viridis",
                    text=avg_text,
                    texttemplate="%{text}",
                    showscale=False
                ))
                fig_avg.update_layout(
                    title="平均",
                    height=250,
                    margin=dict(l=30, r=10, t=40, b=30),
                )
                st.plotly_chart(fig_avg, use_container_width=True, key="single_avg")

            # ===== 5. aまたはbを固定した時のAttention変化 =====
            st.subheader("5️⃣ 入力値によるAttention変化")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**aを固定、bを変化**")
                fixed_a = st.slider("固定するaの値", 0, p - 1, 0, key="fixed_a")

                # bを変化させてAttentionを計算
                b_range = list(range(0, p, max(1, p // 50)))
                attn_by_b = []

                inputs_b = torch.tensor([[fixed_a, b] for b in b_range])
                with torch.no_grad():
                    _, inter_b = model.forward_with_intermediates(inputs_b)
                attn_b = inter_b["block_0_attn_weights"].numpy()

                fig_by_b = go.Figure()
                for h in range(attn_b.shape[1]):
                    fig_by_b.add_trace(go.Scatter(
                        x=b_range,
                        y=attn_b[:, h, 0, 1],
                        name=f"Head {h}",
                        mode="lines+markers"
                    ))
                fig_by_b.update_layout(
                    title=f"a={fixed_a}固定、bを変化させた時のAttn[a→b]",
                    xaxis_title="b",
                    yaxis_title="Attention[a→b]",
                    height=300
                )
                st.plotly_chart(fig_by_b, use_container_width=True, key="attn_by_b")

            with col2:
                st.markdown("**bを固定、aを変化**")
                fixed_b = st.slider("固定するbの値", 0, p - 1, 0, key="fixed_b")

                # aを変化させてAttentionを計算
                a_range = list(range(0, p, max(1, p // 50)))

                inputs_a = torch.tensor([[a, fixed_b] for a in a_range])
                with torch.no_grad():
                    _, inter_a = model.forward_with_intermediates(inputs_a)
                attn_a = inter_a["block_0_attn_weights"].numpy()

                fig_by_a = go.Figure()
                for h in range(attn_a.shape[1]):
                    fig_by_a.add_trace(go.Scatter(
                        x=a_range,
                        y=attn_a[:, h, 0, 1],
                        name=f"Head {h}",
                        mode="lines+markers"
                    ))
                fig_by_a.update_layout(
                    title=f"b={fixed_b}固定、aを変化させた時のAttn[a→b]",
                    xaxis_title="a",
                    yaxis_title="Attention[a→b]",
                    height=300
                )
                st.plotly_chart(fig_by_a, use_container_width=True, key="attn_by_a")

            # ===== 6. 埋め込みの可視化 =====
            st.subheader("6️⃣ 入力埋め込み")
            embed_key = "embed"
            if embed_key in intermediates:
                embeddings = intermediates[embed_key][0].numpy()

                n_dims_show = min(32, embeddings.shape[1])
                fig_emb = go.Figure()
                for i, label in enumerate(labels):
                    fig_emb.add_trace(go.Bar(
                        name=label,
                        x=[f"d{j}" for j in range(n_dims_show)],
                        y=embeddings[i, :n_dims_show],
                    ))
                fig_emb.update_layout(
                    title=f"入力埋め込み（最初の{n_dims_show}次元）",
                    barmode="group",
                    height=300
                )
                st.plotly_chart(fig_emb, use_container_width=True, key="embeddings_bar")

        except Exception as e:
            st.error(f"Attention解析エラー: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ===== Tab 8: Neuron Analysis =====
    with tab8:
        st.header("🧠 MLP Neuron Analysis")

        with st.expander("📚 MLPニューロンの役割", expanded=False):
            st.markdown("""
            **MLP (Multi-Layer Perceptron)** はTransformerの各ブロック内にあり、
            Grokkingにおいて**加法定理の計算**を担当します。

            ### MLPの構造
            ```
            入力 → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → 出力
            ```

            ### Grokkingでの役割
            - **cos·cos と sin·sin の掛け算**: Attention後の表現を処理
            - **引き算**: cos(ω(a+b)) = cos·cos - sin·sin を計算
            - **周波数ごとの処理**: 異なるニューロンが異なる周波数kを担当

            ### ニューロンの見方
            - **活性化パターン**: 入力(a,b)に対してどう反応するか
            - **周波数選択性**: 特定のkに対応するcos/sinパターンを持つか
            - **フーリエ相関**: 理論的なcos(2πkn/p)との相関
            """)

        p = config["p"]
        n_tokens = config.get("n_tokens", 2)
        d_model = config.get("d_model", 128)

        try:
            # MLPの重みを取得
            mlp_weights = {}
            for name, param in model.named_parameters():
                if "ff" in name:
                    mlp_weights[name] = param.detach().cpu().numpy()

            st.subheader("1️⃣ MLP重みの構造")

            # 重みのリスト表示
            weight_info = []
            for name, w in mlp_weights.items():
                weight_info.append({
                    "レイヤー": name,
                    "形状": str(w.shape),
                    "パラメータ数": w.size,
                    "平均": f"{w.mean():.4f}",
                    "標準偏差": f"{w.std():.4f}"
                })
            st.dataframe(pd.DataFrame(weight_info), use_container_width=True)

            # ===== 2. ニューロン活性化パターン =====
            st.subheader("2️⃣ ニューロン活性化パターン")
            st.markdown("各ニューロンが入力(a, b)に対してどのように活性化するか")

            # サンプル入力で活性化を計算
            @st.cache_data
            def compute_neuron_activations(_model_id, _p, _n_tokens):
                """全(a,b)ペアに対するMLP中間活性化を計算"""
                sample_step = max(1, _p // 25)
                a_vals = list(range(0, _p, sample_step))
                b_vals = list(range(0, _p, sample_step))

                inputs = []
                for a in a_vals:
                    for b in b_vals:
                        if _n_tokens == 2:
                            inputs.append([a, b])
                        else:
                            inputs.append([a, b, 0])

                inputs_tensor = torch.tensor(inputs)
                with torch.no_grad():
                    _, intermediates = model.forward_with_intermediates(inputs_tensor)

                # MLP中間出力（GELU後）を取得
                # block_0_ff_outはFFN全体の出力なので、中間層を直接取得
                ff_out = intermediates.get("block_0_ff_out", None)
                post_attn = intermediates.get("block_0_post_attn", None)

                return ff_out, post_attn, a_vals, b_vals

            ff_out, post_attn, a_vals, b_vals = compute_neuron_activations(id(model), p, n_tokens)

            if ff_out is not None:
                n_a, n_b = len(a_vals), len(b_vals)

                # ニューロン選択
                n_neurons_show = min(16, ff_out.shape[-1])
                neuron_idx = st.slider("表示するニューロン開始インデックス", 0, ff_out.shape[-1] - n_neurons_show, 0, key="neuron_start")

                # 4x4グリッドで表示
                cols_per_row = 4
                rows = (n_neurons_show + cols_per_row - 1) // cols_per_row

                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx, col in enumerate(cols):
                        n_idx = neuron_idx + row * cols_per_row + col_idx
                        if n_idx < ff_out.shape[-1] and (row * cols_per_row + col_idx) < n_neurons_show:
                            with col:
                                # 平均pooling後の活性化を取得
                                neuron_act = ff_out[:, :, n_idx].mean(dim=1).numpy()  # (batch,)
                                act_map = neuron_act.reshape(n_a, n_b)

                                fig_n = go.Figure(data=go.Heatmap(
                                    z=act_map,
                                    colorscale="RdBu",
                                    zmid=0,
                                    showscale=False
                                ))
                                fig_n.update_layout(
                                    title=f"N{n_idx}",
                                    height=150,
                                    margin=dict(l=5, r=5, t=30, b=5),
                                    xaxis=dict(showticklabels=False),
                                    yaxis=dict(showticklabels=False)
                                )
                                st.plotly_chart(fig_n, use_container_width=True, key=f"neuron_{n_idx}")

            # ===== 3. ニューロンのフーリエ相関 =====
            st.subheader("3️⃣ ニューロンのフーリエ相関")
            st.markdown("各ニューロンがどの周波数kに対応しているか")

            if ff_out is not None:
                # 各ニューロンの出力とフーリエ基底の相関を計算
                @st.cache_data
                def compute_neuron_fourier_correlation(_model_id, _p):
                    """ニューロン活性化とフーリエ基底の相関"""
                    # 単一トークン入力での活性化
                    inputs = torch.tensor([[n, 0] for n in range(_p)])
                    with torch.no_grad():
                        _, inter = model.forward_with_intermediates(inputs)

                    ff = inter.get("block_0_ff_out", None)
                    if ff is None:
                        return None, None

                    # 各ニューロンの活性化（最初のトークン位置）
                    neuron_acts = ff[:, 0, :].numpy()  # (p, n_neurons)
                    n_neurons = neuron_acts.shape[1]

                    # フーリエ相関行列
                    n_freqs = _p // 2
                    corr_matrix = np.zeros((n_neurons, n_freqs))
                    n = np.arange(_p)

                    for k in range(n_freqs):
                        cos_basis = np.cos(2 * np.pi * k * n / _p)
                        sin_basis = np.sin(2 * np.pi * k * n / _p)

                        for ni in range(n_neurons):
                            act = neuron_acts[:, ni]
                            cos_corr = abs(np.corrcoef(act, cos_basis)[0, 1]) if np.std(act) > 1e-6 else 0
                            sin_corr = abs(np.corrcoef(act, sin_basis)[0, 1]) if np.std(act) > 1e-6 else 0
                            corr_matrix[ni, k] = max(cos_corr, sin_corr) if not np.isnan(cos_corr) and not np.isnan(sin_corr) else 0

                    return corr_matrix, neuron_acts

                corr_matrix, neuron_acts = compute_neuron_fourier_correlation(id(model), p)

                if corr_matrix is not None:
                    # 相関が高いニューロンを抽出
                    max_corrs = corr_matrix.max(axis=1)
                    best_freqs = corr_matrix.argmax(axis=1)
                    top_neurons = np.argsort(max_corrs)[-20:][::-1]

                    col1, col2 = st.columns(2)

                    with col1:
                        # 相関ヒートマップ（上位ニューロンのみ）
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix[top_neurons, :min(30, p//2)],
                            x=[f"k={k}" for k in range(min(30, p//2))],
                            y=[f"N{n}" for n in top_neurons],
                            colorscale="Viridis"
                        ))
                        fig_corr.update_layout(
                            title="ニューロン×周波数 相関（上位20ニューロン）",
                            xaxis_title="周波数 k",
                            yaxis_title="ニューロン",
                            height=400
                        )
                        st.plotly_chart(fig_corr, use_container_width=True, key="neuron_fourier_corr")

                    with col2:
                        # 周波数ごとの最大相関ニューロン
                        freq_max_corr = corr_matrix.max(axis=0)[:min(30, p//2)]
                        fig_freq = go.Figure(data=go.Bar(
                            x=[f"k={k}" for k in range(len(freq_max_corr))],
                            y=freq_max_corr,
                            marker_color=["#FF5722" if c > 0.7 else "#3F51B5" for c in freq_max_corr]
                        ))
                        fig_freq.update_layout(
                            title="周波数ごとの最大フーリエ相関",
                            xaxis_title="周波数 k",
                            yaxis_title="最大相関",
                            height=400
                        )
                        st.plotly_chart(fig_freq, use_container_width=True, key="freq_max_corr")

                    # 上位ニューロンの詳細
                    st.markdown("**上位ニューロンの詳細:**")
                    neuron_detail = []
                    for ni in top_neurons[:10]:
                        neuron_detail.append({
                            "ニューロン": f"N{ni}",
                            "最大相関": f"{max_corrs[ni]:.3f}",
                            "対応周波数": f"k={best_freqs[ni]}",
                            "活性化平均": f"{neuron_acts[:, ni].mean():.3f}",
                            "活性化std": f"{neuron_acts[:, ni].std():.3f}"
                        })
                    st.dataframe(pd.DataFrame(neuron_detail), use_container_width=True)

            # ===== 4. 個別ニューロン波形 =====
            st.subheader("4️⃣ 個別ニューロン波形")

            if neuron_acts is not None:
                n_neurons_total = neuron_acts.shape[1]
                selected_neuron = st.selectbox(
                    "ニューロンを選択",
                    [f"N{i} (k={best_freqs[i]}, corr={max_corrs[i]:.3f})" for i in top_neurons[:20]],
                    key="selected_neuron"
                )
                ni = int(selected_neuron.split()[0][1:])

                col1, col2 = st.columns(2)

                with col1:
                    # ニューロン活性化波形
                    n_range = np.arange(p)
                    act = neuron_acts[:, ni]
                    act_norm = (act - act.mean()) / (act.std() + 1e-8)

                    # 対応するフーリエ基底
                    k_best = best_freqs[ni]
                    cos_basis = np.cos(2 * np.pi * k_best * n_range / p)
                    sin_basis = np.sin(2 * np.pi * k_best * n_range / p)

                    fig_wave = go.Figure()
                    fig_wave.add_trace(go.Scatter(
                        x=n_range.tolist(), y=act_norm.tolist(),
                        name=f"Neuron {ni}",
                        line=dict(color="#4CAF50", width=2)
                    ))
                    fig_wave.add_trace(go.Scatter(
                        x=n_range.tolist(), y=cos_basis.tolist(),
                        name=f"cos(2πk{k_best}n/p)",
                        line=dict(color="#2196F3", width=1, dash="dash")
                    ))
                    fig_wave.add_trace(go.Scatter(
                        x=n_range.tolist(), y=sin_basis.tolist(),
                        name=f"sin(2πk{k_best}n/p)",
                        line=dict(color="#F44336", width=1, dash="dash")
                    ))
                    fig_wave.update_layout(
                        title=f"Neuron {ni} vs Fourier k={k_best}",
                        xaxis_title="入力 n",
                        yaxis_title="活性化（正規化）",
                        height=350
                    )
                    st.plotly_chart(fig_wave, use_container_width=True, key="neuron_wave")

                with col2:
                    # 2D活性化マップ
                    if ff_out is not None:
                        n_a, n_b = len(a_vals), len(b_vals)
                        act_2d = ff_out[:, :, ni].mean(dim=1).numpy().reshape(n_a, n_b)

                        fig_2d = go.Figure(data=go.Heatmap(
                            z=act_2d,
                            x=b_vals,
                            y=a_vals,
                            colorscale="RdBu",
                            zmid=0
                        ))
                        fig_2d.update_layout(
                            title=f"Neuron {ni} 活性化マップ",
                            xaxis_title="b",
                            yaxis_title="a",
                            height=350
                        )
                        st.plotly_chart(fig_2d, use_container_width=True, key="neuron_2d")

        except Exception as e:
            st.error(f"Neuron解析エラー: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

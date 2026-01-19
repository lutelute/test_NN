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
            if os.path.exists(os.path.join(d, "best.pt")) or os.path.exists(os.path.join(d, "final.pt")):
                dirs.append(d)
    return sorted(dirs)


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
                        size=2,
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


def plot_epoch_progress(checkpoint_dir, selected_epoch, history, config):
    """エポック進捗の可視化（埋め込み空間 + 学習曲線）"""
    p = config["p"]

    # 選択されたエポックのモデルをロード
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{selected_epoch}.pt")
    if not os.path.exists(epoch_path):
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


def plot_mlp_output_matrix(model, config):
    """MLP出力行列の可視化"""
    p = config["p"]
    n_tokens = config.get("n_tokens", 2)

    if n_tokens == 2:
        matrix = np.zeros((p, p))
        with torch.no_grad():
            for a in range(p):
                x = torch.tensor([[a, b] for b in range(p)])
                preds = model(x).argmax(dim=-1).numpy()
                matrix[a, :] = preds

        expected = np.array([[(a + b) % p for b in range(p)] for a in range(p)])
        xlabel, ylabel = "b", "a"
        title = "(a+b) mod p"
    else:
        # 3トークンの場合：a, b を固定して c を変化
        matrix = np.zeros((p, p))
        with torch.no_grad():
            for ab_sum in range(p):
                a = ab_sum % p
                b = 0
                for c in range(p):
                    x = torch.tensor([[a, b, c]])
                    pred = model(x).argmax(dim=-1).item()
                    matrix[ab_sum, c] = pred

        expected = np.array([[(ab + c) % p for c in range(p)] for ab in range(p)])
        xlabel, ylabel = "c", "a+b"
        title = "(a+b+c) mod p"

    accuracy = (matrix == expected).mean() * 100

    fig = px.imshow(
        matrix,
        color_continuous_scale="HSV",
        labels=dict(x=xlabel, y=ylabel, color="Prediction"),
        title=f"MLP Output: {title} (Accuracy: {accuracy:.1f}%)"
    )
    fig.update_layout(height=500)

    return fig, accuracy


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

        selected_dir = st.selectbox(
            "Select Checkpoint Directory",
            checkpoint_dirs,
            index=0
        )

        # ファイルパス
        best_path = os.path.join(selected_dir, "best.pt")
        final_path = os.path.join(selected_dir, "final.pt")
        checkpoint_path = best_path if os.path.exists(best_path) else final_path

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Training Progress",
        "🔬 Fourier Analysis",
        "⏱️ Evolution",
        "🎯 Model Output",
        "🎬 Epoch Slider"
    ])

    with tab1:
        st.header("Training Progress")

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
        else:
            st.warning("history.json not found")

    with tab2:
        st.header("Fourier Analysis")

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

        st.subheader("Fourier Basis Comparison")
        fig_comparison = plot_fourier_basis_comparison(analyzer, dominant)
        if fig_comparison:
            st.plotly_chart(fig_comparison, use_container_width=True)

    with tab3:
        st.header("Training Evolution")

        if os.path.exists(fourier_path):
            fourier_history = load_fourier_history(fourier_path)
            fig_evolution = plot_fourier_evolution(fourier_history)
            st.plotly_chart(fig_evolution, use_container_width=True)

            # スライダーで特定エポックの状態を確認
            st.subheader("Epoch Selector")
            epoch_checkpoints = glob.glob(os.path.join(selected_dir, "epoch_*.pt"))

            if epoch_checkpoints:
                epochs_available = sorted([
                    int(os.path.basename(f).replace("epoch_", "").replace(".pt", ""))
                    for f in epoch_checkpoints
                ])

                if epochs_available:
                    selected_epoch = st.select_slider(
                        "Select Epoch",
                        options=epochs_available,
                        value=epochs_available[-1]
                    )

                    epoch_path = os.path.join(selected_dir, f"epoch_{selected_epoch}.pt")
                    if os.path.exists(epoch_path):
                        epoch_model, _, _ = load_model(epoch_path)
                        epoch_analyzer = FourierAnalyzer(epoch_model)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_circle_epoch, _ = plot_embedding_circle(epoch_analyzer)
                            st.plotly_chart(fig_circle_epoch, use_container_width=True)
                        with col2:
                            fig_spectrum_epoch, _ = plot_fourier_spectrum(epoch_analyzer)
                            st.plotly_chart(fig_spectrum_epoch, use_container_width=True)
        else:
            st.warning("fourier_history.json not found")

    with tab4:
        st.header("Model Output Analysis")

        fig_matrix, accuracy = plot_mlp_output_matrix(model, config)
        st.plotly_chart(fig_matrix, use_container_width=True)

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
        st.header("Epoch Slider - 学習進捗の可視化")

        if os.path.exists(history_path):
            history = load_history(history_path)

            # 利用可能なエポックチェックポイントを取得
            epoch_checkpoints = glob.glob(os.path.join(selected_dir, "epoch_*.pt"))
            if epoch_checkpoints:
                epochs_available = sorted([
                    int(os.path.basename(f).replace("epoch_", "").replace(".pt", ""))
                    for f in epoch_checkpoints
                ])

                if epochs_available:
                    # フーリエ基底との相関が高い次元を計算
                    @st.cache_data
                    def get_fourier_dims(_dir, _config, n_dims=10):
                        """フーリエ基底との相関が高い次元を取得"""
                        p = _config["p"]
                        n_tokens = _config.get("n_tokens", 2)

                        # 最終モデルをロード
                        best_path = os.path.join(_dir, "best.pt")
                        final_path = os.path.join(_dir, "final.pt")
                        ref_path = best_path if os.path.exists(best_path) else final_path

                        if not os.path.exists(ref_path):
                            return None, None, None

                        ref_model, _, _ = load_model(ref_path)

                        # サンプル入力
                        if n_tokens == 2:
                            inputs = torch.tensor([[a, b] for a in range(min(p, 30)) for b in range(min(p, 30))])
                        else:
                            inputs = torch.tensor([[a, b, 0] for a in range(min(p, 30)) for b in range(min(p, 30))])

                        with torch.no_grad():
                            _, intermediates = ref_model.forward_with_intermediates(inputs)
                        pooled = intermediates["pooled"].numpy()  # (batch, d_model)

                        # フーリエ基底との相関を計算
                        # 入力の和 (a+b) mod p に対するフーリエ成分
                        n_samples = pooled.shape[0]
                        batch_size = int(np.sqrt(n_samples))

                        # 各次元でフーリエ係数との相関を計算
                        fourier_scores = []
                        best_k_per_dim = []

                        for dim in range(pooled.shape[1]):
                            dim_vals = pooled[:, dim]

                            # 複数の周波数kでテスト（k=1,2,3,...,p//4）
                            best_corr = 0
                            best_k = 1
                            for k in range(1, min(p // 4, 20) + 1):
                                # フーリエ基底: cos(2πk(a+b)/p), sin(2πk(a+b)/p)
                                # 入力インデックスから (a+b) mod p を計算
                                ab_sums = []
                                for idx in range(n_samples):
                                    a = idx // batch_size
                                    b = idx % batch_size
                                    ab_sums.append((a + b) % p)
                                ab_sums = np.array(ab_sums)

                                cos_basis = np.cos(2 * np.pi * k * ab_sums / p)
                                sin_basis = np.sin(2 * np.pi * k * ab_sums / p)

                                # 相関を計算
                                if np.std(dim_vals) > 0.01:
                                    cos_corr = abs(np.corrcoef(dim_vals, cos_basis)[0, 1])
                                    sin_corr = abs(np.corrcoef(dim_vals, sin_basis)[0, 1])
                                    max_corr = max(cos_corr, sin_corr)
                                    if max_corr > best_corr:
                                        best_corr = max_corr
                                        best_k = k

                            fourier_scores.append(best_corr)
                            best_k_per_dim.append(best_k)

                        # フーリエ相関が高い次元を選択
                        top_indices = np.argsort(fourier_scores)[::-1][:n_dims]
                        top_dims = top_indices.tolist()
                        top_k = [best_k_per_dim[i] for i in top_indices]
                        top_corrs = [fourier_scores[i] for i in top_indices]

                        return top_dims, top_k, top_corrs

                    fixed_dims, fourier_k, fourier_corrs = get_fourier_dims(selected_dir, config, n_dims=10)

                    if fixed_dims:
                        # 表示用に整形
                        dim_info = ", ".join([f"d{d}(k={k})" for d, k in zip(fixed_dims[:5], fourier_k[:5])])
                        st.success(f"フーリエ検出次元（上位10）: {dim_info}...")
                    else:
                        st.warning("次元の検出に失敗しました")

                    # 事前キャッシュ: 全エポックの可視化データを読み込み
                    @st.cache_data(show_spinner="エポックデータをプリロード中...")
                    def preload_epoch_data(_dir, _config, _epochs, _fixed_dims, sample_step=1):
                        """全エポックの可視化データを事前計算"""
                        p = _config["p"]
                        n_tokens = _config.get("n_tokens", 2)

                        # 円環用: 各和値に対して複数サンプルを用意して平均を取る
                        # 各(a+b) mod p の値に対して複数のペアを選択
                        np.random.seed(42)
                        samples_per_sum = 5  # 各和値に対して5サンプル
                        all_circle_pairs = []
                        sum_labels = []  # どの和値に対応するか

                        for s in range(p):
                            # s = (a + b) mod p となるペアをsamples_per_sum個選ぶ
                            for i in range(samples_per_sum):
                                a = (s + i * 17) % p  # 17は適当な素数（分散させるため）
                                b = (s - a) % p
                                all_circle_pairs.append([a, b])
                                sum_labels.append(s)

                        sum_labels = np.array(sum_labels)

                        # 相関行列用サンプル
                        corr_pairs = [[np.random.randint(p), np.random.randint(p)] for _ in range(150)]

                        if n_tokens == 2:
                            inputs_circle = torch.tensor(all_circle_pairs)
                            inputs_corr = torch.tensor(corr_pairs)
                        else:
                            inputs_circle = torch.tensor([[a, b, 0] for a, b in all_circle_pairs])
                            inputs_corr = torch.tensor([[a, b, 0] for a, b in corr_pairs])

                        epoch_data = {}
                        sampled_epochs = _epochs[::sample_step]  # サンプリング

                        for ep in sampled_epochs:
                            ep_path = os.path.join(_dir, f"epoch_{ep}.pt")
                            if not os.path.exists(ep_path):
                                continue

                            try:
                                ep_model, _, _ = load_model(ep_path)

                                # 円環用: 各和値に対応する出力
                                with torch.no_grad():
                                    _, inter_circle = ep_model.forward_with_intermediates(inputs_circle)
                                    _, inter_corr = ep_model.forward_with_intermediates(inputs_corr)

                                pooled_all = inter_circle["pooled"].numpy()  # (p*samples_per_sum, d_model)
                                pooled_corr = inter_corr["pooled"].numpy()

                                # 各和値ごとに平均を取る
                                sum_embeddings = np.zeros((p, pooled_all.shape[1]))
                                for s in range(p):
                                    mask = sum_labels == s
                                    sum_embeddings[s] = pooled_all[mask].mean(axis=0)

                                # cos/sin ペアを検出して円環を表示
                                # 各次元がどの周波数のcos/sinに対応するかを調べる
                                best_k = 1
                                best_cos_dim = 0
                                best_sin_dim = 1
                                best_total_corr = 0

                                s_values = np.arange(p)
                                for k in range(1, min(p // 4, 20) + 1):
                                    cos_basis = np.cos(2 * np.pi * k * s_values / p)
                                    sin_basis = np.sin(2 * np.pi * k * s_values / p)

                                    # 各次元とcos/sinの相関を計算
                                    cos_corrs = []
                                    sin_corrs = []
                                    for d in range(sum_embeddings.shape[1]):
                                        dim_vals = sum_embeddings[:, d]
                                        if np.std(dim_vals) > 0.01:
                                            cc = np.corrcoef(dim_vals, cos_basis)[0, 1]
                                            sc = np.corrcoef(dim_vals, sin_basis)[0, 1]
                                            cos_corrs.append((d, cc if not np.isnan(cc) else 0))
                                            sin_corrs.append((d, sc if not np.isnan(sc) else 0))
                                        else:
                                            cos_corrs.append((d, 0))
                                            sin_corrs.append((d, 0))

                                    # 最もcosに相関が高い次元と最もsinに相関が高い次元を見つける
                                    cos_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                                    sin_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

                                    # cos次元とsin次元が異なることを確認
                                    cos_dim = cos_corrs[0][0]
                                    cos_val = abs(cos_corrs[0][1])
                                    sin_dim = sin_corrs[0][0] if sin_corrs[0][0] != cos_dim else sin_corrs[1][0]
                                    sin_val = abs(sin_corrs[0][1]) if sin_corrs[0][0] != cos_dim else abs(sin_corrs[1][1])

                                    total_corr = cos_val + sin_val
                                    if total_corr > best_total_corr:
                                        best_total_corr = total_corr
                                        best_k = k
                                        best_cos_dim = cos_dim
                                        best_sin_dim = sin_dim

                                proj_2d = sum_embeddings[:, [best_cos_dim, best_sin_dim]]

                                # 円環性を計算（角度と理論角度の相関）
                                center = proj_2d.mean(axis=0)
                                centered = proj_2d - center
                                angles = np.arctan2(centered[:, 1], centered[:, 0])
                                expected_angles = 2 * np.pi * np.arange(p) / p - np.pi

                                best_corr = 0
                                for shift in range(p):
                                    shifted_expected = np.roll(expected_angles, shift)
                                    corr = np.corrcoef(angles, shifted_expected)[0, 1]
                                    if not np.isnan(corr):
                                        best_corr = max(best_corr, abs(corr))

                                # 相関行列用データ（軽量: 200サンプル）
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
                            sample_step = max(1, len(epochs_available) // 50)
                            st.info(f"自動設定: {sample_step}エポック間隔")
                        else:
                            sample_step = st.number_input(
                                "エポック刻み",
                                min_value=1,
                                max_value=max(1, len(epochs_available) // 5),
                                value=10,
                                step=5,
                                help="何エポックごとにフレームを作成するか"
                            )

                    with st.spinner("データをプリロード中..."):
                        epoch_data, sampled_epochs = preload_epoch_data(
                            selected_dir, config, epochs_available, fixed_dims, sample_step
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
                                            size=2,
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
                        fig = make_subplots(
                            rows=grid_size + 1, cols=grid_size + 1,
                            column_widths=[0.30] + [0.10] * grid_size,
                            row_heights=[0.125] * grid_size + [0.125],
                            specs=[[{"rowspan": grid_size}] + [{}] * grid_size] +
                                  [[None] + [{}] * grid_size for _ in range(grid_size - 1)] +
                                  [[{"colspan": grid_size + 1}] + [None] * grid_size],
                            horizontal_spacing=0.01,
                            vertical_spacing=0.02
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
                                            size=2,
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
                            height=700,
                            plot_bgcolor="black",
                            paper_bgcolor="black",
                            font=dict(color="white"),
                            margin=dict(t=50, b=80, l=20, r=20),
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


if __name__ == "__main__":
    main()

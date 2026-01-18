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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Training Progress",
        "🔬 Fourier Analysis",
        "⏱️ Evolution",
        "🎯 Model Output"
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


if __name__ == "__main__":
    main()

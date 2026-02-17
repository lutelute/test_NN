#!/usr/bin/env python3
"""
Grokking Analysis Dashboard - Lightweight Version for Streamlit Cloud
No PyTorch dependency. Uses precomputed data from precompute_data.py.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

DATA_DIR = "precomputed_data"

st.set_page_config(
    page_title="Grokking Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== Data Loading =====

@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_npz(path):
    return dict(np.load(path, allow_pickle=True))


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


# ===== Helper Functions =====

def compute_fourier_correlation_from_map(attn_map, p, n_freqs=10):
    """Compute correlation of attention map with Fourier bases."""
    n_a, n_b = attn_map.shape
    a_vals = np.linspace(0, p - 1, n_a)
    b_vals = np.linspace(0, p - 1, n_b)

    correlations = []
    for k in range(1, min(n_freqs + 1, p // 2)):
        cos_b = np.cos(2 * np.pi * k * b_vals / p)
        row_corrs = []
        for row in attn_map:
            if np.std(row) > 1e-6:
                c = abs(np.corrcoef(row, cos_b)[0, 1])
                if not np.isnan(c):
                    row_corrs.append(c)
        corr_b = np.mean(row_corrs) if row_corrs else 0

        cos_a = np.cos(2 * np.pi * k * a_vals / p)
        col_corrs = []
        for col in attn_map.T:
            if np.std(col) > 1e-6:
                c = abs(np.corrcoef(col, cos_a)[0, 1])
                if not np.isnan(c):
                    col_corrs.append(c)
        corr_a = np.mean(col_corrs) if col_corrs else 0

        correlations.append((k, max(corr_a, corr_b), 'a' if corr_a > corr_b else 'b'))

    return sorted(correlations, key=lambda x: x[1], reverse=True)


# ===== Main App =====

def main():
    # Check data directory
    if not os.path.exists(DATA_DIR):
        st.error(f"Precomputed data not found in `{DATA_DIR}/`. Run `python precompute_data.py` first.")
        return

    # Load config
    config = load_json(data_path("config.json"))
    p = config["p"]
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    n_tokens = config.get("n_tokens", 2)
    d_ff = config.get("d_ff", d_model * 4)

    st.title("🧠 Grokking Analysis Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Model Info")
        st.success(f"✅ Model loaded (epoch {config.get('best_epoch', '?')})")
        st.json({
            "p": p,
            "n_tokens": n_tokens,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": config.get("n_layers", 1),
        })
        st.markdown("---")
        st.caption("Lightweight dashboard (no PyTorch)")

    # Tabs
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

    # ===== Tab 1: Training Progress =====
    with tab1:
        st.header("📈 Training Progress")

        with st.expander("📚 Grokkingとは？", expanded=False):
            st.markdown("""
            **Grokking（グロッキング）** は、ニューラルネットワークが**過学習した後に突然汎化する**現象です。

            ### 典型的な学習パターン
            1. **Phase 1: 記憶（Memorization）** - 訓練精度が急速に100%に到達、テスト精度は低いまま
            2. **Phase 2: 汎化（Generalization）** - 突然テスト精度が急上昇 ← **これがGrokking!**

            ### なぜ起こる？
            - **Weight Decay（重み減衰）** が鍵。複雑な記憶解が徐々にペナルティを受け、シンプルなフーリエ解が勝利する。
            """)

        history_path = data_path("history.json")
        if os.path.exists(history_path):
            history = load_json(history_path)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Train Accuracy", f"{history['train_acc'][-1] * 100:.1f}%")
            with col2:
                st.metric("Final Test Accuracy", f"{history['test_acc'][-1] * 100:.1f}%")
            with col3:
                st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
            with col4:
                st.metric("Total Epochs", len(history["train_loss"]))

            # Plot training curves
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Accuracy", "Loss (log scale)"),
                                horizontal_spacing=0.1)

            epochs = list(range(1, len(history["train_loss"]) + 1))
            train_acc = [a * 100 for a in history["train_acc"]]
            test_acc = [a * 100 for a in history["test_acc"]]

            fig.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Accuracy",
                                     line=dict(color="#2196F3", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=test_acc, name="Test Accuracy",
                                     line=dict(color="#F44336", width=2)), row=1, col=1)

            # Grokking detection
            for i, (tr, te) in enumerate(zip(history["train_acc"], history["test_acc"])):
                if tr > 0.99 and te > 0.9:
                    fig.add_vline(x=i + 1, line_dash="dash", line_color="green",
                                  annotation_text=f"Grokking @ {i + 1}", row=1, col=1)
                    break

            fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                                     line=dict(color="#2196F3", width=2)), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=history["test_loss"], name="Test Loss",
                                     line=dict(color="#F44336", width=2)), row=1, col=2)

            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1, range=[0, 105])
            fig.update_yaxes(title_text="Loss", type="log", row=1, col=2)
            fig.update_layout(height=400, showlegend=True, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

            # Grokking detection info
            train_acc_arr = np.array(history["train_acc"])
            test_acc_arr = np.array(history["test_acc"])
            gap = train_acc_arr - test_acc_arr
            max_gap_epoch = np.argmax(gap)
            if gap[max_gap_epoch] > 0.3:
                st.info(f"🎯 Grokking検出: エポック{max_gap_epoch}で過学習ピーク（Train-Test差={gap[max_gap_epoch]*100:.1f}%）、その後汎化")
        else:
            st.warning("history.json not found")

    # ===== Tab 2: Fourier Analysis =====
    with tab2:
        st.header("🔬 Fourier Analysis")

        with st.expander("📚 フーリエ解析とは？", expanded=False):
            st.markdown("""
            **フーリエ解析** では、モデルが学習した内部表現を周波数成分に分解して分析します。

            ### 見るべきポイント
            | グラフ | 意味 | 良い状態 |
            |--------|------|----------|
            | **Fourier Spectrum** | 埋め込みの周波数成分 | 特定のkにピークが立つ |
            | **Embedding Circle** | 埋め込みの2D射影 | きれいな円形になる |
            """)

        tab2_data = load_npz(data_path("tab2_fourier.npz"))
        tab2_meta = load_json(data_path("tab2_meta.json"))

        weights = tab2_data["weights"]  # (p, d_model)
        spectrum = tab2_data["spectrum"]  # (p,)
        proj_2d = tab2_data["proj_2d"]  # (p, 2)
        dominant = tab2_meta["dominant_frequencies"]
        circular = tab2_meta["circular"]
        verification = tab2_meta["verification"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fourier Spectrum")
            half_p = p // 2 + 1
            dominant_freqs = [f[0] for f in dominant]
            colors = ["#FF5722" if i in dominant_freqs else "#3F51B5" for i in range(half_p)]

            fig_spectrum = go.Figure()
            fig_spectrum.add_trace(go.Bar(
                x=list(range(half_p)),
                y=spectrum[:half_p].tolist(),
                marker_color=colors,
                text=[f"k={i}" if i in dominant_freqs else "" for i in range(half_p)],
                textposition="outside"
            ))
            fig_spectrum.update_layout(
                title=f"Fourier Spectrum (p={p})",
                xaxis_title="Frequency k",
                yaxis_title="Power",
                height=400
            )
            st.plotly_chart(fig_spectrum, use_container_width=True)

            st.markdown("**Dominant Frequencies:**")
            for freq, power in dominant[:5]:
                status = "✅" if power >= 0.3 else ("🟡" if power >= 0.1 else "")
                st.markdown(f"- k={freq}: power={power:.4f} {status}")

        with col2:
            st.subheader("Embedding Circle")
            fig_circle = go.Figure()

            x_line = proj_2d[:, 0].tolist() + [proj_2d[0, 0]]
            y_line = proj_2d[:, 1].tolist() + [proj_2d[0, 1]]
            fig_circle.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color="gray", width=0.5), showlegend=False
            ))
            fig_circle.add_trace(go.Scatter(
                x=proj_2d[:, 0].tolist(), y=proj_2d[:, 1].tolist(),
                mode="markers",
                marker=dict(color=list(range(p)), colorscale="HSV", size=10,
                            colorbar=dict(title="Token")),
                text=[f"Token {i}" for i in range(p)],
                hovertemplate="Token %{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
            ))
            fig_circle.update_layout(
                title=f"Embedding Circular Structure<br>(Angle Correlation: {circular['angle_correlation']:.3f})",
                xaxis_title=f"Dimension {circular['top_2_dims'][0]}",
                yaxis_title=f"Dimension {circular['top_2_dims'][1]}",
                height=500,
                xaxis=dict(scaleanchor="y", scaleratio=1)
            )
            st.plotly_chart(fig_circle, use_container_width=True)

            is_fourier = "✅" if verification["is_fourier_representation"] else "❌"
            is_circular = "✅" if circular["is_circular"] else "❌"

            st.markdown(f"""
            **Analysis Results:**
            - Fourier Representation: {is_fourier} (corr={verification['best_correlation']:.3f})
            - Circular Structure: {is_circular} (corr={circular['angle_correlation']:.3f})
            """)

            fourier_corr = verification['best_correlation']
            circular_corr = circular['angle_correlation']
            if fourier_corr >= 0.9 and circular_corr >= 0.9:
                st.success("✅ 優秀: フーリエ表現と円環構造の両方が高品質で学習されています！")
            elif fourier_corr >= 0.7 and circular_corr >= 0.5:
                st.info("🟡 良好: Grokkingが概ね成功しています")
            elif fourier_corr >= 0.7 or circular_corr >= 0.5:
                st.warning("⚠️ 部分的: 一部の指標は良好ですが、完全なGrokkingには至っていません")
            else:
                st.error("❌ 不十分: フーリエ表現がまだ学習されていません")

        st.markdown("---")

        # Interactive Fourier Learning
        st.subheader("Interactive Fourier Learning")

        with st.expander("📚 Why Fourier Representation Can Express Addition", expanded=False):
            st.markdown(r"""
### The Key: Angle Addition Formula

$$\cos\left(\frac{2\pi k(a+b)}{p}\right) = \cos\left(\frac{2\pi ka}{p}\right)\cos\left(\frac{2\pi kb}{p}\right) - \sin\left(\frac{2\pi ka}{p}\right)\sin\left(\frac{2\pi kb}{p}\right)$$

| Layer | Role |
|-------|------|
| **Embedding** | Encode each token as Fourier components |
| **MLP** | Compute products using angle addition formula |
| **Output** | Decode from Fourier space back to answer |
            """)

        n = np.arange(p)

        # Frequency selection
        st.markdown("**Select frequencies to analyze (k values):**")
        available_k = list(range(1, min(p // 2, 25) + 1))
        dominant_k = [d[0] for d in dominant[:5] if d[0] in available_k]
        default_k = dominant_k[:3] if dominant_k else [1, 2, 3]

        col_select1, col_select2 = st.columns(2)
        with col_select1:
            selected_k = st.multiselect("Compare frequencies", options=available_k, default=default_k)
        with col_select2:
            show_superposition = st.checkbox("Show superposition", value=True)
            show_learned = st.checkbox("Show learned embedding", value=True)

        if selected_k:
            st.markdown("#### Individual Frequency Components")

            fig_compare = make_subplots(rows=1, cols=len(selected_k),
                                         subplot_titles=[f"k={k}" for k in selected_k])
            for idx, k in enumerate(selected_k):
                cos_basis = np.cos(2 * np.pi * k * n / p)
                sin_basis = np.sin(2 * np.pi * k * n / p)
                fig_compare.add_trace(go.Scatter(x=n.tolist(), y=cos_basis.tolist(),
                                                  name=f"cos(2πk{k}n/p)", line=dict(color="#2196F3")),
                                       row=1, col=idx + 1)
                fig_compare.add_trace(go.Scatter(x=n.tolist(), y=sin_basis.tolist(),
                                                  name=f"sin(2πk{k}n/p)", line=dict(color="#F44336")),
                                       row=1, col=idx + 1)

            fig_compare.update_layout(height=300, showlegend=False,
                                       plot_bgcolor="black", paper_bgcolor="black",
                                       font=dict(color="white"))
            fig_compare.update_xaxes(title_text="n", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            fig_compare.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(fig_compare, use_container_width=True)

            # Superposition vs Learned Embedding
            st.markdown("#### Superposition vs Learned Embedding")

            variances = np.var(weights, axis=0)
            top_dim = np.argsort(variances)[-1]
            learned_dim = weights[:, top_dim]
            learned_norm = (learned_dim - learned_dim.mean()) / (learned_dim.std() + 1e-8)

            fig_super = go.Figure()

            if show_superposition and len(selected_k) > 0:
                superposition = np.zeros(p)
                for k in selected_k:
                    cos_basis = np.cos(2 * np.pi * k * n / p)
                    sin_basis = np.sin(2 * np.pi * k * n / p)
                    cos_corr = np.corrcoef(learned_dim, cos_basis)[0, 1]
                    sin_corr = np.corrcoef(learned_dim, sin_basis)[0, 1]
                    if not np.isnan(cos_corr):
                        superposition += cos_corr * cos_basis
                    if not np.isnan(sin_corr):
                        superposition += sin_corr * sin_basis

                if superposition.std() > 0:
                    superposition = (superposition - superposition.mean()) / superposition.std()
                fig_super.add_trace(go.Scatter(
                    x=n.tolist(), y=superposition.tolist(),
                    name=f"Superposition (k={','.join(map(str, selected_k))})",
                    line=dict(color="#4CAF50", width=2)
                ))

            if show_learned:
                fig_super.add_trace(go.Scatter(
                    x=n.tolist(), y=learned_norm.tolist(),
                    name=f"Learned (dim {top_dim})",
                    line=dict(color="#FF9800", width=2, dash="dash")
                ))

            fig_super.update_layout(
                title="Fourier Superposition vs Learned Embedding",
                xaxis_title="Token n", yaxis_title="Normalized Value",
                height=400, plot_bgcolor="black", paper_bgcolor="black",
                font=dict(color="white"), legend=dict(orientation="h", y=-0.15)
            )
            fig_super.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            fig_super.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(fig_super, use_container_width=True)

            # Correlation table
            st.markdown("#### Correlation with Fourier Bases")
            corr_data = []
            for k in selected_k:
                cos_basis = np.cos(2 * np.pi * k * n / p)
                sin_basis = np.sin(2 * np.pi * k * n / p)
                cos_corr = np.corrcoef(learned_dim, cos_basis)[0, 1]
                sin_corr = np.corrcoef(learned_dim, sin_basis)[0, 1]
                combined = np.sqrt(cos_corr ** 2 + sin_corr ** 2) if not (
                            np.isnan(cos_corr) or np.isnan(sin_corr)) else 0
                corr_data.append({
                    "k": k,
                    "cos correlation": f"{cos_corr:.3f}" if not np.isnan(cos_corr) else "N/A",
                    "sin correlation": f"{sin_corr:.3f}" if not np.isnan(sin_corr) else "N/A",
                    "combined": f"{combined:.3f}"
                })
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)

    # ===== Tab 3: Evolution =====
    with tab3:
        st.header("⏱️ Training Evolution")

        with st.expander("📚 学習進化の見方", expanded=False):
            st.markdown("""
            **Training Evolution** では、学習中にモデルの内部表現がどう変化するかを追跡します。

            1. **初期**: 点がバラバラに分布
            2. **記憶フェーズ**: 少しずつ構造が現れ始める
            3. **汎化フェーズ**: きれいな円環構造が形成される ← **Grokking完了!**
            """)

        fourier_hist_path = data_path("tab3_fourier_history.json")
        if os.path.exists(fourier_hist_path):
            fourier_history = load_json(fourier_hist_path)

            fig_evolution = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Fourier Correlation", "Angle Correlation",
                                "Spectrum Concentration", "Circularity")
            )
            epochs_fh = fourier_history["epochs"]

            fig_evolution.add_trace(go.Scatter(
                x=epochs_fh, y=fourier_history["best_correlations"],
                line=dict(color="#9C27B0", width=2), fill="tozeroy", name="Fourier Corr"
            ), row=1, col=1)
            fig_evolution.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=1)

            fig_evolution.add_trace(go.Scatter(
                x=epochs_fh, y=fourier_history["angle_correlations"],
                line=dict(color="#00BCD4", width=2), fill="tozeroy", name="Angle Corr"
            ), row=1, col=2)
            fig_evolution.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=2)

            fig_evolution.add_trace(go.Scatter(
                x=epochs_fh, y=fourier_history["spectrum_concentrations"],
                line=dict(color="#FF5722", width=2), fill="tozeroy", name="Spectrum Conc"
            ), row=2, col=1)

            fig_evolution.add_trace(go.Scatter(
                x=epochs_fh, y=fourier_history["circularities"],
                line=dict(color="#4CAF50", width=2), fill="tozeroy", name="Circularity"
            ), row=2, col=2)

            fig_evolution.update_layout(height=600, showlegend=False)
            fig_evolution.update_yaxes(range=[0, 1.05], row=1, col=1)
            fig_evolution.update_yaxes(range=[0, 1.05], row=1, col=2)
            st.plotly_chart(fig_evolution, use_container_width=True)
        else:
            st.warning("Fourier history data not found")

    # ===== Tab 4: Model Output =====
    with tab4:
        st.header("🎯 Model Output Analysis")

        with st.expander("📚 モデル出力の見方", expanded=False):
            st.markdown("""
            ### 2D Heatmap の見方
            - **横軸**: 入力 b, **縦軸**: 入力 a
            - **パターン**: 斜めの縞模様が正しい（(a+b) mod p の等高線）
            """)

        tab4_data = load_npz(data_path("tab4_output.npz"))
        pred_matrix = tab4_data["pred_matrix"]
        logit_matrix = tab4_data["logit_matrix"]
        full_logits = tab4_data["full_logits"]

        if n_tokens == 2:
            expected = np.array([[(a + b) % p for b in range(p)] for a in range(p)])
        else:
            expected = np.array([[(a + c) % p for c in range(p)] for a in range(p)])

        accuracy = (pred_matrix == expected).mean() * 100
        st.metric("Model Accuracy", f"{accuracy:.1f}%")

        col_mode, col_cmap = st.columns(2)
        with col_mode:
            view_mode = st.radio("View Mode",
                                  ["Predictions (Cyclic)", "Logits (Continuous)"],
                                  horizontal=True)
        with col_cmap:
            cyclical_cmaps = ["HSV", "Phase", "Edge", "IceFire", "Twilight"]
            selected_cmap = st.selectbox("Colormap (cyclical)", cyclical_cmaps, index=0)

        col_2d, col_3d = st.columns(2)

        if view_mode == "Predictions (Cyclic)":
            with col_2d:
                st.subheader("2D Heatmap")
                fig_2d = go.Figure(data=go.Heatmap(
                    z=pred_matrix, colorscale=selected_cmap, zmin=0, zmax=p,
                    colorbar=dict(title=f"(a+b) mod {p}")
                ))
                fig_2d.update_layout(xaxis_title="b", yaxis_title="a", height=500,
                                      plot_bgcolor="black", paper_bgcolor="black",
                                      font=dict(color="white"))
                st.plotly_chart(fig_2d, use_container_width=True)

            with col_3d:
                st.subheader("3D Surface")
                fig_3d = go.Figure(data=[go.Surface(
                    z=pred_matrix, colorscale=selected_cmap, cmin=0, cmax=p,
                    colorbar=dict(title=f"(a+b) mod {p}")
                )])
                fig_3d.update_layout(
                    scene=dict(xaxis_title="b", yaxis_title="a", zaxis_title="Prediction",
                               bgcolor="black"),
                    height=500, paper_bgcolor="black", font=dict(color="white")
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        else:
            # Logit view - interpolation using numpy
            from scipy.ndimage import zoom
            smooth_logit = zoom(logit_matrix, 2, order=3)
            x = np.linspace(0, p - 1, smooth_logit.shape[1])
            y = np.linspace(0, p - 1, smooth_logit.shape[0])

            with col_2d:
                st.subheader("2D Heatmap (Logits)")
                fig_2d = go.Figure(data=go.Heatmap(
                    z=smooth_logit, x=x, y=y, colorscale="Viridis",
                    colorbar=dict(title="Logit (confidence)")
                ))
                fig_2d.update_layout(xaxis_title="b", yaxis_title="a", height=500,
                                      plot_bgcolor="black", paper_bgcolor="black",
                                      font=dict(color="white"))
                st.plotly_chart(fig_2d, use_container_width=True)

            with col_3d:
                st.subheader("3D Surface (Wave)")
                smooth_for_3d = zoom(logit_matrix, 2, order=3)
                x3 = np.linspace(0, p - 1, smooth_for_3d.shape[0])
                y3 = np.linspace(0, p - 1, smooth_for_3d.shape[1])
                fig_3d = go.Figure(data=[go.Surface(
                    x=x3, y=y3, z=smooth_for_3d, colorscale="Viridis",
                    colorbar=dict(title="Logit (confidence)")
                )])
                fig_3d.update_layout(
                    title="3D Surface: Correct Class Logit",
                    scene=dict(xaxis_title="b", yaxis_title="a", zaxis_title="Logit",
                               bgcolor="black"),
                    height=500, paper_bgcolor="black", font=dict(color="white")
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        # Test Predictions using precomputed data
        st.subheader("Test Predictions")
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("a", min_value=0, max_value=p - 1, value=0)
            b = st.number_input("b", min_value=0, max_value=p - 1, value=0)
        with col2:
            exp = (a + b) % p
            pred = int(pred_matrix[a, b])
            is_correct = pred == exp

            st.markdown(f"""
            **Input:** ({a}, {b})

            **Prediction:** <span style='color:{"green" if is_correct else "red"};font-size:24px;'>{pred}</span>

            **Expected:** {exp}

            **Correct:** {'✅' if is_correct else '❌'}
            """, unsafe_allow_html=True)

            # Probability distribution from full_logits
            logit_idx = a * p + b
            logits_ab = full_logits[logit_idx]
            # Softmax
            logits_shifted = logits_ab - logits_ab.max()
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / exp_logits.sum()

            fig_probs = go.Figure()
            fig_probs.add_trace(go.Bar(
                x=list(range(p)), y=probs.tolist(),
                marker_color=["green" if i == exp else "blue" for i in range(p)]
            ))
            fig_probs.update_layout(title="Output Probability Distribution",
                                     xaxis_title="Class", yaxis_title="Probability", height=300)
            st.plotly_chart(fig_probs, use_container_width=True)

    # ===== Tab 5: Epoch Slider =====
    with tab5:
        st.header("🎬 Epoch Slider - 学習進捗アニメーション")

        with st.expander("📚 アニメーションの見方", expanded=False):
            st.markdown("""
            | 位置 | 内容 | 見るべきポイント |
            |:---:|:-----|:----------------|
            | **左** | 円環プロット | 円形 = フーリエ学習完了 |
            | **右上** | ニューロン相関 | 対角線以外が低い = 良い |
            | **下** | 学習曲線 | Test急上昇 = Grokking |
            """)

        anim_path = data_path("tab5_animation.npz")
        meta_path = data_path("tab5_meta.json")

        if os.path.exists(anim_path) and os.path.exists(meta_path):
            tab5_data_np = load_npz(anim_path)
            tab5_meta = load_json(meta_path)
            history = load_json(data_path("history.json"))

            proj_2d_all = tab5_data_np["proj_2d"]  # (n_epochs, p, 2)
            pooled_all = tab5_data_np["pooled"]     # (n_epochs, 100, 10)
            angle_corr_all = tab5_data_np["angle_corr"]  # (n_epochs,)
            sampled_epochs = tab5_meta["sampled_epochs"]

            proj_x_range = tab5_meta["proj_x_range"]
            proj_y_range = tab5_meta["proj_y_range"]
            pooled_ranges = tab5_meta["pooled_ranges"]

            grid_size = min(7, pooled_all.shape[2])

            # Training curves
            epochs_list = list(range(1, len(history["train_acc"]) + 1))
            train_acc = [a * 100 for a in history["train_acc"]]
            test_acc = [a * 100 for a in history["test_acc"]]

            # Build animation frames
            frames = []
            slider_steps = []

            for fi in range(len(sampled_epochs)):
                ep = sampled_epochs[fi]
                proj = proj_2d_all[fi]
                pooled = pooled_all[fi]
                angle_corr = float(angle_corr_all[fi])

                frame_traces = []

                # Embedding circle - lines
                x_line = proj[:, 0].tolist() + [proj[0, 0]]
                y_line = proj[:, 1].tolist() + [proj[0, 1]]
                frame_traces.append(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line=dict(color="rgba(128,128,128,0.3)", width=1), showlegend=False
                ))
                # Embedding circle - points
                frame_traces.append(go.Scatter(
                    x=proj[:, 0].tolist(), y=proj[:, 1].tolist(),
                    mode="markers",
                    marker=dict(color=list(range(p)), colorscale="HSV", size=6, opacity=0.9),
                    showlegend=False
                ))

                # Correlation grid
                for i in range(grid_size):
                    for j in range(grid_size):
                        frame_traces.append(go.Scatter(
                            x=pooled[:, j].tolist(), y=pooled[:, i].tolist(),
                            mode="markers",
                            marker=dict(color=list(range(len(pooled))), colorscale="Plasma",
                                        size=3, opacity=0.6),
                            showlegend=False
                        ))

                # Training curves
                frame_traces.append(go.Scatter(x=epochs_list, y=train_acc,
                                                mode="lines", line=dict(color="#2196F3", width=1.5),
                                                showlegend=False))
                frame_traces.append(go.Scatter(x=epochs_list, y=test_acc,
                                                mode="lines", line=dict(color="#F44336", width=1.5),
                                                showlegend=False))
                # Current epoch line
                frame_traces.append(go.Scatter(
                    x=[ep, ep], y=[0, 100], mode="lines",
                    line=dict(color="#FFFF00", width=2), showlegend=False
                ))

                ep_idx = min(ep - 1, len(train_acc) - 1)
                frames.append(go.Frame(
                    data=frame_traces, name=str(ep),
                    layout=go.Layout(annotations=[dict(
                        text=f"Epoch {ep} | Train: {train_acc[ep_idx]:.1f}% | Test: {test_acc[ep_idx]:.1f}% | Circle: {angle_corr:.2f}",
                        xref="paper", yref="paper", x=0.5, y=1.02, showarrow=False,
                        font=dict(size=14, color="white")
                    )])
                ))
                slider_steps.append({
                    "args": [[str(ep)], {"frame": {"duration": 100, "redraw": False},
                                          "transition": {"duration": 50, "easing": "linear"},
                                          "mode": "immediate"}],
                    "label": str(ep), "method": "animate"
                })

            # Build main figure with subplots
            fig = make_subplots(
                rows=grid_size + 1, cols=grid_size + 1,
                column_widths=[0.28] + [0.103] * grid_size,
                row_heights=[0.12] * grid_size + [0.10],
                specs=[[{"rowspan": grid_size}] + [{}] * grid_size] +
                      [[None] + [{}] * grid_size for _ in range(grid_size - 1)] +
                      [[{"colspan": grid_size + 1}] + [None] * grid_size],
                horizontal_spacing=0.008, vertical_spacing=0.015
            )

            # Initial data
            fi0 = 0
            proj0 = proj_2d_all[fi0]
            pooled0 = pooled_all[fi0]
            ep0 = sampled_epochs[fi0]

            x_line = proj0[:, 0].tolist() + [proj0[0, 0]]
            y_line = proj0[:, 1].tolist() + [proj0[0, 1]]
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                      line=dict(color="rgba(128,128,128,0.3)", width=1),
                                      showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=proj0[:, 0].tolist(), y=proj0[:, 1].tolist(),
                                      mode="markers",
                                      marker=dict(color=list(range(p)), colorscale="HSV",
                                                  size=6, opacity=0.9),
                                      showlegend=False), row=1, col=1)

            for i in range(grid_size):
                for j in range(grid_size):
                    fig.add_trace(go.Scatter(
                        x=pooled0[:, j].tolist(), y=pooled0[:, i].tolist(),
                        mode="markers",
                        marker=dict(color=list(range(len(pooled0))), colorscale="Plasma",
                                    size=3, opacity=0.6),
                        showlegend=False
                    ), row=i + 1, col=j + 2)

            fig.add_trace(go.Scatter(x=epochs_list, y=train_acc, mode="lines",
                                      line=dict(color="#2196F3", width=1.5),
                                      name="Train", showlegend=True), row=grid_size + 1, col=1)
            fig.add_trace(go.Scatter(x=epochs_list, y=test_acc, mode="lines",
                                      line=dict(color="#F44336", width=1.5),
                                      name="Test", showlegend=True), row=grid_size + 1, col=1)
            fig.add_trace(go.Scatter(x=[ep0, ep0], y=[0, 100], mode="lines",
                                      line=dict(color="#FFFF00", width=2),
                                      showlegend=False), row=grid_size + 1, col=1)

            first_angle_corr = float(angle_corr_all[0])
            ep0_idx = min(ep0 - 1, len(train_acc) - 1)

            fig.update_layout(
                height=900, plot_bgcolor="black", paper_bgcolor="black",
                font=dict(color="white"), margin=dict(t=40, b=70, l=15, r=15),
                annotations=[dict(
                    text=f"Epoch {ep0} | Train: {train_acc[ep0_idx]:.1f}% | Test: {test_acc[ep0_idx]:.1f}% | Circle: {first_angle_corr:.2f}",
                    xref="paper", yref="paper", x=0.5, y=1.02, showarrow=False,
                    font=dict(size=14, color="white")
                )],
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                updatemenus=[{
                    "type": "buttons", "showactive": False, "y": -0.15, "x": 0.05,
                    "buttons": [
                        {"label": "▶ 再生", "method": "animate", "args": [None, {
                            "frame": {"duration": 100, "redraw": False},
                            "transition": {"duration": 50, "easing": "linear"},
                            "fromcurrent": True, "mode": "immediate"
                        }]},
                        {"label": "⏸ 停止", "method": "animate",
                         "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}]}
                    ]
                }],
                sliders=[{
                    "active": 0, "steps": slider_steps, "x": 0.2, "len": 0.75, "y": -0.08,
                    "currentvalue": {"prefix": "Epoch: ", "visible": True, "xanchor": "center"},
                    "transition": {"duration": 50, "easing": "linear"}
                }]
            )

            fig.update_xaxes(showticklabels=False, showgrid=False)
            fig.update_yaxes(showticklabels=False, showgrid=False)
            fig.update_xaxes(range=proj_x_range, row=1, col=1)
            fig.update_yaxes(range=proj_y_range, row=1, col=1)

            for i in range(grid_size):
                for j in range(grid_size):
                    if j < len(pooled_ranges):
                        fig.update_xaxes(range=pooled_ranges[j], row=i + 1, col=j + 2)
                    if i < len(pooled_ranges):
                        fig.update_yaxes(range=pooled_ranges[i], row=i + 1, col=j + 2)

            last_row = grid_size + 1
            fig.update_xaxes(showticklabels=True, showgrid=True,
                              gridcolor="rgba(255,255,255,0.1)",
                              title_text="Epoch", row=last_row, col=1)
            fig.update_yaxes(showticklabels=True, showgrid=True,
                              gridcolor="rgba(255,255,255,0.1)",
                              title_text="Acc%", range=[0, 105], row=last_row, col=1)

            fig.frames = frames
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Animation data not found")

    # ===== Tab 6: Fourier Theory =====
    with tab6:
        st.header("📐 Fourier Theory")
        st.markdown(r"""
        **Grokkingの核心**: Transformerはモジュラ加算をフーリエ基底で学習します。

        $$\cos(\omega(a+b)) = \cos(\omega a)\cos(\omega b) - \sin(\omega a)\sin(\omega b)$$
        """)

        col1, col2 = st.columns([1, 3])
        with col1:
            freq_k = st.slider("周波数 k", 1, min(p // 4, 20), 8, key="theory_freq_k")

        omega = 2 * np.pi * freq_k / p

        # Section 1: Computation flow
        st.subheader("1️⃣ 計算フロー（Step by Step）")
        col_a, col_b = st.columns(2)
        with col_a:
            a_val = st.number_input("a", 0, p - 1, 15, key="theory_a")
        with col_b:
            b_val = st.number_input("b", 0, p - 1, 25, key="theory_b")

        cos_a = np.cos(omega * a_val)
        sin_a = np.sin(omega * a_val)
        cos_b = np.cos(omega * b_val)
        sin_b = np.sin(omega * b_val)
        cos_cos = cos_a * cos_b
        sin_sin = sin_a * sin_b
        result_fourier = cos_cos - sin_sin
        result_direct = np.cos(omega * (a_val + b_val))
        answer = (a_val + b_val) % p

        # Flow diagram
        flow_fig = go.Figure()
        steps = [
            {"x": 0, "text": f"入力<br>a={a_val}, b={b_val}", "color": "#667EEA"},
            {"x": 1, "text": f"埋め込み<br>cos(ωa)={cos_a:.3f}<br>sin(ωa)={sin_a:.3f}<br>cos(ωb)={cos_b:.3f}<br>sin(ωb)={sin_b:.3f}", "color": "#FFD700"},
            {"x": 2, "text": f"Attention<br>cos·cos={cos_cos:.3f}<br>sin·sin={sin_sin:.3f}", "color": "#FFA500"},
            {"x": 3, "text": f"MLP<br>cos·cos - sin·sin<br>={result_fourier:.3f}", "color": "#4ECDC4"},
            {"x": 4, "text": f"出力<br>({a_val}+{b_val}) mod {p}<br>= {answer}", "color": "#FF6B6B"},
        ]

        for i, step in enumerate(steps):
            flow_fig.add_trace(go.Scatter(
                x=[step["x"]], y=[0], mode="markers+text",
                marker=dict(size=80, color=step["color"], symbol="square"),
                text=step["text"], textposition="middle center",
                textfont=dict(size=10, color="white"), showlegend=False
            ))
            if i < len(steps) - 1:
                flow_fig.add_annotation(
                    x=step["x"] + 0.5, y=0, ax=step["x"] + 0.3, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="white"
                )

        flow_fig.update_layout(
            title=f"計算フロー: ({a_val} + {b_val}) mod {p} = {answer}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(flow_fig, use_container_width=True)
        st.success(f"✅ 加法定理の検証: cos(ω({a_val}+{b_val})) = {result_direct:.6f}, cos·cos - sin·sin = {result_fourier:.6f}, 差 = {abs(result_direct - result_fourier):.2e}")

        # Section 2: 3D Surfaces
        st.subheader("2️⃣ 3D表面: cos·cos, sin·sin, cos(x+y)")
        grid_s = min(30, p)
        X, Y = np.meshgrid(np.arange(grid_s), np.arange(grid_s))
        Z_coscos = np.cos(omega * X) * np.cos(omega * Y)
        Z_sinsin = np.sin(omega * X) * np.sin(omega * Y)
        Z_sum = np.cos(omega * (X + Y))

        surface_col1, surface_col2, surface_col3 = st.columns(3)
        with surface_col1:
            fig1 = go.Figure(data=[go.Surface(z=Z_coscos, x=X, y=Y, colorscale="YlOrBr", showscale=False)])
            fig1.update_layout(title="cos(ωx)·cos(ωy)", scene=dict(xaxis_title="x", yaxis_title="y"),
                                height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
        with surface_col2:
            fig2 = go.Figure(data=[go.Surface(z=Z_sinsin, x=X, y=Y, colorscale="Oranges", showscale=False)])
            fig2.update_layout(title="sin(ωx)·sin(ωy)", scene=dict(xaxis_title="x", yaxis_title="y"),
                                height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        with surface_col3:
            fig3 = go.Figure(data=[go.Surface(z=Z_sum, x=X, y=Y, colorscale="Teal", showscale=False)])
            fig3.update_layout(title="cos(ω(x+y)) = cos·cos - sin·sin",
                                scene=dict(xaxis_title="x", yaxis_title="y"),
                                height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # Section 5: Bar chart
        st.subheader("3️⃣ 加法定理の検証（バーチャート）")
        terms = ["cos(ωa)", "cos(ωb)", "sin(ωa)", "sin(ωb)", "cos·cos", "sin·sin", "LHS", "RHS"]
        values = [cos_a, cos_b, sin_a, sin_b, cos_cos, sin_sin, result_direct, result_fourier]
        colors = ["#667EEA", "#667EEA", "#F5576C", "#F5576C", "#FFD700", "#FFA500", "#4ECDC4", "#4ECDC4"]

        bar_fig = go.Figure(data=[go.Bar(x=terms, y=values, marker_color=colors,
                                          text=[f"{v:.3f}" for v in values], textposition="outside")])
        bar_fig.update_layout(title=f"加法定理: a={a_val}, b={b_val}, k={freq_k}",
                               yaxis_title="値", height=350,
                               yaxis=dict(range=[min(values) - 0.3, max(values) + 0.3]))
        st.plotly_chart(bar_fig, use_container_width=True)

    # ===== Tab 7: Attention Analysis =====
    with tab7:
        st.header("🔍 Attention Analysis")

        st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(78, 205, 196, 0.2)); padding: 20px; border-radius: 12px; margin-bottom: 20px;">

### 🎯 Attentionの役割：「aとbの情報を混ぜる」

$$\\cos(\\omega(a+b)) = \\underbrace{\\cos(\\omega a) \\cdot \\cos(\\omega b)}_{\\text{Attentionで混合}} - \\underbrace{\\sin(\\omega a) \\cdot \\sin(\\omega b)}_{\\text{Attentionで混合}}$$

</div>
""", unsafe_allow_html=True)

        tab7_data_np = load_npz(data_path("tab7_attention.npz"))
        all_attn = tab7_data_np["attention_maps"]  # (batch, heads, seq, seq)
        a_vals = tab7_data_np["a_vals"].tolist()
        b_vals = tab7_data_np["b_vals"].tolist()
        n_a, n_b = len(a_vals), len(b_vals)

        # Section 1: Full attention pattern map
        st.subheader("1️⃣ 全体Attentionパターンマップ")

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

        pattern_map = {
            "a→b (aがbを見る)": (0, 1),
            "b→a (bがaを見る)": (1, 0),
            "a→a (自己注意)": (0, 0),
            "b→b (自己注意)": (1, 1),
        }
        qi, ki = pattern_map[pattern_select]

        if head_select == "全ヘッド平均":
            attn_slice = all_attn.mean(axis=1)[:, qi, ki]
        else:
            head_idx = int(head_select.split()[-1])
            attn_slice = all_attn[:, head_idx, qi, ki]

        attn_map = attn_slice.reshape(n_a, n_b)

        fig_map = go.Figure(data=go.Heatmap(
            z=attn_map, x=b_vals, y=a_vals, colorscale="RdBu", zmid=0.5,
            colorbar=dict(title="Attention")
        ))
        fig_map.update_layout(
            title=f"{pattern_select} - {head_select}",
            xaxis_title="b", yaxis_title="a", height=500, width=600
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Fourier correlation analysis
        top_freqs = compute_fourier_correlation_from_map(attn_map, p, n_freqs=15)

        st.markdown("**📊 Attentionで検出された周波数（相関順）:**")
        freq_cols = st.columns(5)
        for i, (k, corr, direction) in enumerate(top_freqs[:5]):
            with freq_cols[i]:
                if corr >= 0.5:
                    status = "✅"
                elif corr >= 0.3:
                    status = "🟡"
                else:
                    status = ""
                st.metric(f"k={k}", f"{corr:.2f} {status}", f"(∝cos(ω{direction}))")

        # Section 2: Head comparison
        st.subheader("2️⃣ 全ヘッドのAttentionパターン比較")
        n_heads_actual = all_attn.shape[1]
        cols = st.columns(n_heads_actual)

        for h in range(n_heads_actual):
            with cols[h]:
                attn_h = all_attn[:, h, 0, 1].reshape(n_a, n_b)
                freqs_h = compute_fourier_correlation_from_map(attn_h, p, n_freqs=10)
                top_k_h, top_corr_h, _ = freqs_h[0] if freqs_h else (0, 0, 'b')

                fig_h = go.Figure(data=go.Heatmap(z=attn_h, colorscale="Viridis", showscale=False))
                fig_h.update_layout(
                    title=f"Head {h} (k={top_k_h})",
                    height=200, margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(showticklabels=False, title="b"),
                    yaxis=dict(showticklabels=False, title="a")
                )
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption(f"k={top_k_h}: corr={top_corr_h:.2f}")

        # Section 3: Attention weight distribution
        st.subheader("3️⃣ Attention重み分布")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = go.Figure()
            for h in range(n_heads_actual):
                attn_flat = all_attn[:, h, 0, 1].flatten()
                fig_hist.add_trace(go.Histogram(x=attn_flat, name=f"Head {h}", opacity=0.6, nbinsx=30))
            fig_hist.update_layout(title="Attention[a→b]の分布", xaxis_title="Attention Weight",
                                    yaxis_title="Count", barmode="overlay", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            head_patterns = np.array([all_attn[:, h, 0, 1].flatten() for h in range(n_heads_actual)])
            corr_matrix_h = np.corrcoef(head_patterns)
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix_h,
                x=[f"H{i}" for i in range(n_heads_actual)],
                y=[f"H{i}" for i in range(n_heads_actual)],
                colorscale="RdBu", zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr_matrix_h],
                texttemplate="%{text}"
            ))
            fig_corr.update_layout(title="ヘッド間の相関", height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

    # ===== Tab 8: Neuron Analysis =====
    with tab8:
        st.header("🧠 MLP Neuron Analysis")

        with st.expander("📚 MLPニューロンの役割", expanded=False):
            st.markdown("""
            ### MLPの構造
            ```
            入力 → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → 出力
            ```
            - **cos·cos と sin·sin の掛け算**: Attention後の表現を処理
            - **引き算**: cos(ω(a+b)) = cos·cos - sin·sin を計算
            """)

        tab8_path = data_path("tab8_neurons.npz")
        if os.path.exists(tab8_path):
            tab8_data_np = load_npz(tab8_path)
            corr_matrix = tab8_data_np["neuron_corr"]   # (n_neurons, n_freqs)
            neuron_acts = tab8_data_np["activations"]     # (p, n_neurons)

            # Section 1: Neuron activation patterns
            if "activations_2d" in tab8_data_np:
                st.subheader("1️⃣ ニューロン活性化パターン")
                activations_2d = tab8_data_np["activations_2d"]  # (batch, n_neurons)
                act_a_vals = tab8_data_np["a_vals"].tolist()
                act_b_vals = tab8_data_np["b_vals"].tolist()
                n_a_act = len(act_a_vals)
                n_b_act = len(act_b_vals)

                n_neurons_total = activations_2d.shape[1]
                n_neurons_show = min(16, n_neurons_total)
                neuron_idx = st.slider("表示するニューロン開始インデックス", 0,
                                        max(0, n_neurons_total - n_neurons_show), 0, key="neuron_start")

                cols_per_row = 4
                rows = (n_neurons_show + cols_per_row - 1) // cols_per_row
                for row_i in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx, col in enumerate(cols):
                        n_idx = neuron_idx + row_i * cols_per_row + col_idx
                        if n_idx < n_neurons_total and (row_i * cols_per_row + col_idx) < n_neurons_show:
                            with col:
                                act_map = activations_2d[:, n_idx].reshape(n_a_act, n_b_act)
                                fig_n = go.Figure(data=go.Heatmap(
                                    z=act_map, colorscale="RdBu", zmid=0, showscale=False
                                ))
                                fig_n.update_layout(
                                    title=f"N{n_idx}", height=150,
                                    margin=dict(l=5, r=5, t=30, b=5),
                                    xaxis=dict(showticklabels=False),
                                    yaxis=dict(showticklabels=False)
                                )
                                st.plotly_chart(fig_n, use_container_width=True)

            # Section 2: Fourier correlation
            st.subheader("2️⃣ ニューロンのフーリエ相関")

            max_corrs = corr_matrix.max(axis=1)
            best_freqs = corr_matrix.argmax(axis=1)
            top_neurons = np.argsort(max_corrs)[-20:][::-1]

            col1, col2 = st.columns(2)
            with col1:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix[top_neurons, :min(30, p // 2)],
                    x=[f"k={k}" for k in range(min(30, p // 2))],
                    y=[f"N{n}" for n in top_neurons],
                    colorscale="Viridis"
                ))
                fig_corr.update_layout(title="ニューロン×周波数 相関（上位20）",
                                        xaxis_title="周波数 k", yaxis_title="ニューロン", height=400)
                st.plotly_chart(fig_corr, use_container_width=True)

            with col2:
                freq_max_corr = corr_matrix.max(axis=0)[:min(30, p // 2)]
                fig_freq = go.Figure(data=go.Bar(
                    x=[f"k={k}" for k in range(len(freq_max_corr))],
                    y=freq_max_corr,
                    marker_color=["#FF5722" if c > 0.7 else "#3F51B5" for c in freq_max_corr]
                ))
                fig_freq.update_layout(title="周波数ごとの最大フーリエ相関",
                                        xaxis_title="周波数 k", yaxis_title="最大相関", height=400)
                st.plotly_chart(fig_freq, use_container_width=True)

            # Top neurons detail
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

            # Section 3: Individual neuron waveforms
            st.subheader("3️⃣ 個別ニューロン波形")

            selected_neuron = st.selectbox(
                "ニューロンを選択",
                [f"N{i} (k={best_freqs[i]}, corr={max_corrs[i]:.3f})" for i in top_neurons[:20]],
                key="selected_neuron"
            )
            ni = int(selected_neuron.split()[0][1:])

            col1, col2 = st.columns(2)
            with col1:
                n_range = np.arange(p)
                act = neuron_acts[:, ni]
                act_norm = (act - act.mean()) / (act.std() + 1e-8)
                k_best = best_freqs[ni]
                cos_basis = np.cos(2 * np.pi * k_best * n_range / p)
                sin_basis = np.sin(2 * np.pi * k_best * n_range / p)

                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(x=n_range.tolist(), y=act_norm.tolist(),
                                               name=f"Neuron {ni}", line=dict(color="#4CAF50", width=2)))
                fig_wave.add_trace(go.Scatter(x=n_range.tolist(), y=cos_basis.tolist(),
                                               name=f"cos(2πk{k_best}n/p)",
                                               line=dict(color="#2196F3", width=1, dash="dash")))
                fig_wave.add_trace(go.Scatter(x=n_range.tolist(), y=sin_basis.tolist(),
                                               name=f"sin(2πk{k_best}n/p)",
                                               line=dict(color="#F44336", width=1, dash="dash")))
                fig_wave.update_layout(title=f"Neuron {ni} vs Fourier k={k_best}",
                                        xaxis_title="入力 n", yaxis_title="活性化（正規化）", height=350)
                st.plotly_chart(fig_wave, use_container_width=True)

            with col2:
                if "activations_2d" in tab8_data_np:
                    act_2d = activations_2d[:, ni].reshape(n_a_act, n_b_act)
                    fig_2d = go.Figure(data=go.Heatmap(
                        z=act_2d, x=act_b_vals, y=act_a_vals, colorscale="RdBu", zmid=0
                    ))
                    fig_2d.update_layout(title=f"Neuron {ni} 活性化マップ",
                                          xaxis_title="b", yaxis_title="a", height=350)
                    st.plotly_chart(fig_2d, use_container_width=True)
        else:
            st.warning("Neuron data not found")


if __name__ == "__main__":
    main()

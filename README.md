# Grokking: Neural Network Generalization Study

Transformerモデルを使った**Grokking現象**（遅延汎化）の研究プロジェクト。

## 概要

Grokkingとは、ニューラルネットワークが訓練データを完全に暗記した**後**に、突然テストデータへの汎化が起こる現象です。本プロジェクトでは、モジュラー加算タスクを用いてこの現象を再現・解析します。

### タスク
- **2トークン版**: `(a + b) mod p` を予測
- **3トークン版**: `(a + b + c) mod p` を予測

### 主な発見
- モデルは**フーリエ表現**を学習する（埋め込みが円周構造を形成）
- 強い正則化（Weight Decay）がGrokkingを促進
- 暗記 → Grokking の遷移でフーリエ相関が急上昇

## プロジェクト構成

```
test_NN/
├── train_with_analysis.py   # 学習スクリプト（フーリエ解析付き）
├── model.py                 # Transformerモデル定義
├── data.py                  # データセット生成
├── analyze.py               # フーリエ解析ツール
├── visualize.py             # 可視化スクリプト
├── visualize_presentation.py # プレゼン用高品質可視化
├── interactive_dashboard.py # Streamlitダッシュボード
├── monitor_complete.py      # リアルタイム学習モニター
├── checkpoints_*/           # 学習済みモデル
└── figures_*/               # 可視化結果
```

## セットアップ

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly plotext
```

## 使い方

### 学習

```bash
# 2トークン版（高速、推奨）
python train_with_analysis.py --p 67 --n_tokens 2 --epochs 10000 --checkpoint_dir checkpoints_2token

# 3トークン版（大規模）
python train_with_analysis.py --p 67 --n_tokens 3 --epochs 15000 --checkpoint_dir checkpoints_3token
```

### リアルタイムモニター

別ターミナルで実行（絶対パス必須）:
```bash
source /path/to/venv/bin/activate && python /path/to/monitor_complete.py
```

### 可視化

```bash
# プレゼン用図を生成
python visualize_presentation.py --checkpoint_dir checkpoints_2token --output_dir figures

# インタラクティブダッシュボード
streamlit run interactive_dashboard.py
```

## 結果例

### 2トークン P=67
- 暗記完了: Epoch 83
- Grokking発生: Epoch 221
- 最終精度: Train 100%, Test 100%

### モデル構成
- `d_model`: 128
- `n_heads`: 4
- `n_layers`: 1
- `weight_decay`: 1.0

## 参考文献

- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- [A Mechanistic Interpretability Analysis of Grokking](https://arxiv.org/abs/2301.05217)

# Grokking 現象の可視化 - テクニカルノート

## 概要

このプロジェクトは、ニューラルネットワークにおける **Grokking現象**（突然の汎化）を可視化し、内部表現の変化を観察するためのツールです。

## Grokking現象とは

Grokking（グロッキング）は、Power et al. (2022) "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" で報告された現象です。

- **訓練初期**: モデルは訓練データを完全に記憶（Train Acc: 100%）
- **長期訓練後**: 突然、テストデータにも汎化（Test Acc: 0% → 100%）
- **特徴**: 過学習状態から突然汎化が起こる

```
Train Acc: ████████████████████ 100% (早期に到達)
Test Acc:  ░░░░░░░░░░████████████ 100% (数百エポック後に突然上昇)
                     ↑
                  Grokking!
```

## タスク: モジュラー加算

```
入力: (a, b) where a, b ∈ {0, 1, ..., p-1}
出力: (a + b) mod p
```

- p = 97（素数）
- 訓練データ: 全ペアの30%
- テストデータ: 残り70%

## フーリエ表現理論

Grokking後、モデルは内部的に**フーリエ表現**を学習します。

### 理論的背景

モジュラー加算を解くために、モデルは数値 n を以下のように埋め込みます：

```
e(n) ≈ [cos(2πkn/p), sin(2πkn/p)]  (周波数 k のフーリエ基底)
```

この表現を使うと、加算が以下のように計算できます：

```
e(a) + e(b) の角度 = 2πk(a+b)/p mod 2π
```

### 円環構造

フーリエ表現が学習されると、和の値 `s = (a+b) mod p` に対する内部表現は**円環上**に配置されます：

```
       s=24
      ·   ·  s=0
    ·       ·
  s=48       s=72
    ·       ·
      ·   ·
       s=96
```

## 可視化ダッシュボード

### 起動方法

```bash
source venv/bin/activate
streamlit run interactive_dashboard.py --server.port 8502
```

### 主要機能

#### 1. Epoch Slider タブ

学習過程をアニメーションで観察：

- **左パネル**: 円環構造（(a+b) mod p の表現）
- **右パネル**: 7×7 ニューロン相関行列
- **下パネル**: 学習曲線 + 現在位置

#### 2. 円環構造の検出

各エポックで以下を計算：

1. 全ての和値 s ∈ {0, ..., p-1} に対するpooled層出力を取得
2. cos(2πks/p) と sin(2πks/p) に最も相関する次元を検出
3. その2次元でプロット → 円環が形成されれば Fourier 表現を学習

**Circle値**（角度相関）:
- < 0.5: 円環未形成（記憶段階）
- \> 0.9: 円環形成（Fourier表現学習済み）

#### 3. ニューロン相関行列

上位7次元のニューロン出力の散布図グリッド：
- 対角: 各次元の分布
- 非対角: 次元間の相関

Grokking前後で構造が劇的に変化します。

## 実装詳細

### cos/sin ペア検出アルゴリズム

```python
for k in range(1, 20):
    cos_basis = np.cos(2 * np.pi * k * s_values / p)
    sin_basis = np.sin(2 * np.pi * k * s_values / p)

    # 各次元との相関を計算
    for d in range(d_model):
        cos_corr[d] = corrcoef(embeddings[:, d], cos_basis)
        sin_corr[d] = corrcoef(embeddings[:, d], sin_basis)

    # 最良のcos次元とsin次元を選択（異なる次元）
    best_cos_dim = argmax(|cos_corr|)
    best_sin_dim = argmax(|sin_corr|, exclude=best_cos_dim)
```

### データ効率化

大量のチェックポイントを扱うため：

1. **サンプリング**: 各和値に5サンプル（p×5 = 485点）
2. **平均化**: 同じ和値のサンプルを平均
3. **フレーム間引き**: 最大50フレームに制限

## チェックポイント

| ディレクトリ | 刻み | ファイル数 | 用途 |
|-------------|------|-----------|------|
| `checkpoints_demo` | 10 epoch | 500 | 概観 |
| `checkpoints_demo_2ep` | 2 epoch | 2500 | 詳細観察 |
| `checkpoints_demo_5ep` | 5 epoch | 1000 | バランス |

## 学習パラメータ

```python
p = 97              # 素数
d_model = 128       # 埋め込み次元
n_heads = 4         # アテンションヘッド
n_layers = 1        # Transformer層数
lr = 1e-3           # 学習率
weight_decay = 1.0  # 重み減衰（重要！）
train_ratio = 0.3   # 訓練データ比率
```

**Note**: `weight_decay=1.0` が Grokking に重要。小さいと汎化しない。

## 観察されるタイムライン

```
Epoch     0-50:   ランダム状態、Circle ≈ 0.1
Epoch   50-150:   訓練データ記憶、Train Acc → 100%
Epoch  150-200:   内部表現の再構成開始
Epoch  200-250:   Grokking! Test Acc 急上昇、Circle → 0.9+
Epoch  250+:      安定した Fourier 表現
```

## 参考文献

1. Power, A., et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
2. Neel Nanda et al. (2023). "Progress measures for grokking via mechanistic interpretability"

## ファイル構成

```
test_NN/
├── model.py                 # Transformerモデル定義
├── train_grokking_demo.py   # 学習スクリプト
├── interactive_dashboard.py # 可視化ダッシュボード
├── analyze.py               # フーリエ解析
├── checkpoints_demo/        # 10epoch刻み
├── checkpoints_demo_2ep/    # 2epoch刻み
└── checkpoints_demo_5ep/    # 5epoch刻み
```

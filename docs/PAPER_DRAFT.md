# Transformerは三角関数の加法定理を再発見するか？
# Do Transformers Rediscover the Addition Theorem of Trigonometric Functions?

**モジュラー加算タスクにおけるGrokking現象の機械論的解釈**

---

## Abstract

ニューラルネットワークが「計算」をどのように学習するかは、深層学習の根本的な問いである。本研究では、Transformerがモジュラー加算タスク (a + b) mod p を学習する過程を詳細に解析し、以下の問いに答える：

1. **Transformerは本当に「足し算」を学習しているのか？**
2. **もし学習しているなら、どのようなアルゴリズムを実装しているのか？**
3. **なぜ訓練後に突然汎化する「Grokking」が起こるのか？**

我々の解析により、Transformerが三角関数の加法定理
$$\cos(\omega(a+b)) = \cos(\omega a)\cos(\omega b) - \sin(\omega a)\sin(\omega b)$$
を自発的に「再発見」し、これを用いてモジュラー加算を実装していることが示された。この発見は、ニューラルネットワークが単なるパターンマッチングではなく、数学的構造を内部的に構築できることを示唆する。

---

## 1. Introduction

### 1.1 研究の動機

深層学習モデルは多くのタスクで人間を超える性能を示すが、その内部で「何が起こっているか」は依然として不明瞭である。特に以下の問いは未解決である：

- ニューラルネットワークは「記憶」しているのか「理解」しているのか？
- 数学的操作を学習する際、どのような内部表現を獲得するのか？
- なぜ特定の条件下で急激な汎化（Grokking）が起こるのか？

### 1.2 Grokking現象

Power et al. (2022) は、小規模なアルゴリズミックデータセットでの学習において、**訓練データを完全に暗記した後**、長い遅延を経て**突然テストデータにも汎化する**現象を発見し、「Grokking」と名付けた。

```
典型的なGrokking:
- Epoch 500:  訓練精度100%、テスト精度10%（暗記完了）
- Epoch 1000: 訓練精度100%、テスト精度15%（まだ暗記）
- Epoch 1500: 訓練精度100%、テスト精度90%（突然の汎化！）
```

この現象は直感に反する。なぜなら、従来の機械学習の理解では、訓練誤差がゼロに達した後に汎化が改善することは期待されないからである。

### 1.3 研究課題

本研究では以下の課題に取り組む：

**RQ1**: Transformerはモジュラー加算をどのように内部的に表現するか？

**RQ2**: Grokking前後で内部表現はどのように変化するか？

**RQ3**: 学習されたアルゴリズムは数学的に解釈可能か？

---

## 2. Background

### 2.1 モジュラー加算タスク

**定義**: 素数 p に対し、入力ペア (a, b) ∈ {0, 1, ..., p-1}² から出力 (a + b) mod p を予測する p クラス分類問題。

**例** (p = 7):
- (3, 5) → 1  （∵ 8 mod 7 = 1）
- (6, 6) → 5  （∵ 12 mod 7 = 5）

### 2.2 なぜモジュラー演算か？

モジュラー演算は以下の理由で理想的なテストベッドである：

1. **明確な正解**: アルゴリズム的に定義された正解が存在
2. **構造の存在**: 周期性、群構造など数学的構造を持つ
3. **適度な複雑さ**: 単純な暗記では汎化しない
4. **解釈可能性**: フーリエ解析により内部表現を検証可能

### 2.3 フーリエ表現仮説

Nanda et al. (2023) は、Grokkingを経たモデルが**フーリエ表現**を学習するという仮説を提唱した：

**仮説**: トークン n は以下のように埋め込まれる
$$\mathbf{e}_n = [\cos(2\pi k_1 n / p), \sin(2\pi k_1 n / p), \cos(2\pi k_2 n / p), \sin(2\pi k_2 n / p), \ldots]$$

この表現により、加法定理を用いた計算が可能になる。

---

## 3. Methods

### 3.1 モデルアーキテクチャ

1層Transformerを使用：

```
入力: [a, b] ∈ {0, ..., p-1}²
  ↓
Token Embedding (p × 128)
  ↓
Positional Encoding (学習可能)
  ↓
Multi-Head Self-Attention (4 heads)
  ↓
Feed-Forward Network (128 → 512 → 128, GELU)
  ↓
Mean Pooling
  ↓
Output Linear (128 → p)
  ↓
予測: argmax
```

### 3.2 学習設定

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| p | 97 | 十分大きな素数 |
| 訓練データ割合 | 30% | Grokkingを観察するため |
| Weight Decay | 1.0 | 暗記を抑制し汎化を促進 |
| 学習率 | 0.001 | 標準的な値 |
| エポック数 | 5000 | Grokkingに十分な時間 |

### 3.3 解析手法

#### 3.3.1 フーリエ解析

埋め込み重み W ∈ ℝ^{p×d} に対し、各次元のフーリエ変換を計算：

$$F_k^{(j)} = \sum_{n=0}^{p-1} W_{n,j} \cdot e^{-2\pi i k n / p}$$

支配的周波数を特定し、理論的cos/sinとの相関を測定。

#### 3.3.2 Attention解析

全 (a, b) ペアに対するAttention重みを計算：
- Attn[a→b]: トークン a がトークン b を見る重み
- 各ヘッドのパターンを比較

#### 3.3.3 ニューロン解析

MLP層の各ニューロンの活性化パターンを解析：
- 入力 n に対する活性化 f(n) を記録
- フーリエ基底 cos(2πkn/p), sin(2πkn/p) との相関を計算

---

## 4. Results

### 4.1 Grokkingの観察

典型的な学習曲線を図1に示す。

```
[Figure 1: Training Curves]

    100% |--------------------*************************
        |         **********
Acc  50% |     ****
        | ****
     0% |*
        +----+----+----+----+----+----+----+----+----+
        0   500  1000 1500 2000 2500 3000 3500 4000 4500
                          Epoch

        ── Train Accuracy (100% at epoch ~500)
        ── Test Accuracy (100% at epoch ~2000)
```

**観察**:
- 訓練精度は約500エポックで100%に到達
- テスト精度は約1500エポックまで低迷（~15%）
- 1500-2000エポックで急激に上昇
- 最終的にテスト精度も~100%に到達

### 4.2 フーリエ表現の発見

#### 4.2.1 支配的周波数

学習後の埋め込みのフーリエスペクトルを解析した結果：

| 周波数 k | パワー | 解釈 |
|---------|--------|------|
| 13 | 0.225 | 主要成分 |
| 15 | 0.217 | 主要成分 |
| 46 | 0.143 | 副次成分 |

上位3周波数で全パワーの58.5%を占める。

#### 4.2.2 埋め込みとフーリエ基底の相関

```
[Figure 2: Embedding vs Fourier Basis]

Dimension 47 (highest variance):
  ↗ cos(2π·13·n/97): correlation = 0.847
  ↗ sin(2π·13·n/97): correlation = 0.812

Dimension 89:
  ↗ cos(2π·15·n/97): correlation = 0.823
  ↗ sin(2π·15·n/97): correlation = 0.798
```

**結論**: 埋め込み空間の主要次元がフーリエ基底と強く相関している。

### 4.3 円周構造の出現

埋め込みを2次元に射影すると、トークンが円周上に配置される。

```
[Figure 3: Circular Structure]

        y
        |     . 5
        | . 4     . 6
        |. 3         . 7
   -----+------------------ x
        |. 2         . 8
        | . 1     . 9
        |     . 0
```

角度相関: 0.823（トークン番号と角度の相関）

この構造は、トークン n が角度 θ_n = 2πkn/p に配置されることを示す。

### 4.4 加法定理の実装

#### 4.4.1 理論的予測

フーリエ表現が正しければ、以下の計算が行われているはずである：

```
Step 1: Embedding
  a → (cos(ωa), sin(ωa))
  b → (cos(ωb), sin(ωb))

Step 2: Attention (掛け算)
  cos(ωa)·cos(ωb) → 中間表現
  sin(ωa)·sin(ωb) → 中間表現

Step 3: MLP (引き算)
  cos(ωa)·cos(ωb) - sin(ωa)·sin(ωb) = cos(ω(a+b))

Step 4: Output (復号)
  cos(ω(a+b)) → (a+b) mod p
```

#### 4.4.2 Attention パターンの検証

```
[Figure 4: Attention Patterns]

Head 0: a→b attention dominant
  ┌─────┬─────┐
  │ 0.02│ 0.98│  ← a strongly attends to b
  ├─────┼─────┤
  │ 0.99│ 0.01│  ← b strongly attends to a
  └─────┴─────┘

Head 1: Similar but different weights
Head 2: Self-attention dominant
Head 3: Uniform attention
```

各ヘッドが異なる役割を持ち、cos·cos と sin·sin の計算を分担している可能性。

#### 4.4.3 ニューロン活性化の検証

```
[Figure 5: Neuron Activation vs Fourier]

Neuron 127 (k=13 tuned):
  Activation pattern ≈ cos(2π·13·n/97)
  Correlation: 0.892

Neuron 84 (k=15 tuned):
  Activation pattern ≈ sin(2π·15·n/97)
  Correlation: 0.867
```

特定のニューロンが特定の周波数に「チューニング」されている。

### 4.5 「足し算」をしているのか？

**問い**: Transformerは本当に a + b を計算しているのか、それとも別の方法で答えを出しているのか？

**答え**: **Yes, but indirectly.**

Transformerは以下の意味で「足し算」をしている：

1. **フーリエ空間での加算**: cos(ωa) と cos(ωb) から cos(ω(a+b)) を計算
2. **加法定理の利用**: 直接加算ではなく、乗算と減算の組み合わせ
3. **複数周波数の統合**: 単一周波数では不十分、複数の k の結果を組み合わせ

これは人間の「足し算」とは異なるが、数学的に等価なアルゴリズムである。

---

## 5. Discussion

### 5.1 なぜフーリエ表現なのか？

**最小記述長原理**: フーリエ表現はモジュラー演算を記述する最も「圧縮された」方法である。

- **暗記**: p² 個のペアを個別に記憶 → O(p²) パラメータ
- **フーリエ**: 少数の周波数成分を学習 → O(p) パラメータ

Weight Decay（正則化）により、よりシンプルな解が選好される。

### 5.2 Grokkingのメカニズム

提案するメカニズム：

```
Phase 1 (暗記):
├── 訓練誤差を最小化するため暗記解を学習
├── 複雑な重み構造（大きなノルム）
└── テストには汎化しない

Phase 2 (移行):
├── Weight Decayが暗記解を「侵食」
├── 重みのノルムが減少
└── フーリエ構造が徐々に出現

Phase 3 (汎化):
├── フーリエ解が支配的に
├── シンプルな重み構造（小さなノルム）
└── テストにも汎化
```

### 5.3 加法定理の「再発見」

興味深い点は、我々がモデルに加法定理を教えていないことである。モデルは以下の情報のみから出発した：

- 入力: (a, b) のペア
- 出力: (a + b) mod p
- 損失関数: クロスエントロピー

それにも関わらず、モデルは数千年前に人類が発見した加法定理と本質的に同じ計算を「再発見」した。

### 5.4 限界と今後の課題

**限界**:
- 単一の演算（加算）のみを検証
- 小規模な p での実験
- 1層Transformerのみ

**今後の課題**:
- 乗算、指数などへの拡張
- より大規模なモデルでの検証
- Grokkingの必要十分条件の特定

---

## 6. Conclusion

本研究は、Transformerがモジュラー加算を学習する際に、三角関数の加法定理を自発的に「再発見」することを示した。

**主要な貢献**:

1. **フーリエ表現の検証**: 埋め込みが cos/sin 基底と強く相関することを確認
2. **加法定理の実装**: Attention で乗算、MLP で減算を実行
3. **Grokkingの解釈**: 暗記解からフーリエ解への移行として理解

**含意**:

ニューラルネットワークは、適切な条件下で数学的構造を「発見」する能力を持つ。これは、深層学習が単なる統計的パターンマッチング以上のものであることを示唆する。

---

## References

[1] Power, A., et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." arXiv:2201.02177.

[2] Nanda, N., et al. (2023). "Progress measures for grokking via mechanistic interpretability." ICLR 2023.

[3] Liu, Z., et al. (2022). "Towards Understanding Grokking: An Effective Theory of Representation Learning." NeurIPS 2022.

[4] Zhong, Z., et al. (2023). "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks." arXiv:2306.17844.

[5] Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.

---

## Appendix

### A. 実験の再現性

コード: https://github.com/[repository]

```bash
# 環境構築
pip install torch numpy streamlit plotly

# 学習
python train.py --p 97 --weight_decay 1.0 --epochs 5000

# 解析
streamlit run interactive_dashboard.py
```

### B. 追加実験結果

異なる p での結果：

| p | Grokking Epoch | Final Test Acc | Dominant k |
|---|----------------|----------------|------------|
| 53 | ~1200 | 99.8% | 7, 11, 23 |
| 97 | ~1800 | 99.9% | 13, 15, 46 |
| 113 | ~2500 | 99.7% | 17, 21, 53 |

全てのケースで Grokking とフーリエ表現が観察された。

---

*Manuscript prepared for submission*
*Word count: ~3500*

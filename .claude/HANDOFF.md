# 引き継ぎ情報 - Grokking可視化プロジェクト

## 現在の状態

### 完了したタスク
- [x] model.py - 3トークン入力対応（n_tokens=3）
- [x] data.py - 3値モジュラー加算タスク (a+b+c) mod p
- [x] train_with_analysis.py - 3トークン対応、バッチサイズオプション追加
- [x] analyze.py - 3トークン対応
- [x] visualize.py - 新関数追加（plot_fourier_surface, plot_correlation_visualization, plot_model_architecture）
- [x] interactive_tool.py - tool.png再現レイアウト（5行×6列 + 波形パネル）

### 未完了タスク
- [ ] 学習実行（p=113, 3トークン, 50000エポック）- **高速化が必要**

---

## 学習高速化の選択肢

### 現状の問題
- p=113, n_tokens=3 → データサイズ 113³ = 1,442,897 サンプル
- 50,000エポック × 8秒/エポック ≈ 120時間

### 高速化オプション

#### オプション1: pを小さくする（推奨）
```bash
# p=97 または p=67 を使用
python train_with_analysis.py --p 67 --n_tokens 3 --epochs 30000
```
- p=67: データサイズ 67³ = 300,763（約5倍高速）
- p=97: データサイズ 97³ = 912,673（約1.5倍高速）

#### オプション2: 2トークンに戻す
```bash
# 元のGrokking論文と同じ設定
python train_with_analysis.py --p 113 --n_tokens 2 --epochs 30000
```
- データサイズ: 113² = 12,769（約100倍高速）
- ただし説明用figの114×3と異なる

#### オプション3: エポック数を減らす
```bash
python train_with_analysis.py --p 113 --n_tokens 3 --epochs 10000
```
- Grokkingは通常10,000-30,000エポックで発生

#### オプション4: 学習率スケジューリング追加
- train_with_analysis.pyにLR schedulerを追加
- より少ないエポックで収束可能

---

## 次回作業手順

### 1. 高速化オプションを選択して学習実行
```bash
cd /Users/shigenoburyuto/Documents/GitHub/test_NN
source venv/bin/activate

# 例: p=67で高速学習
python train_with_analysis.py --p 67 --n_tokens 3 --epochs 30000 --batch_size 4096
```

### 2. 学習完了後、可視化確認
```bash
# インタラクティブツール
python interactive_tool.py --checkpoint_dir checkpoints --p 67

# 全可視化を生成
python visualize.py --checkpoint checkpoints/final.pt
```

### 3. 説明用figとの比較
- figures/ フォルダに生成された図と 説明用fig/ を比較

---

## ファイル構成

```
test_NN/
├── model.py           # 3トークン対応Transformer
├── data.py            # 3値モジュラー加算データセット
├── train_with_analysis.py  # 学習スクリプト（フーリエ解析付き）
├── analyze.py         # フーリエ解析
├── visualize.py       # 可視化（新関数追加済み）
├── interactive_tool.py # インタラクティブツール（再設計済み）
├── checkpoints/       # チェックポイント保存先
├── checkpoints_old/   # 以前の2トークンモデルのチェックポイント
├── figures/           # 生成された図
└── 説明用fig/         # 目標の図（config.png, tool.png, xy面.png, 相関.png）
```

---

## 補足

- 説明用figの「114×3」は p+1（0〜p-1 + 等号記号?）の可能性あり
- フーリエ相関の閾値は0.9（analyze.pyで定義）
- Grokking検出条件: train_acc > 99% かつ test_acc > 90%

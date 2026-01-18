# Grokking プロジェクト メモ

## リアルタイムモニターの起動方法

別ターミナルでグラフ付きモニターを起動する場合、**絶対パス**を使う：

```bash
source /Users/shigenoburyuto/Documents/GitHub/test_NN/venv/bin/activate && python /Users/shigenoburyuto/Documents/GitHub/test_NN/monitor_complete.py
```

または、先にcdしてから：
```bash
cd /Users/shigenoburyuto/Documents/GitHub/test_NN
source venv/bin/activate && python monitor_complete.py
```

## AppleScriptで別ターミナルを開く場合

```bash
osascript -e 'tell application "Terminal"
    activate
    do script "source /Users/shigenoburyuto/Documents/GitHub/test_NN/venv/bin/activate && python /Users/shigenoburyuto/Documents/GitHub/test_NN/monitor_complete.py"
end tell'
```

## 学習コマンド

### 2トークン版（高速）
```bash
python train_with_analysis.py --p 67 --n_tokens 2 --epochs 10000 --checkpoint_dir checkpoints_2token_p67
```

### 3トークン版
```bash
python train_with_analysis.py --p 67 --n_tokens 3 --epochs 15000 --checkpoint_dir checkpoints_3token_p67
```

## 可視化

### プレゼン用可視化生成
```bash
python visualize_presentation.py --checkpoint_dir checkpoints_2token_p67 --output_dir figures_presentation
```

### インタラクティブダッシュボード
```bash
streamlit run interactive_dashboard.py
```

## 生成されるファイル

| ディレクトリ | 内容 |
|-------------|------|
| `checkpoints_*/` | モデルチェックポイント、history.json |
| `figures_presentation_*/` | プレゼン用PNG、インタラクティブHTML |

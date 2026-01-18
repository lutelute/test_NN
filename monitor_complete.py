#!/usr/bin/env python3
"""
Grokking 学習モニター 完成版
進捗バー + グラフ + 詳細情報
"""

import re
import time
import os
from datetime import datetime

try:
    import plotext as plt
except ImportError:
    print("plotextをインストールしてください: pip install plotext")
    exit(1)

# 設定
LOG_FILE = "/private/tmp/claude/-Users-shigenoburyuto-Documents-GitHub-test-NN/tasks/b59e48f.output"
TOTAL_EPOCHS = 15000
REFRESH_INTERVAL = 3
MAX_HISTORY = 150


def parse_all_progress(log_file):
    """ログファイルから進捗履歴を解析"""
    epochs, train_accs, test_accs = [], [], []

    try:
        with open(log_file, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')

        pattern = r'Training:\s+\d+%\|[^|]*\|\s+(\d+)/\d+\s+\[[^\]]+\].*?train_acc=([\d.]+)%.*?test_acc=([\d.]+)%'

        last_epoch = -1
        for match in re.finditer(pattern, content):
            epoch = int(match.group(1))
            if epoch > last_epoch and epoch % 30 == 0:
                epochs.append(epoch)
                train_accs.append(float(match.group(2)))
                test_accs.append(float(match.group(3)))
                last_epoch = epoch

        return epochs, train_accs, test_accs
    except:
        return [], [], []


def parse_latest_progress(log_file):
    """最新の進捗を取得"""
    try:
        with open(log_file, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(80000, file_size)
            f.seek(max(0, file_size - read_size))
            content = f.read().decode('utf-8', errors='ignore')

        pattern = r'Training:\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+)\s+\[([^\]]+)<([^\]]+),\s*([\d.]+)(?:s/it|it/s).*?train_acc=([\d.]+)%.*?test_acc=([\d.]+)%'
        matches = list(re.finditer(pattern, content))

        if matches:
            last = matches[-1]
            return {
                'percent': int(last.group(1)),
                'epoch': int(last.group(2)),
                'total': int(last.group(3)),
                'elapsed': last.group(4),
                'remaining': last.group(5),
                'speed': float(last.group(6)),
                'train_acc': float(last.group(7)),
                'test_acc': float(last.group(8)),
            }
    except:
        pass
    return None


def create_progress_bar(percent, width=50):
    """プログレスバーを作成"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"


def get_grokking_status(train_acc, test_acc):
    """Grokking状態を判定"""
    if train_acc > 99 and test_acc > 95:
        return "🎉 GROKKING 完了!", "green"
    elif train_acc > 99 and test_acc > 50:
        return "⚡ GROKKING 発生中!", "yellow"
    elif train_acc > 99:
        return "⏳ 暗記完了 - Grokking待機中...", "cyan"
    elif train_acc > 80:
        return "📈 暗記フェーズ後半", "blue"
    elif train_acc > 50:
        return "📚 暗記フェーズ進行中", "magenta"
    elif train_acc > 10:
        return "🔄 学習初期段階", "white"
    else:
        return "🚀 学習開始", "white"


def main():
    print("=" * 60)
    print("   🧠 Grokking 学習モニター 完成版")
    print("=" * 60)
    print(f"📁 監視: {LOG_FILE}")
    print("⏳ 起動中...")
    time.sleep(2)

    while True:
        try:
            epochs, train_accs, test_accs = parse_all_progress(LOG_FILE)
            latest = parse_latest_progress(LOG_FILE)

            if epochs and latest:
                # 画面クリア
                plt.clear_terminal()

                # ヘッダー
                print("=" * 70)
                print("   🧠 Grokking 学習モニター")
                print("=" * 70)
                print()

                # プログレスバー
                bar = create_progress_bar(latest['percent'])
                print(f"  {bar} {latest['percent']:3d}%")
                print()

                # 詳細情報（2列表示）
                print(f"  📊 エポック:     {latest['epoch']:>6,} / {latest['total']:,}")
                print(f"  ⏱️  経過時間:    {latest['elapsed']:>12}")
                print(f"  ⏳ 残り時間:    {latest['remaining']:>12}")
                print(f"  🚀 速度:        {latest['speed']:>10.2f} s/epoch")
                print()

                # 精度情報
                status, color = get_grokking_status(latest['train_acc'], latest['test_acc'])
                print(f"  🎯 訓練精度:    {latest['train_acc']:>10.2f}%")
                print(f"  🧪 テスト精度:  {latest['test_acc']:>10.2f}%")
                print()
                print(f"  状態: {status}")
                print()

                # グラフ
                print("-" * 70)

                plt.clear_figure()
                plt.theme('dark')
                plt.plot_size(70, 15)

                # データをプロット
                epochs_plot = epochs[-MAX_HISTORY:]
                train_plot = train_accs[-MAX_HISTORY:]
                test_plot = test_accs[-MAX_HISTORY:]

                plt.plot(epochs_plot, train_plot, label="Train Acc", marker="braille", color="green")
                plt.plot(epochs_plot, test_plot, label="Test Acc", marker="braille", color="cyan")

                plt.title("Accuracy History")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")
                plt.ylim(0, 105)

                plt.show()

                print("-" * 70)
                print(f"  更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Ctrl+C で終了")

            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] データ待機中...", end="")

            time.sleep(REFRESH_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("   👋 モニター終了")
            print("=" * 60)
            break
        except Exception as e:
            print(f"\nエラー: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()

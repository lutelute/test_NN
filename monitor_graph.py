#!/usr/bin/env python3
"""
リアルタイム学習進捗モニター（グラフ付き）
ターミナルでグラフを表示
"""

import re
import time
import os
from datetime import datetime
from collections import deque

try:
    import plotext as plt
except ImportError:
    print("plotextをインストールしてください: pip install plotext")
    exit(1)

# ログファイルパス
LOG_FILE = "/private/tmp/claude/-Users-shigenoburyuto-Documents-GitHub-test-NN/tasks/b4d8e54.output"
TOTAL_EPOCHS = 30000
REFRESH_INTERVAL = 3  # 秒
MAX_HISTORY = 200  # グラフに表示する最大データ点数

def parse_all_progress(log_file):
    """ログファイルから全ての進捗を解析"""
    epochs = []
    train_accs = []
    test_accs = []

    try:
        with open(log_file, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')

        # 進捗を抽出（100エポックごとのログを取得）
        pattern = r'Training:\s+\d+%\|[^|]*\|\s+(\d+)/\d+\s+\[[^\]]+\].*?train_acc=([\d.]+)%.*?test_acc=([\d.]+)%'

        last_epoch = -1
        for match in re.finditer(pattern, content):
            epoch = int(match.group(1))
            # 重複を避け、一定間隔でサンプリング
            if epoch > last_epoch and epoch % 50 == 0:
                epochs.append(epoch)
                train_accs.append(float(match.group(2)))
                test_accs.append(float(match.group(3)))
                last_epoch = epoch

        return epochs, train_accs, test_accs
    except Exception as e:
        return [], [], []

def parse_latest_progress(log_file):
    """最新の進捗を取得"""
    try:
        with open(log_file, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(50000, file_size)
            f.seek(max(0, file_size - read_size))
            content = f.read().decode('utf-8', errors='ignore')

        pattern = r'Training:\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+)\s+\[([^\]]+)\].*?train_acc=([\d.]+)%.*?test_acc=([\d.]+)%'
        matches = list(re.finditer(pattern, content))

        if matches:
            last = matches[-1]
            return {
                'percent': int(last.group(1)),
                'epoch': int(last.group(2)),
                'total': int(last.group(3)),
                'time': last.group(4),
                'train_acc': float(last.group(5)),
                'test_acc': float(last.group(6)),
            }
    except Exception:
        pass
    return None

def main():
    print("📊 学習進捗モニター（グラフ付き）起動中...")
    print(f"📁 監視ファイル: {LOG_FILE}")
    print("Ctrl+C で終了\n")
    time.sleep(1)

    while True:
        try:
            # データ取得
            epochs, train_accs, test_accs = parse_all_progress(LOG_FILE)
            latest = parse_latest_progress(LOG_FILE)

            if epochs and latest:
                # 画面クリア
                plt.clear_terminal()
                plt.clear_figure()

                # グラフ設定
                plt.theme('dark')
                plt.title(f"Grokking 学習進捗 - Epoch {latest['epoch']:,}/{latest['total']:,} ({latest['percent']}%)")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")

                # 最新のMAX_HISTORY件のみプロット
                epochs_plot = epochs[-MAX_HISTORY:]
                train_plot = train_accs[-MAX_HISTORY:]
                test_plot = test_accs[-MAX_HISTORY:]

                # プロット
                plt.plot(epochs_plot, train_plot, label=f"Train: {latest['train_acc']:.1f}%", color="green")
                plt.plot(epochs_plot, test_plot, label=f"Test: {latest['test_acc']:.1f}%", color="cyan")

                # Y軸範囲
                plt.ylim(0, 105)

                # 表示
                plt.show()

                # 追加情報
                print(f"\n⏱️  経過時間: {latest['time']}")
                print(f"🎯 訓練精度: {latest['train_acc']:.2f}%  |  🧪 テスト精度: {latest['test_acc']:.2f}%")

                # Grokking状態
                if latest['train_acc'] > 95 and latest['test_acc'] > 90:
                    print("🎉 GROKKING 発生中!")
                elif latest['train_acc'] > 95:
                    print("📈 暗記完了 - Grokking待機中...")
                elif latest['train_acc'] > 50:
                    print("📚 暗記フェーズ進行中...")
                else:
                    print("🚀 学習開始段階...")

                print(f"\n更新: {datetime.now().strftime('%H:%M:%S')} | Ctrl+C で終了")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] データ待機中...")

            time.sleep(REFRESH_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n👋 モニター終了")
            break
        except Exception as e:
            print(f"エラー: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()

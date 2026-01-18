#!/usr/bin/env python3
"""
リアルタイム学習進捗モニター
別ターミナルで実行: python monitor.py &
"""

import re
import time
import os
import sys
from datetime import datetime

# ログファイルパス
LOG_FILE = "/private/tmp/claude/-Users-shigenoburyuto-Documents-GitHub-test-NN/tasks/b4d8e54.output"
TOTAL_EPOCHS = 30000
REFRESH_INTERVAL = 2  # 秒

def parse_latest_progress(log_file):
    """ログファイルから最新の進捗を解析"""
    try:
        with open(log_file, 'rb') as f:
            # ファイル末尾から読み込み
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(50000, file_size)
            f.seek(max(0, file_size - read_size))
            content = f.read().decode('utf-8', errors='ignore')

        # 最新の進捗を抽出
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
    except Exception as e:
        pass
    return None

def create_progress_bar(percent, width=40):
    """プログレスバーを作成"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"

def clear_screen():
    """画面をクリア"""
    os.system('clear' if os.name != 'nt' else 'cls')

def main():
    print("🔍 学習進捗モニター起動中...")
    print(f"📁 監視ファイル: {LOG_FILE}")
    print("Ctrl+C で終了\n")
    time.sleep(1)

    history = []

    while True:
        try:
            progress = parse_latest_progress(LOG_FILE)

            if progress:
                clear_screen()

                # ヘッダー
                print("=" * 60)
                print("   🧠 Grokking 学習モニター")
                print("=" * 60)
                print()

                # 進捗バー
                bar = create_progress_bar(progress['percent'])
                print(f"  進捗: {bar} {progress['percent']}%")
                print()

                # 詳細情報
                print(f"  📊 エポック:    {progress['epoch']:,} / {progress['total']:,}")
                print(f"  ⏱️  経過時間:   {progress['time']}")
                print()
                print(f"  🎯 訓練精度:   {progress['train_acc']:6.2f}%")
                print(f"  🧪 テスト精度: {progress['test_acc']:6.2f}%")
                print()

                # Grokking検出
                if progress['train_acc'] > 95 and progress['test_acc'] > 90:
                    print("  🎉 GROKKING 発生中!")
                elif progress['train_acc'] > 95:
                    print("  📈 暗記完了 - Grokking待機中...")
                elif progress['train_acc'] > 50:
                    print("  📚 暗記フェーズ進行中...")
                else:
                    print("  🚀 学習開始段階...")

                print()
                print("-" * 60)
                print(f"  更新時刻: {datetime.now().strftime('%H:%M:%S')}")
                print("  Ctrl+C で終了")

                # 履歴に追加（グラフ用）
                history.append({
                    'epoch': progress['epoch'],
                    'train': progress['train_acc'],
                    'test': progress['test_acc']
                })

            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ログ待機中...")

            time.sleep(REFRESH_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n👋 モニター終了")
            break
        except Exception as e:
            print(f"エラー: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()

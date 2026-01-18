#!/usr/bin/env python3
"""
リアルタイム進捗モニター
Usage: python monitor_live.py [--watch]
"""
import json
import os
import time
import sys
import re

def get_from_history(checkpoint_dir):
    """history.jsonから進捗を取得"""
    history_path = os.path.join(checkpoint_dir, "history.json")
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r") as f:
            h = json.load(f)
        return {
            "epoch": len(h["train_acc"]),
            "train_acc": h["train_acc"][-1] * 100,
            "test_acc": h["test_acc"][-1] * 100,
            "train_loss": h["train_loss"][-1],
        }
    except:
        return None

def get_from_output(output_file):
    """出力ファイルから最新の進捗を取得"""
    if not os.path.exists(output_file):
        return None
    try:
        # ファイル末尾から読み取り
        with open(output_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 5000))
            content = f.read().decode("utf-8", errors="ignore")
        
        # 最新の精度を抽出
        acc_match = re.findall(r"train_acc=([0-9.]+)%, test_acc=([0-9.]+)%", content)
        epoch_match = re.findall(r"(\d+)/15000", content)
        
        if acc_match and epoch_match:
            return {
                "epoch": int(epoch_match[-1]),
                "train_acc": float(acc_match[-1][0]),
                "test_acc": float(acc_match[-1][1]),
            }
    except:
        pass
    return None

def display_progress():
    """進捗を表示"""
    configs = [
        ("2-token P=67", "checkpoints_2token_p67", None, 10000),
        ("3-token P=67", "checkpoints_3token_p67", 
         "/private/tmp/claude/-Users-shigenoburyuto-Documents-GitHub-test-NN/tasks/b59e48f.output", 15000),
    ]
    
    print("\033[2J\033[H")  # Clear
    print("╔" + "═" * 58 + "╗")
    print("║" + "  📊 Grokking Training Monitor".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    
    for name, checkpoint_dir, output_file, max_epoch in configs:
        # history.jsonを優先、なければ出力ファイルから
        p = get_from_history(checkpoint_dir)
        if p is None and output_file:
            p = get_from_output(output_file)
        
        print("║" + f"  {name}".ljust(58) + "║")
        
        if p:
            # プログレスバー
            progress = min(p["epoch"] / max_epoch, 1.0)
            bar_len = 40
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            # 状態判定
            if p["test_acc"] > 90:
                status = "✅ Grokking!"
                status_color = "\033[92m"  # Green
            elif p["train_acc"] > 90 and p["test_acc"] < 50:
                status = "📝 Memorizing"
                status_color = "\033[93m"  # Yellow
            else:
                status = "🔄 Training"
                status_color = "\033[94m"  # Blue
            
            print("║" + f"    [{bar}] {p['epoch']:,}/{max_epoch:,}".ljust(58) + "║")
            print("║" + f"    Train: {p['train_acc']:6.2f}%   Test: {p['test_acc']:6.2f}%   {status}".ljust(58) + "║")
        else:
            print("║" + "    ⏳ Waiting for data...".ljust(58) + "║")
        
        print("║" + " " * 58 + "║")
    
    print("╠" + "═" * 58 + "╣")
    print("║" + f"  Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}".ljust(58) + "║")
    print("╚" + "═" * 58 + "╝")

def main():
    watch = "--watch" in sys.argv or "-w" in sys.argv
    
    if watch:
        print("Monitoring... (Ctrl+C to stop)")
        try:
            while True:
                display_progress()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        display_progress()

if __name__ == "__main__":
    main()

"""
Grokking用 Transformerモデル
モジュラー加算タスク用のシンプルな1層Transformer

3トークン入力対応: (a, b, c) → (a + b + c) mod p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置エンコーディング（学習可能）"""

    def __init__(self, d_model: int, max_len: int = 2):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pos_embedding(positions)


class TransformerBlock(nn.Module):
    """単一のTransformerブロック"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_intermediates: bool = False):
        # Self-attention with residual
        # average_attn_weights=False でヘッドごとのAttention重みを取得
        attn_out, attn_weights = self.attention(x, x, x, average_attn_weights=False)
        post_attn = self.ln1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(post_attn)
        output = self.ln2(post_attn + self.dropout(ff_out))

        if return_intermediates:
            return output, {
                "attn_out": attn_out,
                "attn_weights": attn_weights,
                "post_attn": post_attn,
                "ff_out": ff_out,
            }
        return output


class ModularAdditionTransformer(nn.Module):
    """
    モジュラー加算用Transformer

    入力: (a, b, c) の3トークン列
    出力: p クラスの分類（(a + b + c) mod p を予測）
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 1,
                 dropout: float = 0.0, n_tokens: int = 3):
        """
        Args:
            p: 素数（語彙サイズ = 出力クラス数）
            d_model: 埋め込み次元
            n_heads: アテンションヘッド数
            n_layers: Transformerブロック数
            dropout: ドロップアウト率
            n_tokens: 入力トークン数（デフォルト3）
        """
        super().__init__()

        self.p = p
        self.d_model = d_model
        self.n_tokens = n_tokens

        # トークン埋め込み（0 ~ p-1 の数値を埋め込む）
        self.token_embedding = nn.Embedding(p, d_model)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_len=n_tokens)

        # Transformerブロック
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 出力層
        self.output_layer = nn.Linear(d_model, p)

        # 重みの初期化
        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (batch, n_tokens) - 各要素は 0 ~ p-1 の整数

        Returns:
            logits: (batch, p) - 各クラスのロジット
        """
        # トークン埋め込み
        x = self.token_embedding(x)  # (batch, n_tokens, d_model)

        # 位置エンコーディング
        x = self.pos_encoding(x)

        # Transformerブロック
        for block in self.transformer_blocks:
            x = block(x)

        # 最後のトークンの出力を使用（または平均を取る）
        # ここでは両方のトークンの平均を使用
        x = x.mean(dim=1)  # (batch, d_model)

        # 出力層
        logits = self.output_layer(x)  # (batch, p)

        return logits

    def get_embedding_weights(self):
        """埋め込み層の重みを取得（フーリエ解析用）"""
        return self.token_embedding.weight.detach().cpu().numpy()

    def forward_with_intermediates(self, x):
        """
        中間層の出力を含めてフォワードパスを実行

        Args:
            x: (batch, n_tokens) - 各要素は 0 ~ p-1 の整数

        Returns:
            logits: (batch, p)
            intermediates: 各層の出力を含む辞書
        """
        intermediates = {}

        # トークン埋め込み
        embed = self.token_embedding(x)  # (batch, n_tokens, d_model)
        intermediates["embed"] = embed.detach()

        # 位置エンコーディング
        embed_pos = self.pos_encoding(embed)
        intermediates["embed_pos"] = embed_pos.detach()

        # Transformerブロック
        h = embed_pos
        for i, block in enumerate(self.transformer_blocks):
            h, block_intermediates = block(h, return_intermediates=True)
            intermediates[f"block_{i}_attn_out"] = block_intermediates["attn_out"].detach()
            intermediates[f"block_{i}_attn_weights"] = block_intermediates["attn_weights"].detach()
            intermediates[f"block_{i}_post_attn"] = block_intermediates["post_attn"].detach()
            intermediates[f"block_{i}_ff_out"] = block_intermediates["ff_out"].detach()
            intermediates[f"block_{i}_output"] = h.detach()

        # プーリング前
        intermediates["pre_pool"] = h.detach()

        # プーリング（平均）
        pooled = h.mean(dim=1)
        intermediates["pooled"] = pooled.detach()

        # 出力層
        logits = self.output_layer(pooled)
        intermediates["logits"] = logits.detach()

        return logits, intermediates

    def get_layer_outputs_for_analysis(self, p: int = None):
        """
        各数値(0〜p-1)に対する各層の出力を取得（フーリエ解析用）

        Returns:
            dict: 各層の出力 (p, d_model) の辞書
        """
        if p is None:
            p = self.p

        self.eval()
        device = next(self.parameters()).device

        # 全ての (a, b, c) トリプルを生成して実行
        # ここでは単純に a=0, b=0 として、c=0〜p-1 の埋め込みを取得
        inputs = torch.zeros(p, self.n_tokens, dtype=torch.long, device=device)
        inputs[:, -1] = torch.arange(p, device=device)  # (0, 0, 0), (0, 0, 1), ..., (0, 0, p-1)

        with torch.no_grad():
            _, intermediates = self.forward_with_intermediates(inputs)

        # 各層の出力をCPU numpy配列に変換
        outputs = {}
        for key, value in intermediates.items():
            if value.dim() == 3:  # (batch, seq, d_model)
                # 各位置のトークン出力を分けて保存
                for pos in range(self.n_tokens):
                    outputs[f"{key}_pos{pos}"] = value[:, pos, :].cpu().numpy()
            elif value.dim() == 2:  # (batch, d_model) or (batch, p)
                outputs[key] = value.cpu().numpy()

        return outputs


if __name__ == "__main__":
    # テスト
    p = 113
    model = ModularAdditionTransformer(p, n_tokens=3)

    # ダミー入力（3トークン）
    x = torch.randint(0, p, (32, 3))
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 埋め込み重みの形状
    embed_weights = model.get_embedding_weights()
    print(f"Embedding weights shape: {embed_weights.shape}")

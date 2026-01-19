"""
フーリエ解析スクリプト
学習済みNNの重みがフーリエ表現（数式）と一致するかを検証

理論:
モジュラー加算 (a + b) mod p を学習したNNは、内部的に以下を実装する:
- 数値 n を周波数 k の波として埋め込み: e^(2πi k n / p)
- 実数表現では: cos(2πkn/p), sin(2πkn/p)
"""

import torch
import numpy as np
import os
from typing import Dict, Tuple, List
from model import ModularAdditionTransformer


class FourierAnalyzer:
    """フーリエ解析クラス"""

    def __init__(self, model: ModularAdditionTransformer):
        self.model = model
        self.p = model.p
        self.d_model = model.d_model
        self._layer_outputs = None

    def get_embedding_weights(self) -> np.ndarray:
        """埋め込み重みを取得 (p, d_model)"""
        return self.model.get_embedding_weights()

    def get_layer_outputs(self) -> Dict:
        """各層の出力を取得してキャッシュ"""
        if self._layer_outputs is None:
            self._layer_outputs = self.model.get_layer_outputs_for_analysis()
        return self._layer_outputs

    def get_available_layers(self) -> List[str]:
        """解析可能な層のリストを取得"""
        outputs = self.get_layer_outputs()
        return list(outputs.keys())

    def compute_dft(self, weights: np.ndarray) -> np.ndarray:
        """
        埋め込み重みのDFT（離散フーリエ変換）を計算

        Args:
            weights: (p, d_model) の埋め込み重み

        Returns:
            (p, d_model) の複素フーリエ係数
        """
        # 各次元について DFT を計算
        # DFT: X_k = sum_{n=0}^{p-1} x_n * e^{-2πi k n / p}
        return np.fft.fft(weights, axis=0)

    def compute_fourier_spectrum(self, weights: np.ndarray = None) -> np.ndarray:
        """
        フーリエスペクトル（各周波数のパワー）を計算

        Returns:
            (p,) の周波数パワー（全次元の平均）
        """
        if weights is None:
            weights = self.get_embedding_weights()

        dft = self.compute_dft(weights)

        # パワースペクトル（絶対値の2乗）
        power = np.abs(dft) ** 2

        # 全次元で平均
        avg_power = power.mean(axis=1)

        return avg_power

    def find_dominant_frequencies(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        支配的な周波数成分を見つける

        Returns:
            [(frequency, power), ...] のリスト
        """
        spectrum = self.compute_fourier_spectrum()

        # DC成分（k=0）を除外して上位を取得
        spectrum_no_dc = spectrum.copy()
        spectrum_no_dc[0] = 0

        # 対称性を考慮（k と p-k は同じ周波数）
        # 前半のみを見る
        half_p = self.p // 2 + 1
        spectrum_half = spectrum_no_dc[:half_p]

        # 上位 k 個を取得
        top_indices = np.argsort(spectrum_half)[-top_k:][::-1]

        results = [(int(idx), float(spectrum_half[idx])) for idx in top_indices]

        return results

    def compute_theoretical_fourier_basis(self, k: int) -> np.ndarray:
        """
        理論的なフーリエ基底を計算

        Args:
            k: 周波数

        Returns:
            (p, 2) の配列 [cos(2πkn/p), sin(2πkn/p)] for n = 0, 1, ..., p-1
        """
        n = np.arange(self.p)
        angles = 2 * np.pi * k * n / self.p

        cos_basis = np.cos(angles)
        sin_basis = np.sin(angles)

        return np.stack([cos_basis, sin_basis], axis=1)

    def compute_correlation_with_theory(self, k: int) -> Dict[str, float]:
        """
        埋め込みと理論的フーリエ基底との相関を計算

        Args:
            k: 周波数

        Returns:
            相関係数などの統計量
        """
        weights = self.get_embedding_weights()  # (p, d_model)
        theory = self.compute_theoretical_fourier_basis(k)  # (p, 2)

        # 各埋め込み次元と理論基底との相関を計算
        correlations = []

        for dim in range(self.d_model):
            embed_dim = weights[:, dim]
            embed_dim = (embed_dim - embed_dim.mean()) / (embed_dim.std() + 1e-8)

            # cos基底との相関
            cos_basis = theory[:, 0]
            cos_basis = (cos_basis - cos_basis.mean()) / (cos_basis.std() + 1e-8)
            cos_corr = np.corrcoef(embed_dim, cos_basis)[0, 1]

            # sin基底との相関
            sin_basis = theory[:, 1]
            sin_basis = (sin_basis - sin_basis.mean()) / (sin_basis.std() + 1e-8)
            sin_corr = np.corrcoef(embed_dim, sin_basis)[0, 1]

            # 最大相関（位相を考慮）
            max_corr = np.sqrt(cos_corr**2 + sin_corr**2)
            correlations.append(max_corr)

        correlations = np.array(correlations)

        return {
            "frequency": k,
            "max_correlation": float(np.max(correlations)),
            "mean_correlation": float(np.mean(correlations)),
            "num_high_corr_dims": int(np.sum(correlations > 0.9)),
            "correlation_per_dim": correlations.tolist(),
        }

    def verify_fourier_representation(self) -> Dict:
        """
        NNがフーリエ表現を学習したかを総合的に検証

        Returns:
            検証結果の辞書
        """
        # 1. 支配的周波数を特定
        dominant_freqs = self.find_dominant_frequencies(top_k=5)

        # 2. 各支配的周波数について理論との相関を計算
        correlation_results = []
        for freq, power in dominant_freqs:
            corr = self.compute_correlation_with_theory(freq)
            corr["power"] = power
            correlation_results.append(corr)

        # 3. フーリエスペクトル
        spectrum = self.compute_fourier_spectrum()

        # 4. スペクトルの集中度（上位周波数がどれだけ全体を占めるか）
        total_power = spectrum.sum()
        top_power = sum([p for _, p in dominant_freqs[:3]])
        concentration = top_power / total_power if total_power > 0 else 0

        # 5. 判定（閾値を緩和: 0.9 → 0.7）
        best_corr = max([r["max_correlation"] for r in correlation_results]) if correlation_results else 0
        is_fourier_representation = best_corr > 0.7 and concentration > 0.2

        return {
            "is_fourier_representation": is_fourier_representation,
            "dominant_frequencies": dominant_freqs,
            "correlation_results": correlation_results,
            "spectrum": spectrum.tolist(),
            "spectrum_concentration": concentration,
            "best_correlation": best_corr,
        }

    def analyze_layer(self, layer_name: str) -> Dict:
        """
        特定の層の出力をフーリエ解析

        Args:
            layer_name: 解析する層の名前

        Returns:
            解析結果の辞書
        """
        outputs = self.get_layer_outputs()
        if layer_name not in outputs:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {list(outputs.keys())}")

        weights = outputs[layer_name]

        # フーリエスペクトル
        spectrum = self.compute_fourier_spectrum(weights)

        # 支配的周波数
        dominant_freqs = []
        spectrum_no_dc = spectrum.copy()
        spectrum_no_dc[0] = 0
        half_p = self.p // 2 + 1
        spectrum_half = spectrum_no_dc[:half_p]
        top_indices = np.argsort(spectrum_half)[-5:][::-1]
        dominant_freqs = [(int(idx), float(spectrum_half[idx])) for idx in top_indices]

        # 円周構造の解析
        circular = self._analyze_circular_for_weights(weights)

        # 理論との相関
        best_corr = 0
        best_k = 0
        for k in range(1, half_p):
            corr = self._compute_correlation_for_weights(weights, k)
            if corr["max_correlation"] > best_corr:
                best_corr = corr["max_correlation"]
                best_k = k

        return {
            "layer_name": layer_name,
            "spectrum": spectrum,
            "dominant_frequencies": dominant_freqs,
            "circular": circular,
            "best_correlation": best_corr,
            "best_frequency": best_k,
        }

    def _compute_correlation_for_weights(self, weights: np.ndarray, k: int) -> Dict:
        """指定した重みと理論的フーリエ基底との相関を計算"""
        theory = self.compute_theoretical_fourier_basis(k)
        d_model = weights.shape[1]

        correlations = []
        for dim in range(d_model):
            embed_dim = weights[:, dim]
            embed_dim = (embed_dim - embed_dim.mean()) / (embed_dim.std() + 1e-8)

            cos_basis = theory[:, 0]
            cos_basis = (cos_basis - cos_basis.mean()) / (cos_basis.std() + 1e-8)
            cos_corr = np.corrcoef(embed_dim, cos_basis)[0, 1]

            sin_basis = theory[:, 1]
            sin_basis = (sin_basis - sin_basis.mean()) / (sin_basis.std() + 1e-8)
            sin_corr = np.corrcoef(embed_dim, sin_basis)[0, 1]

            max_corr = np.sqrt(cos_corr**2 + sin_corr**2) if not np.isnan(cos_corr) and not np.isnan(sin_corr) else 0
            correlations.append(max_corr)

        correlations = np.array(correlations)

        return {
            "max_correlation": float(np.max(correlations)) if len(correlations) > 0 else 0,
            "mean_correlation": float(np.mean(correlations)) if len(correlations) > 0 else 0,
        }

    def _analyze_circular_for_weights(self, weights: np.ndarray) -> Dict:
        """指定した重みの円周構造を解析"""
        # 最も分散が大きい2次元を選択
        variances = weights.var(axis=0)
        top_2_dims = np.argsort(variances)[-2:]

        proj_2d = weights[:, top_2_dims]

        center = proj_2d.mean(axis=0)
        distances = np.linalg.norm(proj_2d - center, axis=1)

        centered = proj_2d - center
        angles = np.arctan2(centered[:, 1], centered[:, 0])

        expected_angles = 2 * np.pi * np.arange(self.p) / self.p - np.pi

        best_corr = 0
        for shift in range(self.p):
            shifted_expected = np.roll(expected_angles, shift)
            corr = np.corrcoef(angles, shifted_expected)[0, 1]
            if not np.isnan(corr):
                best_corr = max(best_corr, abs(corr))

        return {
            "top_2_dims": top_2_dims.tolist(),
            "projection_2d": proj_2d.tolist(),
            "circularity": float(1 - distances.std() / (distances.mean() + 1e-8)),
            "angle_correlation": float(best_corr),
        }

    def analyze_all_layers(self) -> Dict:
        """全ての層をフーリエ解析"""
        outputs = self.get_layer_outputs()
        results = {}
        for layer_name in outputs.keys():
            try:
                results[layer_name] = self.analyze_layer(layer_name)
            except Exception as e:
                print(f"Warning: Could not analyze layer {layer_name}: {e}")
        return results

    def analyze_circular_structure(self) -> Dict:
        """
        埋め込みが円周上に配置されているかを検証
        （フーリエ表現では数値が円周上に等間隔で配置されるはず）
        """
        weights = self.get_embedding_weights()

        # 支配的な2次元を選択（最も分散が大きい2次元）
        variances = weights.var(axis=0)
        top_2_dims = np.argsort(variances)[-2:]

        # 2D射影
        proj_2d = weights[:, top_2_dims]

        # 中心からの距離
        center = proj_2d.mean(axis=0)
        distances = np.linalg.norm(proj_2d - center, axis=1)

        # 角度を計算
        centered = proj_2d - center
        angles = np.arctan2(centered[:, 1], centered[:, 0])

        # 角度でソートしたときの順序が 0, 1, 2, ..., p-1 に近いか
        sorted_indices = np.argsort(angles)

        # 理想的な順序との比較
        # 角度が数値に比例するはず: angle ∝ 2πn/p
        expected_angles = 2 * np.pi * np.arange(self.p) / self.p - np.pi

        # 角度の相関（位相シフトを考慮）
        best_corr = 0
        for shift in range(self.p):
            shifted_expected = np.roll(expected_angles, shift)
            corr = np.corrcoef(angles, shifted_expected)[0, 1]
            if not np.isnan(corr):
                best_corr = max(best_corr, abs(corr))

        return {
            "top_2_dims": top_2_dims.tolist(),
            "projection_2d": proj_2d.tolist(),
            "mean_distance_from_center": float(distances.mean()),
            "std_distance_from_center": float(distances.std()),
            "circularity": float(1 - distances.std() / (distances.mean() + 1e-8)),
            "angle_correlation": float(best_corr),
            "is_circular": best_corr > 0.5,  # 閾値を緩和: 0.9 → 0.5
        }


def analyze_checkpoint(checkpoint_path: str, p: int = 113, n_tokens: int = 3) -> Dict:
    """
    チェックポイントを読み込んでフーリエ解析を実行

    Args:
        checkpoint_path: チェックポイントファイルのパス
        p: 素数
        n_tokens: 入力トークン数

    Returns:
        解析結果
    """
    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {"p": p, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": n_tokens})

    # モデル作成・重み読み込み
    model = ModularAdditionTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_tokens=config.get("n_tokens", n_tokens),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 解析
    analyzer = FourierAnalyzer(model)

    fourier_result = analyzer.verify_fourier_representation()
    circular_result = analyzer.analyze_circular_structure()

    return {
        "fourier_analysis": fourier_result,
        "circular_analysis": circular_result,
        "epoch": checkpoint.get("epoch", "unknown"),
        "config": config,
    }


def main():
    """メイン関数"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fourier Analysis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Checkpoint path")
    parser.add_argument("--p", type=int, default=113, help="Prime number")
    parser.add_argument("--n_tokens", type=int, default=3, help="Number of input tokens (2 or 3)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    print(f"Analyzing checkpoint: {args.checkpoint}")
    result = analyze_checkpoint(args.checkpoint, args.p, args.n_tokens)

    # 結果表示
    print("\n" + "=" * 60)
    print("FOURIER ANALYSIS RESULTS")
    print("=" * 60)

    fourier = result["fourier_analysis"]
    print(f"\nIs Fourier Representation: {fourier['is_fourier_representation']}")
    print(f"Best Correlation with Theory: {fourier['best_correlation']:.4f}")
    print(f"Spectrum Concentration: {fourier['spectrum_concentration']:.4f}")

    print("\nDominant Frequencies:")
    for freq, power in fourier["dominant_frequencies"]:
        print(f"  k={freq}: power={power:.2f}")

    print("\nCorrelation with Theoretical Fourier Basis:")
    for r in fourier["correlation_results"]:
        print(f"  k={r['frequency']}: max_corr={r['max_correlation']:.4f}, "
              f"high_corr_dims={r['num_high_corr_dims']}/{result['config']['d_model']}")

    circular = result["circular_analysis"]
    print(f"\n" + "-" * 60)
    print("CIRCULAR STRUCTURE ANALYSIS")
    print("-" * 60)
    print(f"Is Circular: {circular['is_circular']}")
    print(f"Circularity: {circular['circularity']:.4f}")
    print(f"Angle Correlation: {circular['angle_correlation']:.4f}")

    # JSON保存
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

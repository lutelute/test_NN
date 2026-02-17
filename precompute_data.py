#!/usr/bin/env python3
"""
Precompute data from checkpoints for Streamlit Cloud deployment.
Extracts all analysis data and saves as numpy/JSON files (~5MB total).

Usage:
    python precompute_data.py [--checkpoint-dir checkpoints_demo_2ep] [--output-dir precomputed_data]
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch

from model import ModularAdditionTransformer
from analyze import FourierAnalyzer


def _sanitize_for_json(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {
        "p": 97, "d_model": 128, "n_heads": 4, "n_layers": 1, "n_tokens": 2
    })

    model = ModularAdditionTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_tokens=config.get("n_tokens", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint.get("epoch", 0)


def get_epoch_path(checkpoint_dir, epoch):
    """Get epoch file path (supports both naming conventions)."""
    path1 = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:05d}.pt")
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    if os.path.exists(path2):
        return path2
    return None


def compute_tab2_fourier(model, config):
    """Compute Fourier analysis data for Tab 2."""
    print("  Computing Tab 2: Fourier Analysis...")
    analyzer = FourierAnalyzer(model)
    p = config["p"]

    # Embedding weights
    weights = analyzer.get_embedding_weights()  # (p, d_model)

    # Fourier spectrum
    spectrum = analyzer.compute_fourier_spectrum()  # (p,)

    # Dominant frequencies
    dominant = analyzer.find_dominant_frequencies(top_k=5)

    # Circular structure
    circular = analyzer.analyze_circular_structure()
    proj_2d = np.array(circular["projection_2d"])  # (p, 2)

    # Verification result
    verification = analyzer.verify_fourier_representation()

    meta = {
        "circular": {
            "top_2_dims": circular["top_2_dims"],
            "angle_correlation": float(circular["angle_correlation"]),
            "is_circular": bool(circular["is_circular"]),
            "circularity": float(circular["circularity"]),
            "mean_distance_from_center": float(circular["mean_distance_from_center"]),
            "std_distance_from_center": float(circular["std_distance_from_center"]),
        },
        "dominant_frequencies": [[int(f), float(p)] for f, p in dominant],
        "verification": {
            "is_fourier_representation": bool(verification["is_fourier_representation"]),
            "best_correlation": float(verification["best_correlation"]),
            "spectrum_concentration": float(verification["spectrum_concentration"]),
            "correlation_results": _sanitize_for_json(verification["correlation_results"]),
        },
    }

    return weights, spectrum, proj_2d, meta


def compute_tab4_output(model, config):
    """Compute model output data for Tab 4."""
    print("  Computing Tab 4: Model Output...")
    p = config["p"]
    n_tokens = config.get("n_tokens", 2)

    if n_tokens == 2:
        all_inputs = torch.tensor([[a, b] for a in range(p) for b in range(p)])
        expected = np.array([[(a + b) % p for b in range(p)] for a in range(p)])
    else:
        all_inputs = torch.tensor([[a, 0, c] for a in range(p) for c in range(p)])
        expected = np.array([[(a + c) % p for c in range(p)] for a in range(p)])

    with torch.no_grad():
        all_logits = model(all_inputs)

    pred_matrix = all_logits.argmax(dim=-1).numpy().reshape(p, p)

    correct_indices = expected.flatten()
    logit_matrix = all_logits[np.arange(p * p), correct_indices].numpy().reshape(p, p)

    full_logits = all_logits.numpy()  # (p*p, p)

    return pred_matrix, logit_matrix, full_logits


def compute_tab7_attention(model, config):
    """Compute attention data for Tab 7."""
    print("  Computing Tab 7: Attention Analysis...")
    p = config["p"]
    n_tokens = config.get("n_tokens", 2)
    sample_step = max(1, p // 30)

    a_vals = list(range(0, p, sample_step))
    b_vals = list(range(0, p, sample_step))

    inputs = []
    for a in a_vals:
        for b in b_vals:
            if n_tokens == 2:
                inputs.append([a, b])
            else:
                inputs.append([a, b, 0])

    inputs_tensor = torch.tensor(inputs)
    with torch.no_grad():
        _, intermediates = model.forward_with_intermediates(inputs_tensor)

    attn_weights = intermediates["block_0_attn_weights"].numpy()  # (batch, heads, seq, seq)

    # Embeddings for sample display
    embeddings = intermediates["embed"][0].numpy()  # (n_tokens, d_model) - first sample

    return attn_weights, a_vals, b_vals, embeddings


def compute_tab8_neurons(model, config):
    """Compute neuron analysis data for Tab 8."""
    print("  Computing Tab 8: Neuron Analysis...")
    p = config["p"]
    n_tokens = config.get("n_tokens", 2)

    # Single token activations for Fourier correlation
    if n_tokens == 2:
        inputs_single = torch.tensor([[n, 0] for n in range(p)])
    else:
        inputs_single = torch.tensor([[n, 0, 0] for n in range(p)])

    with torch.no_grad():
        _, inter_single = model.forward_with_intermediates(inputs_single)

    ff_out_single = inter_single.get("block_0_ff_out", None)
    if ff_out_single is None:
        print("    WARNING: block_0_ff_out not found")
        return None, None

    neuron_acts = ff_out_single[:, 0, :].numpy()  # (p, n_neurons)
    n_neurons = neuron_acts.shape[1]
    n_freqs = p // 2

    # Fourier correlation matrix
    corr_matrix = np.zeros((n_neurons, n_freqs))
    n = np.arange(p)

    for k in range(n_freqs):
        cos_basis = np.cos(2 * np.pi * k * n / p)
        sin_basis = np.sin(2 * np.pi * k * n / p)

        for ni in range(n_neurons):
            act = neuron_acts[:, ni]
            if np.std(act) > 1e-6:
                cos_corr = abs(np.corrcoef(act, cos_basis)[0, 1])
                sin_corr = abs(np.corrcoef(act, sin_basis)[0, 1])
                if not np.isnan(cos_corr) and not np.isnan(sin_corr):
                    corr_matrix[ni, k] = max(cos_corr, sin_corr)

    # Also compute activations for 2D map
    sample_step = max(1, p // 25)
    a_vals = list(range(0, p, sample_step))
    b_vals = list(range(0, p, sample_step))

    inputs_2d = []
    for a in a_vals:
        for b in b_vals:
            if n_tokens == 2:
                inputs_2d.append([a, b])
            else:
                inputs_2d.append([a, b, 0])

    with torch.no_grad():
        _, inter_2d = model.forward_with_intermediates(torch.tensor(inputs_2d))

    ff_out_2d = inter_2d.get("block_0_ff_out", None)
    activations_2d = None
    if ff_out_2d is not None:
        # Mean pool over token positions
        activations_2d = ff_out_2d.mean(dim=1).numpy()  # (batch, n_neurons)

    return corr_matrix, neuron_acts, activations_2d, a_vals, b_vals


def compute_tab3_fourier_history(checkpoint_dir, config, max_epochs=50):
    """Compute Fourier metrics over training epochs for Tab 3."""
    print("  Computing Tab 3: Fourier History...")

    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))) + \
                  sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))

    if not epoch_files:
        print("    No epoch checkpoints found")
        return None

    # Extract epoch numbers
    epochs_available = []
    for f in epoch_files:
        basename = os.path.basename(f)
        ep_str = basename.replace("checkpoint_epoch_", "").replace("epoch_", "").replace(".pt", "")
        try:
            epochs_available.append(int(ep_str))
        except ValueError:
            continue

    epochs_available = sorted(set(epochs_available))

    # Sample epochs to keep it manageable
    if len(epochs_available) > max_epochs:
        step = len(epochs_available) // max_epochs
        sampled_epochs = epochs_available[::step]
        if epochs_available[-1] not in sampled_epochs:
            sampled_epochs.append(epochs_available[-1])
    else:
        sampled_epochs = epochs_available

    result = {
        "epochs": [],
        "best_correlations": [],
        "angle_correlations": [],
        "spectrum_concentrations": [],
        "circularities": [],
    }

    for i, ep in enumerate(sampled_epochs):
        ep_path = get_epoch_path(checkpoint_dir, ep)
        if ep_path is None:
            continue

        try:
            model, _, _ = load_model_from_checkpoint(ep_path)
            analyzer = FourierAnalyzer(model)

            verification = analyzer.verify_fourier_representation()
            circular = analyzer.analyze_circular_structure()

            result["epochs"].append(ep)
            result["best_correlations"].append(verification["best_correlation"])
            result["angle_correlations"].append(circular["angle_correlation"])
            result["spectrum_concentrations"].append(verification["spectrum_concentration"])
            result["circularities"].append(circular["circularity"])

            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(sampled_epochs)} epochs")
        except Exception as e:
            print(f"    Warning: Failed to process epoch {ep}: {e}")
            continue

    print(f"    Computed {len(result['epochs'])} epochs")
    return result


def compute_tab5_animation(checkpoint_dir, config, max_frames=30):
    """Compute animation frame data for Tab 5."""
    print("  Computing Tab 5: Animation Data...")

    p = config["p"]
    d_model = config.get("d_model", 128)
    n_tokens = config.get("n_tokens", 2)

    # Get available epochs
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))) + \
                  sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))

    epochs_available = []
    for f in epoch_files:
        basename = os.path.basename(f)
        ep_str = basename.replace("checkpoint_epoch_", "").replace("epoch_", "").replace(".pt", "")
        try:
            epochs_available.append(int(ep_str))
        except ValueError:
            continue
    epochs_available = sorted(set(epochs_available))

    if not epochs_available:
        return None, None

    # Sample epochs for animation frames
    step = max(1, len(epochs_available) // max_frames)
    sampled_epochs = epochs_available[::step]
    if epochs_available[-1] not in sampled_epochs:
        sampled_epochs.append(epochs_available[-1])

    # First, find fixed dimensions from best model
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(best_path):
        best_path = get_epoch_path(checkpoint_dir, epochs_available[-1])

    ref_model, _, _ = load_model_from_checkpoint(best_path)

    # Compute fixed dims using best model
    np.random.seed(42)
    samples_per_sum = 3
    all_circle_pairs = []
    sum_labels = []
    for s in range(p):
        for i in range(samples_per_sum):
            a = (s + i * 17) % p
            b = (s - a) % p
            all_circle_pairs.append([a, b])
            sum_labels.append(s)
    sum_labels_arr = np.array(sum_labels)

    if n_tokens == 2:
        inputs_circle = torch.tensor(all_circle_pairs)
    else:
        inputs_circle = torch.tensor([[a, b, 0] for a, b in all_circle_pairs])

    with torch.no_grad():
        _, ref_inter = ref_model.forward_with_intermediates(inputs_circle)
    ref_pooled = ref_inter["pooled"].numpy()

    # Compute sum embeddings for reference model
    ref_sum_embeddings = np.zeros((p, d_model))
    for s in range(p):
        mask = sum_labels_arr == s
        ref_sum_embeddings[s] = ref_pooled[mask].mean(axis=0)

    # Find best cos/sin dimension pair
    s_values = np.arange(p)
    best_k = 1
    best_cos_dim = 0
    best_sin_dim = 1
    best_total_corr = 0

    for k in range(1, min(p // 4, 20) + 1):
        cos_basis = np.cos(2 * np.pi * k * s_values / p)
        sin_basis = np.sin(2 * np.pi * k * s_values / p)

        cos_corrs = []
        sin_corrs = []
        for d in range(d_model):
            dim_vals = ref_sum_embeddings[:, d]
            if np.std(dim_vals) > 0.01:
                cc = np.corrcoef(dim_vals, cos_basis)[0, 1]
                sc = np.corrcoef(dim_vals, sin_basis)[0, 1]
                cos_corrs.append((d, cc if not np.isnan(cc) else 0))
                sin_corrs.append((d, sc if not np.isnan(sc) else 0))
            else:
                cos_corrs.append((d, 0))
                sin_corrs.append((d, 0))

        cos_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        sin_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        cos_dim = cos_corrs[0][0]
        cos_val = abs(cos_corrs[0][1])
        sin_dim = sin_corrs[0][0] if sin_corrs[0][0] != cos_dim else sin_corrs[1][0]
        sin_val = abs(sin_corrs[0][1]) if sin_corrs[0][0] != cos_dim else abs(sin_corrs[1][1])

        total_corr = cos_val + sin_val
        if total_corr > best_total_corr:
            best_total_corr = total_corr
            best_k = k
            best_cos_dim = cos_dim
            best_sin_dim = sin_dim

    # Find top Fourier-correlated dims
    fourier_scores = []
    for dim in range(d_model):
        dim_vals = ref_sum_embeddings[:, dim]
        best_corr = 0
        for k in range(1, min(p // 4, 20) + 1):
            cos_basis = np.cos(2 * np.pi * k * s_values / p)
            sin_basis = np.sin(2 * np.pi * k * s_values / p)
            if np.std(dim_vals) > 0.01:
                cos_corr = abs(np.corrcoef(dim_vals, cos_basis)[0, 1])
                sin_corr = abs(np.corrcoef(dim_vals, sin_basis)[0, 1])
                if not np.isnan(cos_corr):
                    best_corr = max(best_corr, cos_corr)
                if not np.isnan(sin_corr):
                    best_corr = max(best_corr, sin_corr)
        fourier_scores.append(best_corr)

    top_indices = np.argsort(fourier_scores)[::-1][:10]
    fixed_dims = top_indices.tolist()

    # Correlation sample inputs
    np.random.seed(42)
    corr_pairs = [[np.random.randint(p), np.random.randint(p)] for _ in range(100)]
    if n_tokens == 2:
        inputs_corr = torch.tensor(corr_pairs)
    else:
        inputs_corr = torch.tensor([[a, b, 0] for a, b in corr_pairs])

    # Now compute data for each sampled epoch
    all_proj_2d = []
    all_pooled = []
    all_angle_corr = []
    valid_epochs = []

    for i, ep in enumerate(sampled_epochs):
        ep_path = get_epoch_path(checkpoint_dir, ep)
        if ep_path is None:
            continue
        try:
            ep_model, _, _ = load_model_from_checkpoint(ep_path)

            with torch.no_grad():
                _, inter_circle = ep_model.forward_with_intermediates(inputs_circle)
                _, inter_corr = ep_model.forward_with_intermediates(inputs_corr)

            pooled_all = inter_circle["pooled"].numpy()
            pooled_corr = inter_corr["pooled"].numpy()

            # Compute sum embeddings
            sum_embeddings = np.zeros((p, pooled_all.shape[1]))
            for s in range(p):
                mask = sum_labels_arr == s
                sum_embeddings[s] = pooled_all[mask].mean(axis=0)

            # Project using fixed dims
            proj_2d = sum_embeddings[:, [best_cos_dim, best_sin_dim]]

            # Compute angle correlation
            center = proj_2d.mean(axis=0)
            centered = proj_2d - center
            angles = np.arctan2(centered[:, 1], centered[:, 0])
            expected_angles = 2 * np.pi * best_k * np.arange(p) / p - np.pi

            best_angle_corr = 0
            for shift in range(p):
                shifted_expected = np.roll(expected_angles, shift)
                corr = np.corrcoef(angles, shifted_expected)[0, 1]
                if not np.isnan(corr):
                    best_angle_corr = max(best_angle_corr, abs(corr))

            # Correlation matrix data
            pooled_sampled = pooled_corr[:, fixed_dims[:10]] if fixed_dims else pooled_corr[:, :10]

            all_proj_2d.append(proj_2d)
            all_pooled.append(pooled_sampled)
            all_angle_corr.append(best_angle_corr)
            valid_epochs.append(ep)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(sampled_epochs)} epochs")
        except Exception as e:
            print(f"    Warning: Failed to process epoch {ep}: {e}")
            continue

    if not valid_epochs:
        return None, None

    # Stack arrays
    proj_2d_stack = np.stack(all_proj_2d)  # (n_epochs, p, 2)
    pooled_stack = np.stack(all_pooled)     # (n_epochs, 100, 10)
    angle_corr_arr = np.array(all_angle_corr)  # (n_epochs,)

    # Compute axis ranges for fixed axes
    all_proj_x = proj_2d_stack[:, :, 0].flatten()
    all_proj_y = proj_2d_stack[:, :, 1].flatten()

    def calc_range(data):
        mn, mx = float(data.min()), float(data.max())
        margin = (mx - mn) * 0.1 + 0.01
        return [mn - margin, mx + margin]

    proj_x_range = calc_range(all_proj_x)
    proj_y_range = calc_range(all_proj_y)

    pooled_ranges = []
    grid_size = min(7, pooled_stack.shape[2])
    for i in range(grid_size):
        pooled_ranges.append(calc_range(pooled_stack[:, :, i].flatten()))

    meta = {
        "fixed_dims": fixed_dims,
        "best_pair": {
            "k": int(best_k),
            "cos_dim": int(best_cos_dim),
            "sin_dim": int(best_sin_dim),
            "total_corr": float(best_total_corr),
        },
        "proj_x_range": proj_x_range,
        "proj_y_range": proj_y_range,
        "pooled_ranges": pooled_ranges,
        "sampled_epochs": valid_epochs,
    }

    return {
        "proj_2d": proj_2d_stack,
        "pooled": pooled_stack,
        "angle_corr": angle_corr_arr,
    }, meta


def main():
    parser = argparse.ArgumentParser(description="Precompute data for Streamlit Cloud")
    parser.add_argument("--checkpoint-dir", default="checkpoints_demo_2ep",
                        help="Checkpoint directory")
    parser.add_argument("--output-dir", default="precomputed_data",
                        help="Output directory for precomputed data")
    parser.add_argument("--max-animation-frames", type=int, default=30,
                        help="Maximum animation frames for Tab 5")
    parser.add_argument("--max-fourier-epochs", type=int, default=50,
                        help="Maximum epochs for Fourier history (Tab 3)")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir

    # Validate
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(best_path):
        print(f"ERROR: {best_path} not found")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load best model
    print(f"Loading model from {best_path}...")
    model, config, epoch = load_model_from_checkpoint(best_path)
    print(f"  Model loaded: p={config['p']}, d_model={config['d_model']}, epoch={epoch}")

    # Save config
    config_out = {k: v for k, v in config.items()}
    config_out["best_epoch"] = epoch
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)
    print("  Saved config.json")

    # Copy history.json
    history_path = os.path.join(checkpoint_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(history, f)
        print(f"  Copied history.json ({len(history.get('train_acc', []))} epochs)")
    else:
        print("  WARNING: history.json not found")

    # Tab 2: Fourier Analysis
    weights, spectrum, proj_2d, tab2_meta = compute_tab2_fourier(model, config)
    np.savez_compressed(
        os.path.join(output_dir, "tab2_fourier.npz"),
        weights=weights,
        spectrum=spectrum,
        proj_2d=proj_2d,
    )
    with open(os.path.join(output_dir, "tab2_meta.json"), "w") as f:
        json.dump(tab2_meta, f, indent=2)
    print("  Saved tab2_fourier.npz + tab2_meta.json")

    # Tab 3: Fourier History
    fourier_history = compute_tab3_fourier_history(
        checkpoint_dir, config, max_epochs=args.max_fourier_epochs
    )
    if fourier_history:
        with open(os.path.join(output_dir, "tab3_fourier_history.json"), "w") as f:
            json.dump(_sanitize_for_json(fourier_history), f)
        print("  Saved tab3_fourier_history.json")

    # Tab 4: Model Output
    pred_matrix, logit_matrix, full_logits = compute_tab4_output(model, config)
    np.savez_compressed(
        os.path.join(output_dir, "tab4_output.npz"),
        pred_matrix=pred_matrix,
        logit_matrix=logit_matrix,
        full_logits=full_logits,
    )
    print("  Saved tab4_output.npz")

    # Tab 5: Animation
    tab5_data, tab5_meta = compute_tab5_animation(
        checkpoint_dir, config, max_frames=args.max_animation_frames
    )
    if tab5_data and tab5_meta:
        np.savez_compressed(
            os.path.join(output_dir, "tab5_animation.npz"),
            proj_2d=tab5_data["proj_2d"],
            pooled=tab5_data["pooled"],
            angle_corr=tab5_data["angle_corr"],
        )
        with open(os.path.join(output_dir, "tab5_meta.json"), "w") as f:
            json.dump(_sanitize_for_json(tab5_meta), f, indent=2)
        print("  Saved tab5_animation.npz + tab5_meta.json")

    # Tab 7: Attention
    attn_maps, attn_a_vals, attn_b_vals, embeddings = compute_tab7_attention(model, config)
    np.savez_compressed(
        os.path.join(output_dir, "tab7_attention.npz"),
        attention_maps=attn_maps,
        a_vals=np.array(attn_a_vals),
        b_vals=np.array(attn_b_vals),
    )
    np.savez_compressed(
        os.path.join(output_dir, "tab7_embeddings.npz"),
        embeddings=embeddings,
    )
    print("  Saved tab7_attention.npz + tab7_embeddings.npz")

    # Tab 8: Neurons
    result = compute_tab8_neurons(model, config)
    if result[0] is not None:
        corr_matrix, neuron_acts, activations_2d, neuron_a_vals, neuron_b_vals = result
        save_dict = {
            "neuron_corr": corr_matrix,
            "activations": neuron_acts,
        }
        if activations_2d is not None:
            save_dict["activations_2d"] = activations_2d
            save_dict["a_vals"] = np.array(neuron_a_vals)
            save_dict["b_vals"] = np.array(neuron_b_vals)
        np.savez_compressed(
            os.path.join(output_dir, "tab8_neurons.npz"),
            **save_dict,
        )
        print("  Saved tab8_neurons.npz")

    # Summary
    print("\n" + "=" * 60)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 60)

    total_size = 0
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        total_size += size
        print(f"  {f}: {size / 1024:.1f} KB")

    print(f"\n  Total: {total_size / (1024 * 1024):.2f} MB")
    print(f"  Output directory: {output_dir}/")


if __name__ == "__main__":
    main()

"""Visualize the top-k neuron activation values for each layer of the TSN video encoder and BYOL-A audio encoder.

This script is intended to be run after the neuron probing step in README step 4. It loads the
``*_neuron_scores.npz`` output from the probing stage, selects the top-k channel indices for each
layer, collects the actual activation values from the encoder on the probe pairs, and saves heatmaps
showing the mean activation values of those neurons.

Examples:

  # Video encoder (TSN)
  py -3 -m UniCaCLF.visualize_neuron_values \
      --modality video \
      --pairs ../deepfake-data/manifests/video_only_probe_pairs.jsonl \
      --rgb-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth \
      --flow-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth \
      --scores ../deepfake-data/results/unicaclf_probe/tsn/tsn_neuron_scores.npz \
      --output-dir ../deepfake-data/results/unicaclf_probe/tsn_vis \
      --max-pairs 20 --top-k 10

  # Audio encoder (BYOL-A)
  py -3 -m UniCaCLF.visualize_neuron_values \
      --modality audio \
      --pairs ../deepfake-data/manifests/audio_only_probe_pairs.jsonl \
      --byola-repo byol-a \
      --checkpoint ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth \
      --norm-stats ../deepfake-data/manifests/byola_train_stats.json \
      --scores ../deepfake-data/results/unicaclf_probe/byola/byola_neuron_scores.npz \
      --output-dir ../deepfake-data/results/unicaclf_probe/byola_vis \
      --max-pairs 20 --top-k 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

try:
    from .probe_common import read_pairs
    from .probe_tsn_neurons import build_tsn, flow_input, rgb_input, register_bottleneck_hooks, sampled_times
    from .probe_byola_neurons import decode_audio, load_byola, load_stats, register_hooks as register_audio_hooks, logmel_pair
except ImportError:  # allows direct execution from UniCaCLF/
    from probe_common import read_pairs
    from probe_tsn_neurons import build_tsn, flow_input, rgb_input, register_bottleneck_hooks, sampled_times
    from probe_byola_neurons import decode_audio, load_byola, load_stats, register_hooks as register_audio_hooks, logmel_pair


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    candidates = [path, Path.cwd() / path]
    base = Path(__file__).resolve().parent
    for parent in [base, *base.parents]:
        candidates.append(parent / path)
        candidates.append(parent / path.as_posix())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def load_top_indices(path: Path, top_k: int) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    layers: dict[str, np.ndarray] = {}
    for key in sorted(data.files):
        if key.endswith("_top_indices"):
            layer_name = key[: -len("_top_indices")]
            indices = np.asarray(data[key], dtype=np.int64)
            if indices.size:
                layers[layer_name] = indices[:top_k]
    if not layers:
        raise ValueError(f"No top indices found in {path}")
    return layers


class LayerValueCollector:
    def __init__(self, layer_names: list[str]):
        self.layer_names = layer_names
        self.real_values: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
        self.fake_values: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}

    def add(self, layer_name: str, real: np.ndarray, fake: np.ndarray) -> None:
        self.real_values[layer_name].append(real)
        self.fake_values[layer_name].append(fake)

    def summarize(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        real_mean = {
            name: np.mean(np.stack(self.real_values[name], axis=0), axis=0)
            if self.real_values[name]
            else np.zeros(0, dtype=np.float32)
            for name in self.layer_names
        }
        fake_mean = {
            name: np.mean(np.stack(self.fake_values[name], axis=0), axis=0)
            if self.fake_values[name]
            else np.zeros(0, dtype=np.float32)
            for name in self.layer_names
        }
        return real_mean, fake_mean


def _pooled_channels(activation: torch.Tensor) -> np.ndarray:
    if activation.ndim == 4:
        return activation.float().mean(dim=(2, 3)).cpu().numpy()
    if activation.ndim == 3:
        return activation.float().mean(dim=1).cpu().numpy()
    if activation.ndim == 2:
        return activation.float().cpu().numpy()
    raise ValueError(f"Unsupported activation tensor shape {tuple(activation.shape)}")


def collect_video_neuron_values(
    model: nn.Module,
    captured: dict[str, torch.Tensor | None],
    pairs: list,
    top_indices: dict[str, np.ndarray],
    device: torch.device,
    amp: bool,
    image_size: int,
    flow_method: str,
    flow_bound: float,
    decode_threads: int,
    samples_per_period: int,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    collector = LayerValueCollector(list(top_indices))
    for pair in tqdm(pairs, desc=f"Collecting {prefix} neuron values", leave=False):
        times = sampled_times(pair.start_sec, pair.end_sec, samples_per_period)
        if prefix == "rgb":
            real_input = rgb_input(pair.original_file, times, image_size, decode_threads)
            fake_input = rgb_input(pair.fake_file, times, image_size, decode_threads)
        else:
            real_input = flow_input(pair.original_file, times, image_size, flow_method, flow_bound, pair.start_sec, pair.end_sec, decode_threads)
            fake_input = flow_input(pair.fake_file, times, image_size, flow_method, flow_bound, pair.start_sec, pair.end_sec, decode_threads)

        for key in captured:
            captured[key] = None
        batch = torch.cat((real_input, fake_input), dim=0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            _ = model(batch)

        for layer_name, activation in captured.items():
            if activation is None:
                continue
            pooled = _pooled_channels(activation)
            if pooled.ndim != 2:
                raise RuntimeError(f"Unexpected pooled shape for {layer_name}: {pooled.shape}")
            length = real_input.shape[0]
            real_vec = pooled[:length]
            fake_vec = pooled[length:]
            if real_vec.shape[0] != fake_vec.shape[0]:
                raise RuntimeError(f"Mismatch when splitting pooled activations for {layer_name}: {real_vec.shape} vs {fake_vec.shape}")
            real_mean = real_vec.mean(axis=0)
            fake_mean = fake_vec.mean(axis=0)
            if layer_name in top_indices:
                collector.add(layer_name, real_mean[top_indices[layer_name]], fake_mean[top_indices[layer_name]])
    return collector.summarize()


def collect_audio_neuron_values(
    model: nn.Module,
    captured: dict[str, torch.Tensor | None],
    pairs: list,
    top_indices: dict[str, np.ndarray],
    device: torch.device,
    amp: bool,
    transform: nn.Module,
    mean: float,
    std: float,
    sample_rate: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    collector = LayerValueCollector(list(top_indices))
    for pair in tqdm(pairs, desc="Collecting audio neuron values", leave=False):
        real_wave = decode_audio(pair.original_file, pair.start_sec, pair.end_sec, sample_rate)
        fake_wave = decode_audio(pair.fake_file, pair.start_sec, pair.end_sec, sample_rate)
        pair_input = logmel_pair(real_wave, fake_wave, transform, mean, std, device)
        for key in captured:
            captured[key] = None
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            _ = model(pair_input)
        for layer_name, activation in captured.items():
            if activation is None:
                continue
            pooled = _pooled_channels(activation)
            if pooled.ndim != 2:
                raise RuntimeError(f"Unexpected pooled shape for {layer_name}: {pooled.shape}")
            real_mean = pooled[0]
            fake_mean = pooled[1]
            if layer_name in top_indices:
                collector.add(layer_name, real_mean[top_indices[layer_name]], fake_mean[top_indices[layer_name]])
    return collector.summarize()


def save_heatmaps(output_dir: Path, layer_names: list[str], top_indices: dict[str, np.ndarray], real_mean: dict[str, np.ndarray], fake_mean: dict[str, np.ndarray], modality: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_layers = [name for name in layer_names if name in real_mean and name in fake_mean]
    if not ordered_layers:
        raise ValueError("No layers were collected")

    display_k = 10
    real_matrix = []
    fake_matrix = []
    delta_matrix = []

    for layer_name in ordered_layers:
        indices = top_indices[layer_name]
        real_values = real_mean[layer_name]
        fake_values = fake_mean[layer_name]
        if real_values.size < len(indices):
            raise ValueError(f"Layer {layer_name} had only {real_values.size} values but expected {len(indices)}")
        width = min(len(indices), real_values.size, display_k)
        real_row = real_values[:width].astype(np.float32)
        fake_row = fake_values[:width].astype(np.float32)
        delta_row = (fake_values[:width] - real_values[:width]).astype(np.float32)
        if width < display_k:
            real_row = np.pad(real_row, (0, display_k - width), mode="constant", constant_values=0.0)
            fake_row = np.pad(fake_row, (0, display_k - width), mode="constant", constant_values=0.0)
            delta_row = np.pad(delta_row, (0, display_k - width), mode="constant", constant_values=0.0)
        real_matrix.append(real_row)
        fake_matrix.append(fake_row)
        delta_matrix.append(delta_row)

    real_matrix = np.stack(real_matrix, axis=0)
    fake_matrix = np.stack(fake_matrix, axis=0)
    delta_matrix = np.stack(delta_matrix, axis=0)

    def plot_matrix(matrix: np.ndarray, title: str, save_name: str, cmap: str = "viridis") -> None:
        fig, axis = plt.subplots(figsize=(max(8, 0.8 * display_k), max(3.2, 0.55 * len(ordered_layers))), dpi=180)
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_xticks(np.arange(display_k))
        axis.set_xticklabels([f"n{idx + 1}" for idx in range(display_k)], rotation=45, ha="right")
        axis.set_yticks(np.arange(len(ordered_layers)))
        axis.set_yticklabels(ordered_layers)
        axis.set_title(title)
        axis.set_xlabel("top-k neurons")
        axis.set_ylabel("layer")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.5 else "black")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / save_name, bbox_inches="tight")
        plt.close(fig)

    plot_matrix(real_matrix, f"{modality.title()} encoder: mean activation (real)", f"{modality}_real_activation_heatmap.png")
    plot_matrix(fake_matrix, f"{modality.title()} encoder: mean activation (fake)", f"{modality}_fake_activation_heatmap.png")
    plot_matrix(delta_matrix, f"{modality.title()} encoder: fake - real activation delta", f"{modality}_delta_activation_heatmap.png", cmap="coolwarm")

    export_rows = []
    for layer_name in ordered_layers:
        indices = top_indices[layer_name]
        real_values = real_mean[layer_name]
        fake_values = fake_mean[layer_name]
        for rank, channel_index in enumerate(indices, 1):
            export_rows.append({
                "layer": layer_name,
                "rank": rank,
                "channel_index": int(channel_index),
                "real_mean": float(real_values[rank - 1]),
                "fake_mean": float(fake_values[rank - 1]),
                "delta": float(fake_values[rank - 1] - real_values[rank - 1]),
            })
    (output_dir / f"{modality}_top_neuron_values.csv").write_text(
        "layer,rank,channel_index,real_mean,fake_mean,delta\n" + "\n".join(
            f"{row['layer']},{row['rank']},{row['channel_index']},{row['real_mean']:.6f},{row['fake_mean']:.6f},{row['delta']:.6f}"
            for row in export_rows
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("video", "audio"), required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="Path to the probing output .npz file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--rgb-checkpoint", type=Path)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples-per-period", type=int, default=16)
    parser.add_argument("--flow-method", choices=("tvl1", "farneback"), default="tvl1")
    parser.add_argument("--flow-bound", type=float, default=20.0)
    parser.add_argument("--decode-threads", type=int, default=1)

    parser.add_argument("--byola-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=60)
    parser.add_argument("--f-max", type=float, default=7800)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    device = torch.device(args.device)
    args.pairs = resolve_path(args.pairs)
    args.scores = resolve_path(args.scores)
    args.output_dir = resolve_path(args.output_dir)
    args.rgb_checkpoint = resolve_path(args.rgb_checkpoint)
    args.flow_checkpoint = resolve_path(args.flow_checkpoint)
    args.byola_repo = resolve_path(args.byola_repo)
    args.checkpoint = resolve_path(args.checkpoint)
    args.norm_stats = resolve_path(args.norm_stats)
    pairs = read_pairs(args.pairs, "video" if args.modality == "video" else "audio")
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        generator = np.random.default_rng(args.sample_seed)
        indices = np.sort(generator.choice(len(pairs), size=args.max_pairs, replace=False))
        pairs = [pairs[int(index)] for index in indices]

    top_indices = load_top_indices(args.scores, args.top_k)

    if args.modality == "video":
        if not args.rgb_checkpoint or not args.flow_checkpoint:
            raise ValueError("Video mode requires --rgb-checkpoint and --flow-checkpoint")
        rgb_model = build_tsn(args.rgb_checkpoint, "RGB", device)
        flow_model = build_tsn(args.flow_checkpoint, "Flow", device)
        rgb_hidden, rgb_handles = register_bottleneck_hooks(rgb_model, "rgb")
        flow_hidden, flow_handles = register_bottleneck_hooks(flow_model, "flow")
        try:
            rgb_real, rgb_fake = collect_video_neuron_values(
                rgb_model,
                rgb_hidden,
                pairs,
                {name: value for name, value in top_indices.items() if name.startswith("rgb_")},
                device,
                args.amp,
                args.image_size,
                args.flow_method,
                args.flow_bound,
                args.decode_threads,
                args.samples_per_period,
                "rgb",
            )
            flow_real, flow_fake = collect_video_neuron_values(
                flow_model,
                flow_hidden,
                pairs,
                {name: value for name, value in top_indices.items() if name.startswith("flow_")},
                device,
                args.amp,
                args.image_size,
                args.flow_method,
                args.flow_bound,
                args.decode_threads,
                args.samples_per_period,
                "flow",
            )
            real_mean = {**rgb_real, **flow_real}
            fake_mean = {**rgb_fake, **flow_fake}
            save_heatmaps(args.output_dir, list(top_indices.keys()), top_indices, real_mean, fake_mean, "video")
        finally:
            for handle in rgb_handles + flow_handles:
                handle.remove()
    else:
        if not args.byola_repo or not args.checkpoint or not args.norm_stats:
            raise ValueError("Audio mode requires --byola-repo, --checkpoint and --norm-stats")
        mean, std = load_stats(args.norm_stats)
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=args.n_fft, win_length=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels, f_min=args.f_min, f_max=args.f_max).to(device)
        model = load_byola(args.byola_repo, args.checkpoint, device, args.feature_dim)
        captured, handles = register_audio_hooks(model)
        try:
            real_mean, fake_mean = collect_audio_neuron_values(
                model,
                captured,
                pairs,
                top_indices,
                device,
                args.amp,
                transform,
                mean,
                std,
                args.sample_rate,
            )
            save_heatmaps(args.output_dir, list(top_indices.keys()), top_indices, real_mean, fake_mean, "audio")
        finally:
            for handle in handles:
                handle.remove()

    print(f"Saved visualization figures and CSV to {args.output_dir}")


if __name__ == "__main__":
    main()

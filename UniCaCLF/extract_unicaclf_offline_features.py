"""Extract frozen TSN/BYOL-A final or selected-neuron sequences for UniCaCLF.

The output deliberately follows UniCaCLF's legacy offline layout:

  <output-root>/tsn/rgb/<split>/<id>.npy
  <output-root>/tsn/flow/<split>/<id>.npy
  <output-root>/byola/<split>/<id>.npy

For ``--representation final`` these are TSN Res5 global-pooled features
(2048 per stream) and temporal BYOL-A FC2 features (2048).  For
``--representation neurons`` they are, respectively, the concatenation of
top-rho selected TSN RGB/Flow block channels and top-rho selected BYOL-A
channels.  Every selected layer is spatial/frequency pooled, interpolated to
the TSN time grid, temporally channel-normalised, then concatenated.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from decord import VideoReader, cpu
from tqdm import tqdm

try:
    from .probe_byola_neurons import decode_audio, load_byola, load_stats, register_hooks as register_audio_hooks
    from .probe_tsn_neurons import build_tsn, flow_input, register_bottleneck_hooks, rgb_input
    from .distributed_utils import cleanup_distributed, init_distributed, is_distributed, is_main_process
except ImportError:
    from probe_byola_neurons import decode_audio, load_byola, load_stats, register_hooks as register_audio_hooks
    from probe_tsn_neurons import build_tsn, flow_input, register_bottleneck_hooks, rgb_input
    from distributed_utils import cleanup_distributed, init_distributed, is_distributed, is_main_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lavdf-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True, help="Fixed 12k/2k/4k metadata subset JSON")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--representation", choices=("final", "neurons", "both"), required=True,
                        help="both extracts final and selected-neuron caches in one encoder pass")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--rgb-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--byola-repo", type=Path, required=True)
    parser.add_argument("--byola-checkpoint", type=Path, required=True)
    parser.add_argument("--byola-norm-stats", type=Path, required=True)
    parser.add_argument("--tsn-scores", type=Path, help="Required for --representation neurons")
    parser.add_argument("--byola-scores", type=Path, help="Required for --representation neurons")
    parser.add_argument("--video-stride-frames", type=int, default=4)
    parser.add_argument("--video-batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--flow-method", choices=("tvl1", "farneback"), default="tvl1")
    parser.add_argument("--flow-bound", type=float, default=20.0)
    parser.add_argument("--decode-threads", type=int, default=2, help="Video-decoder threads per rank")
    parser.add_argument("--max-items", type=int, default=None, help="Debug limit per selected split")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_indices(path: Path, suffix: str) -> OrderedDict[str, np.ndarray]:
    result: OrderedDict[str, np.ndarray] = OrderedDict()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            if key.endswith("_top_indices"):
                result[key.removesuffix("_top_indices")] = np.asarray(data[key], dtype=np.int64)
    if not result:
        raise ValueError(f"No *_top_indices in {path}")
    if suffix == "tsn" and not any(name.startswith("rgb_") for name in result):
        raise ValueError(f"{path} has no TSN RGB selections")
    return result


def temporal_standardise(sequence: torch.Tensor) -> torch.Tensor:
    """Per-video, per-channel normalisation after top-channel selection."""
    mean = sequence.mean(dim=0, keepdim=True)
    std = sequence.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    return (sequence - mean) / std


def interpolate(sequence: torch.Tensor, length: int) -> torch.Tensor:
    if sequence.shape[0] == length:
        return sequence
    return F.interpolate(sequence.T.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0).T


def video_time_grid(path: Path, stride: int, decode_threads: int) -> tuple[np.ndarray, float]:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=decode_threads)
    frames, fps = len(reader), float(reader.get_avg_fps())
    if frames < 1 or fps <= 0:
        raise ValueError(f"Unreadable video: {path}")
    # The centre is an actual RGB frame.  The same seconds are used for audio.
    centers = np.arange(0, frames, stride, dtype=np.int64)
    return centers.astype(np.float64) / fps, frames / fps


def run_tsn_stream(model, captured, tensor: torch.Tensor, device: torch.device, amp: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    for key in captured:
        captured[key] = None
    chunks, final = [], []
    # Caller already batches the input; this helper exists for one batch only.
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        value = model(tensor.to(device, non_blocking=True))
    final.append(value.float().cpu())
    hidden = {}
    for key, value in captured.items():
        if value is None:
            raise RuntimeError(f"No activation captured for {key}")
        hidden[key] = value.float().mean(dim=(2, 3)).cpu()  # spatial mean: T,C
    return torch.cat(final), hidden


def extract_tsn(path: Path, rgb_model, flow_model, rgb_hooks, flow_hooks, args, device):
    times, duration = video_time_grid(path, args.video_stride_frames, args.decode_threads)
    rgb_all, flow_all = [], []
    rgb_hidden: dict[str, list[torch.Tensor]] = {key: [] for key in rgb_hooks}
    flow_hidden: dict[str, list[torch.Tensor]] = {key: [] for key in flow_hooks}
    for begin in range(0, len(times), args.video_batch_size):
        now = times[begin:begin + args.video_batch_size]
        rgb, rgb_layers = run_tsn_stream(
            rgb_model, rgb_hooks,
            rgb_input(str(path), now, args.image_size, decode_threads=args.decode_threads),
            device, args.amp,
        )
        flow, flow_layers = run_tsn_stream(
            flow_model, flow_hooks,
            flow_input(
                str(path), now, args.image_size, args.flow_method, args.flow_bound, 0.0, duration,
                decode_threads=args.decode_threads,
            ),
            device, args.amp,
        )
        rgb_all.append(rgb); flow_all.append(flow)
        for key, value in rgb_layers.items(): rgb_hidden[key].append(value)
        for key, value in flow_layers.items(): flow_hidden[key].append(value)
    return (
        torch.cat(rgb_all), torch.cat(flow_all),
        {key: torch.cat(value) for key, value in rgb_hidden.items()},
        {key: torch.cat(value) for key, value in flow_hidden.items()},
        times.astype(np.float32), duration,
    )


def extract_audio(path: Path, duration: float, model, hooks, mean: float, std: float, target_len: int, args, device):
    wave = decode_audio(str(path), 0.0, duration, 16000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
        n_mels=64, f_min=60, f_max=7800,
    ).to(device)
    value = ((mel(wave.unsqueeze(0).to(device)) + torch.finfo(torch.float32).eps).log() - mean) / std
    for key in hooks: hooks[key] = None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
        final = model(value.unsqueeze(1))
    hidden = {}
    for key, activation in hooks.items():
        if activation is None: raise RuntimeError(f"No activation captured for {key}")
        hidden[key] = activation.float().mean(dim=2).squeeze(0).cpu() if key.startswith("audio_conv") else activation.float().squeeze(0).cpu()
    return interpolate(final.float().squeeze(0).cpu(), target_len), {key: interpolate(x, target_len) for key, x in hidden.items()}


def selected_concat(hidden: dict[str, torch.Tensor], indices: OrderedDict[str, np.ndarray], prefix: str, target_len: int) -> torch.Tensor:
    values = []
    for name, channels in indices.items():
        if not name.startswith(prefix):
            continue
        if name not in hidden:
            raise KeyError(f"Selected layer {name} is absent from encoder hooks")
        value = interpolate(hidden[name], target_len)
        values.append(temporal_standardise(value[:, torch.from_numpy(channels)]))
    if not values:
        raise ValueError(f"No selected layers with prefix {prefix}")
    return torch.cat(values, dim=1)


def main() -> None:
    args = parse_args()
    if args.representation in {"neurons", "both"} and (args.tsn_scores is None or args.byola_scores is None):
        raise ValueError("--tsn-scores and --byola-scores are required for --representation neurons/both")
    if args.video_stride_frames < 1 or args.video_batch_size < 1 or args.decode_threads < 1:
        raise ValueError("--video-stride-frames, --video-batch-size and --decode-threads must be positive")
    device, rank, world_size = init_distributed(args.device)
    tsn_indices = load_indices(args.tsn_scores, "tsn") if args.representation in {"neurons", "both"} else None
    audio_indices = load_indices(args.byola_scores, "byola") if args.representation in {"neurons", "both"} else None
    final_root = args.output_root if args.representation == "final" else args.output_root / "final"
    neuron_root = args.output_root if args.representation == "neurons" else args.output_root / "neurons"
    rgb_model = build_tsn(args.rgb_checkpoint, "RGB", device)
    flow_model = build_tsn(args.flow_checkpoint, "Flow", device)
    rgb_hooks, rgb_handles = register_bottleneck_hooks(rgb_model, "rgb")
    flow_hooks, flow_handles = register_bottleneck_hooks(flow_model, "flow")
    mean, std = load_stats(args.byola_norm_stats)
    audio_model = load_byola(args.byola_repo, args.byola_checkpoint, device, 2048)
    audio_hooks, audio_handles = register_audio_hooks(audio_model)
    records = json.loads(args.subset.read_text(encoding="utf-8"))
    records = [x for x in records if x.get("split") in args.splits]
    failures = []
    processed = 0
    try:
        for split in args.splits:
            split_records = [x for x in records if x.get("split") == split]
            if args.max_items is not None: split_records = split_records[:args.max_items]
            # Apply max-items before sharding: it is a global debug limit, not
            # an accidental max-items-per-GPU multiplier.
            split_records = split_records[rank::world_size]
            for item in tqdm(
                split_records, desc=f"Extract {args.representation} {split}",
                disable=not is_main_process(),
            ):
                key = Path(item["file"]).stem
                final_output = [final_root / "tsn" / "rgb" / split / f"{key}.npy", final_root / "tsn" / "flow" / split / f"{key}.npy", final_root / "byola" / split / f"{key}.npy"]
                neuron_output = [neuron_root / "tsn" / "rgb" / split / f"{key}.npy", neuron_root / "tsn" / "flow" / split / f"{key}.npy", neuron_root / "byola" / split / f"{key}.npy"]
                wanted = ([] if args.representation == "neurons" else final_output) + ([] if args.representation == "final" else neuron_output)
                if all(x.is_file() for x in wanted) and not args.overwrite: continue
                try:
                    path = args.lavdf_root / item["file"]
                    rgb, flow, rgb_hidden, flow_hidden, times, duration = extract_tsn(path, rgb_model, flow_model, rgb_hooks, flow_hooks, args, device)
                    audio, audio_hidden = extract_audio(path, duration, audio_model, audio_hooks, mean, std, len(times), args, device)
                    if args.representation in {"final", "both"}:
                        for path_out, value_out in zip(final_output, (rgb, flow, audio)):
                            path_out.parent.mkdir(parents=True, exist_ok=True)
                            np.save(path_out, value_out.numpy().astype(np.float32))
                    if args.representation in {"neurons", "both"}:
                        values = (selected_concat(rgb_hidden, tsn_indices, "rgb_", len(times)),
                                  selected_concat(flow_hidden, tsn_indices, "flow_", len(times)),
                                  selected_concat(audio_hidden, audio_indices, "audio_", len(times)))
                        for path_out, value_out in zip(neuron_output, values):
                            path_out.parent.mkdir(parents=True, exist_ok=True)
                            np.save(path_out, value_out.numpy().astype(np.float32))
                    processed += 1
                except Exception as error:
                    failures.append(f"{item['file']}: {type(error).__name__}: {error}")
    finally:
        for handle in rgb_handles + flow_handles + audio_handles: handle.remove()

    if is_distributed():
        all_failures = [None] * dist.get_world_size()
        all_processed = [None] * dist.get_world_size()
        dist.all_gather_object(all_failures, failures)
        dist.all_gather_object(all_processed, processed)
        failures = [failure for local in all_failures for failure in local]
        processed = sum(all_processed)

    settings = {"representation": args.representation, "subset": str(args.subset), "video_stride_frames": args.video_stride_frames,
                "decode_threads_per_rank": args.decode_threads, "world_size": world_size,
                "tsn_score_file": str(args.tsn_scores) if args.tsn_scores else None, "byola_score_file": str(args.byola_scores) if args.byola_scores else None}
    if args.representation == "final":
        settings.update(rgb_dim=2048, flow_dim=2048, video_dim=4096, audio_dim=2048)
    elif args.representation == "neurons":
        rgb_dim = sum(len(v) for name, v in tsn_indices.items() if name.startswith("rgb_"))
        flow_dim = sum(len(v) for name, v in tsn_indices.items() if name.startswith("flow_"))
        audio_dim = sum(len(v) for name, v in audio_indices.items() if name.startswith("audio_"))
        settings.update(rgb_dim=rgb_dim, flow_dim=flow_dim, video_dim=rgb_dim + flow_dim, audio_dim=audio_dim)
    else:
        rgb_dim = sum(len(v) for name, v in tsn_indices.items() if name.startswith("rgb_")); flow_dim = sum(len(v) for name, v in tsn_indices.items() if name.startswith("flow_")); audio_dim = sum(len(v) for name, v in audio_indices.items() if name.startswith("audio_"))
        settings.update(final_root=str(final_root), neuron_root=str(neuron_root), final_video_dim=4096, final_audio_dim=2048,
                        neuron_rgb_dim=rgb_dim, neuron_flow_dim=flow_dim, neuron_video_dim=rgb_dim + flow_dim, neuron_audio_dim=audio_dim)
    if is_distributed():
        # Ensure all ranks have finished writing their disjoint video files
        # before rank 0 declares the cache complete.
        dist.barrier()
    if is_main_process():
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "extraction_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        if failures:
            (args.output_root / "extraction_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
            print(f"Finished {processed} items with {len(failures)} failures; see {args.output_root / 'extraction_failures.txt'}")
        else:
            print(f"Finished {processed} items. Results: {args.output_root}")
    cleanup_distributed()


if __name__ == "__main__":
    main()

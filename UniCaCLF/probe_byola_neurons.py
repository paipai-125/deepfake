"""Online paired neuron probing for UniCaCLF's BYOL-A AudioNTT2020 encoder."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchaudio
from tqdm import tqdm

try:
    from .probe_common import RunningVectorStats, read_pairs, save_probe_results
    from .distributed_utils import audit_pair_coverage, cleanup_distributed, init_distributed, is_distributed, is_main_process
except ImportError:
    from probe_common import RunningVectorStats, read_pairs, save_probe_results
    from distributed_utils import audit_pair_coverage, cleanup_distributed, init_distributed, is_distributed, is_main_process


def load_stats(path: Path) -> tuple[float, float]:
    if path.suffix == ".npy":
        value = np.load(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict): value = [value["mean"], value["std"]]
    mean, std = map(float, value)
    if std <= 0: raise ValueError(f"Invalid BYOL-A std in {path}: {std}")
    return mean, std


def load_byola(repo: Path, checkpoint: Path, device: torch.device, dim: int):
    sys.path.insert(0, str(repo.resolve()))
    try:
        from byol_a.models import AudioNTT2020Feature as Encoder
        try: model = Encoder(d=dim)
        except TypeError: model = Encoder(n_mels=64, d=dim)
    except ImportError:
        # Official BYOL-A exposes this temporal representation before global
        # max+mean pooling as AudioNTT2020Task6.
        from byol_a.models import AudioNTT2020Task6 as Encoder
        model = Encoder(n_mels=64, d=dim)
    if not hasattr(model, "load_weight"):
        raise RuntimeError("The supplied BYOL-A repository has no AudioNTT2020 load_weight method")
    model.load_weight(str(checkpoint), device)
    return model.to(device).eval()


def register_hooks(model: nn.Module):
    captured: OrderedDict[str, torch.Tensor | None] = OrderedDict()
    handles = []
    features = getattr(model, "features", None)
    fc = getattr(model, "fc", None)
    if not isinstance(features, nn.Sequential) or not isinstance(fc, nn.Sequential):
        raise RuntimeError("Expected the official AudioNTT2020 Feature/Task6 model with .features and .fc Sequential modules")
    conv_index = 0
    for index, module in enumerate(features):
        if isinstance(module, nn.MaxPool2d):
            conv_index += 1; name = f"audio_conv{conv_index}"
            captured[name] = None
            handles.append(module.register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out.detach())))
    fc_index = 0
    for module in fc:
        if isinstance(module, nn.ReLU):
            fc_index += 1; name = f"audio_fc{fc_index}"
            captured[name] = None
            handles.append(module.register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out.detach())))
    if conv_index != 3 or fc_index != 2:
        raise RuntimeError(f"Unexpected AudioNTT2020 topology: found {conv_index} Conv blocks and {fc_index} FC activations")
    return captured, handles


def decode_audio(path: str, start: float, end: float, sample_rate: int) -> torch.Tensor:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None: raise RuntimeError("FFmpeg is required to decode MP4 audio")
    duration = end - start
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode: raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    wave = torch.frombuffer(bytearray(process.stdout), dtype=torch.float32).clone()
    target = round(duration * sample_rate)
    return torch.nn.functional.pad(wave[:target], (0, max(0, target - wave.numel())))


def logmel_pair(real: torch.Tensor, fake: torch.Tensor, transform, mean: float, std: float, device: torch.device) -> torch.Tensor:
    waves = torch.stack((real, fake)).to(device)
    # [B,T] -> [B,64,Tmel], then AudioNTT expects [B,1,64,Tmel].
    value = (transform(waves) + torch.finfo(torch.float32).eps).log()
    return ((value - mean) / std).unsqueeze(1)


def collect_deltas(model, captured, pair_input, device, amp) -> dict[str, np.ndarray]:
    """Return all layer deltas before mutating any Welford accumulator."""
    for key in captured: captured[key] = None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        _ = model(pair_input)
    deltas = {}
    for name, activation in captured.items():
        if activation is None: raise RuntimeError(f"BYOL-A hook did not capture {name}")
        if name.startswith("audio_conv"):
            # [2,C,F,T] -> frequency mean -> time mean -> [2,C]
            pooled = activation.float().mean(dim=(2, 3)).cpu().numpy()
        else:
            # [2,T,D] -> [2,D]
            pooled = activation.float().mean(dim=1).cpu().numpy()
        deltas[name] = pooled[1] - pooled[0]
    return deltas


def gather_stats(stats: dict[str, RunningVectorStats]) -> dict[str, RunningVectorStats] | None:
    """Gather rank-local Welford states and merge them once on rank 0."""
    if not is_distributed():
        return stats
    local = {name: state.export_state() for name, state in stats.items()}
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if not is_main_process():
        return None
    merged = {name: RunningVectorStats(state.mean.size) for name, state in stats.items()}
    for part in gathered:
        for name, (count, mean, m2) in part.items():
            merged[name].merge_state(count, mean, m2)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--byola-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True, help="JSON [mean,std] or NPY produced from the train subset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=60)
    parser.add_argument("--f-max", type=float, default=7800)
    parser.add_argument("--top-ratio", type=float, default=0.10)
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="Number of strict pairs to probe (default: 0 means every pair in the manifest)",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=2026,
        help="Fixed RNG seed used when --max-pairs selects a subset",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.top_ratio <= 1: parser.error("--top-ratio must be in (0,1]")
    if args.max_pairs < 0: parser.error("--max-pairs must be >= 0")
    device, rank, world_size = init_distributed(args.device)
    mean, std = load_stats(args.norm_stats)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=args.n_fft, win_length=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels, f_min=args.f_min, f_max=args.f_max).to(device)
    model = load_byola(args.byola_repo, args.checkpoint, device, args.feature_dim)
    captured, handles = register_hooks(model)
    stats = {name: RunningVectorStats(64 if name.startswith("audio_conv") else args.feature_dim) for name in captured}
    selected_pairs = read_pairs(args.pairs, "audio")
    if args.max_pairs > 0 and len(selected_pairs) > args.max_pairs:
        # Randomly select a reproducible global subset before rank sharding.
        # Sorting restores manifest order while preserving the sampled membership.
        generator = np.random.default_rng(args.sample_seed)
        indices = np.sort(generator.choice(len(selected_pairs), size=args.max_pairs, replace=False))
        selected_pairs = [selected_pairs[int(index)] for index in indices]
    expected_ids = [pair.pair_id for pair in selected_pairs]
    pairs = selected_pairs[rank::world_size]
    failures = []
    attempted_ids, success_ids, failed_ids = [], [], []
    try:
        for pair in tqdm(pairs, desc="BYOL-A paired neuron probe", disable=not is_main_process()):
            attempted_ids.append(pair.pair_id)
            try:
                real = decode_audio(pair.original_file, pair.start_sec, pair.end_sec, args.sample_rate)
                fake = decode_audio(pair.fake_file, pair.start_sec, pair.end_sec, args.sample_rate)
                deltas = collect_deltas(model, captured, logmel_pair(real, fake, transform, mean, std, device), device, args.amp)
                # Commit only after every layer has successfully produced its
                # delta, keeping all layer counts identical.
                for name, value in deltas.items(): stats[name].update(value)
                success_ids.append(pair.pair_id)
            except Exception as error:
                failures.append(f"{pair.pair_id}: {type(error).__name__}: {error}")
                failed_ids.append(pair.pair_id)
    finally:
        for handle in handles: handle.remove()
    merged = gather_stats(stats)
    if is_distributed():
        gathered_failures = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_failures, failures)
        failures = [value for part in gathered_failures for value in part]
    audit = audit_pair_coverage(expected_ids, attempted_ids, success_ids, failed_ids)
    if is_main_process():
        audit["sampling"] = {
            "max_pairs": args.max_pairs,
            "sample_seed": args.sample_seed if args.max_pairs > 0 else None,
            "method": "fixed_rng_without_replacement" if args.max_pairs > 0 else "all_pairs",
        }
        if audit["successful_pairs"]:
            counts = {state.count for state in merged.values()}
            if counts != {audit["successful_pairs"]}:
                raise RuntimeError(f"Non-atomic BYOL-A statistics: layer counts={counts}, audit={audit}")
            save_probe_results(args.output_dir, "byola", merged, args.top_ratio)
        if failures:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "byola_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "byola_run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"Processed {audit['successful_pairs']}/{audit['expected_pairs']} pairs. Results: {args.output_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()

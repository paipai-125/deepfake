"""Compute train-subset log-Mel mean/std required by BYOL-A feature extraction."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
from tqdm import tqdm

try:
    from .distributed_utils import cleanup_distributed, init_distributed, is_main_process
except ImportError:
    from distributed_utils import cleanup_distributed, init_distributed, is_main_process


def decode(path: Path, rate: int) -> torch.Tensor:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None: raise RuntimeError("FFmpeg is required")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if out.returncode: raise RuntimeError(out.stderr.decode(errors="replace"))
    return torch.frombuffer(bytearray(out.stdout), dtype=torch.float32).clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lavdf-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device, rank, world_size = init_distributed(args.device)
    records = [x for x in json.loads(args.subset.read_text(encoding="utf-8")) if x.get("split") == args.split]
    if args.max_items: records = records[:args.max_items]
    records = records[rank::world_size]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=1024, win_length=1024, hop_length=160,
        n_mels=64, f_min=60, f_max=7800,
    ).to(device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    total = torch.zeros((), dtype=torch.float64, device=device)
    total2 = torch.zeros((), dtype=torch.float64, device=device)
    for item in tqdm(records, desc="BYOL-A normalisation stats", disable=not is_main_process()):
        wave = decode(args.lavdf_root / item["file"], args.sample_rate).to(device)
        value = (mel(wave) + torch.finfo(torch.float32).eps).log().double()
        total += value.sum(); total2 += (value * value).sum(); count += value.numel()
    if world_size > 1:
        dist.all_reduce(total); dist.all_reduce(total2); dist.all_reduce(count)
    mean = total / count; std = torch.sqrt((total2 / count - mean * mean).clamp_min(1e-12))
    if is_main_process():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"mean": float(mean), "std": float(std), "n_values": int(count.item())}, indent=2),
            encoding="utf-8",
        )
        print(args.output, float(mean), float(std))
    cleanup_distributed()


if __name__ == "__main__":
    main()

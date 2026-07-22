"""Online paired neuron probing for UniCaCLF's TSN RGB and Flow encoders.

The input must be a manifest built by ``build_probe_pairs.py --modality video``;
therefore each pair is an authentic original and a *visual-only* LAV-DF fake.
No internal activations are written to disk: spatially pooled pair deltas are
accumulated online and only scores/top-rho indices are saved.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import re

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm
from torchvision.models import resnet50

try:
    from .probe_common import RunningVectorStats, read_pairs, save_probe_results
    from .distributed_utils import cleanup_distributed, init_distributed, is_distributed, is_main_process
except ImportError:  # allows direct execution from UniCaCLF/
    from probe_common import RunningVectorStats, read_pairs, save_probe_results
    from distributed_utils import cleanup_distributed, init_distributed, is_distributed, is_main_process


RGB_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
FLOW_MEAN, FLOW_STD = 128.0, 128.0


class TSNResNet50Encoder(nn.Module):
    """The ResNet-50 backbone used by the official TSN RGB/Flow checkpoints.

    We deliberately use torchvision rather than an MMACTION recognizer.  The
    released checkpoint names its encoder weights ``backbone.*`` and the
    ResNet-50 blocks (including the stride placement of the PyTorch style
    bottleneck) are architecturally identical.  This avoids tying feature
    probing to the legacy MMACTION 0.x / MMCV 1.x stack.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = resnet50(weights=None)
        if in_channels != 3:
            first = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, first.out_channels, kernel_size=first.kernel_size,
                stride=first.stride, padding=first.padding, bias=False,
            )
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_tsn(checkpoint: Path, modality: str, device: torch.device) -> torch.nn.Module:
    channels = 3 if modality == "RGB" else 10  # RGB 1 frame; Flow 5 (x,y) fields.
    model = TSNResNet50Encoder(channels)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    # The official TSN checkpoints include ``cls_head.*`` parameters.  This
    # probe intentionally builds a backbone-only model, so only load the
    # ResNet parameters and keep a strict check for those.
    def torchvision_key(key: str) -> str:
        """Map MMAction ConvModule keys to torchvision ResNet keys."""
        key = key.removeprefix("module.").removeprefix("backbone.")
        # Stem: ``conv1 = ConvModule(conv, bn)`` in MMAction.
        key = key.replace("conv1.conv.", "conv1.", 1)
        key = key.replace("conv1.bn.", "bn1.", 1)
        # Bottleneck ConvModules: layer1.0.conv2.conv -> layer1.0.conv2,
        # and their embedded BN -> layer1.0.bn2.
        key = re.sub(r"(layer[1-4]\.\d+\.conv[1-3])\.conv\.", r"\1.", key)
        key = re.sub(r"(layer[1-4]\.\d+)\.conv([1-3])\.bn\.", r"\1.bn\2.", key)
        # Projection shortcut is also a ConvModule in MMAction but a two-item
        # Sequential in torchvision.
        key = re.sub(r"(layer[1-4]\.\d+\.downsample)\.conv\.", r"\1.0.", key)
        key = re.sub(r"(layer[1-4]\.\d+\.downsample)\.bn\.", r"\1.1.", key)
        return key

    state = {
        torchvision_key(key): value
        for key, value in state.items()
        if key.removeprefix("module.").startswith("backbone.")
    }
    if not state:
        raise RuntimeError(f"TSN {modality} checkpoint contains no backbone.* parameters: {checkpoint}")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    # ``fc`` is intentionally replaced by Identity; all convolutional/BN
    # backbone parameters must still match exactly.
    critical_missing = [key for key in missing if not key.startswith("fc.")]
    classifier_keys = {"fc.weight", "fc.bias"}
    critical_unexpected = [key for key in unexpected if key not in classifier_keys]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            f"TSN {modality} backbone checkpoint mismatch; "
            f"missing={critical_missing}, unexpected={critical_unexpected}"
        )
    return model.to(device).eval()


def register_bottleneck_hooks(model: torch.nn.Module, prefix: str):
    captured: OrderedDict[str, torch.Tensor | None] = OrderedDict()
    handles = []
    for stage in range(1, 5):
        blocks = getattr(model.backbone, f"layer{stage}")
        for block_index, block in enumerate(blocks, 1):
            name = f"{prefix}_res{stage + 1}_b{block_index}"
            captured[name] = None
            handles.append(block.register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.__setitem__(name, output.detach())
            ))
    return captured, handles


def sampled_times(start: float, end: float, count: int) -> np.ndarray:
    if end <= start:
        raise ValueError(f"Invalid period [{start}, {end}]")
    return start + (np.arange(count, dtype=np.float64) + 0.5) * (end - start) / count


def decode_frames(path: str, times: np.ndarray, decode_threads: int = 2) -> tuple[np.ndarray, float]:
    reader = VideoReader(path, ctx=cpu(0), num_threads=decode_threads)
    fps = float(reader.get_avg_fps())
    if fps <= 0 or len(reader) == 0:
        raise ValueError(f"Unreadable video {path}")
    indices = np.clip(np.rint(times * fps).astype(np.int64), 0, len(reader) - 1)
    return reader.get_batch(indices).asnumpy(), fps


def resize_crop(images: np.ndarray, size: int) -> torch.Tensor:
    """N,H,W,C uint8 -> N,C,size,size float; deterministic TSN center crop."""
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
    _, _, height, width = tensor.shape
    scale = 256.0 / min(height, width)
    new_h, new_w = round(height * scale), round(width * scale)
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    return tensor[:, :, top:top + size, left:left + size]


def rgb_input(path: str, times: np.ndarray, size: int, decode_threads: int = 2) -> torch.Tensor:
    frames, _ = decode_frames(path, times, decode_threads)
    value = resize_crop(frames, size)
    return (value - RGB_MEAN) / RGB_STD


def _dense_flow(first: np.ndarray, second: np.ndarray, method: str, bound: float) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    if method == "tvl1":
        factory = getattr(getattr(cv2, "optflow", None), "DualTVL1OpticalFlow_create", None)
        factory = factory or getattr(cv2, "DualTVL1OpticalFlow_create", None)
        if factory is None:
            raise RuntimeError("--flow-method tvl1 requires opencv-contrib-python (cv2.optflow.DualTVL1OpticalFlow_create)")
        flow = factory().calc(left, right, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(left, right, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    encoded = np.clip((flow + bound) * (255.0 / (2.0 * bound)), 0, 255).astype(np.uint8)
    return encoded[..., 0], encoded[..., 1]


def flow_input(
    path: str, times: np.ndarray, size: int, method: str, bound: float,
    period_start: float, period_end: float, decode_threads: int = 2,
) -> torch.Tensor:
    """Build 5 x/y flow fields per centre without leaving the fake interval."""
    reader = VideoReader(path, ctx=cpu(0), num_threads=decode_threads)
    fps = float(reader.get_avg_fps())
    centers = np.clip(np.rint(times * fps).astype(np.int64), 0, len(reader) - 1)
    # A 5-flow sample needs six source frames.  Clamp it to the annotated
    # interval rather than only to the full-video boundaries, so no genuine
    # visual content outside a fake period leaks into its paired activation.
    lower = int(np.clip(np.ceil(period_start * fps), 0, len(reader) - 1))
    upper = int(np.clip(np.floor(period_end * fps), lower, len(reader) - 1))
    centers = np.clip(centers, lower, upper)
    windows = [np.clip(np.arange(c - 2, c + 4), lower, upper) for c in centers]
    required = np.unique(np.concatenate(windows))
    decoded = reader.get_batch(required).asnumpy()
    frame_map = {int(index): frame for index, frame in zip(required, decoded)}
    samples = []
    for indices in windows:
        channels = []
        for left, right in zip(indices[:-1], indices[1:]):
            x, y = _dense_flow(frame_map[int(left)], frame_map[int(right)], method, bound)
            channels.extend((x, y))
        samples.append(np.stack(channels, axis=-1))
    value = resize_crop(np.stack(samples), size)
    return (value - FLOW_MEAN) / FLOW_STD


def accumulate_stream(model, captured, real: torch.Tensor, fake: torch.Tensor, stats, device, amp: bool) -> None:
    length = real.shape[0]
    batch = torch.cat((real, fake), dim=0).to(device, non_blocking=True)
    for key in captured:
        captured[key] = None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        _ = model(batch)
    for name, activation in captured.items():
        if activation is None:
            raise RuntimeError(f"TSN hook did not capture {name}")
        # [2*N,C,H,W] -> [2,N,C] -> one paired vector [C].
        pooled = activation.float().mean(dim=(2, 3)).reshape(2, length, -1).mean(dim=1).cpu().numpy()
        stats[name].update(pooled[1] - pooled[0])


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
    parser.add_argument("--rgb-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-period", type=int, default=16, help="RGB/Flow temporal locations per fake period")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--flow-method", choices=("tvl1", "farneback"), default="tvl1")
    parser.add_argument("--flow-bound", type=float, default=20.0)
    parser.add_argument("--top-ratio", type=float, default=0.10)
    parser.add_argument(
        "--max-pairs", type=int, default=100,
        help="Number of strict pairs to probe (default: 100; use 0 for every pair in the manifest)",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.top_ratio <= 1: parser.error("--top-ratio must be in (0,1]")
    if args.max_pairs < 0: parser.error("--max-pairs must be >= 0")
    device, rank, world_size = init_distributed(args.device)
    rgb_model = build_tsn(args.rgb_checkpoint, "RGB", device)
    flow_model = build_tsn(args.flow_checkpoint, "Flow", device)
    rgb_hidden, rgb_handles = register_bottleneck_hooks(rgb_model, "rgb")
    flow_hidden, flow_handles = register_bottleneck_hooks(flow_model, "flow")
    stats: Dict[str, RunningVectorStats] = {}
    # Initialising widths from a single module is safer than hard-coding ResNet stage dimensions.
    for model, prefix in ((rgb_model, "rgb"), (flow_model, "flow")):
        for stage in range(1, 5):
            for block_index, block in enumerate(getattr(model.backbone, f"layer{stage}"), 1):
                stats[f"{prefix}_res{stage + 1}_b{block_index}"] = RunningVectorStats(block.conv3.out_channels)
    pairs = read_pairs(args.pairs, "video")
    if args.max_pairs > 0: pairs = pairs[:args.max_pairs]
    pairs = pairs[rank::world_size]
    failures = []
    try:
        for pair in tqdm(pairs, desc="TSN paired neuron probe", disable=not is_main_process()):
            try:
                times = sampled_times(pair.start_sec, pair.end_sec, args.samples_per_period)
                accumulate_stream(rgb_model, rgb_hidden, rgb_input(pair.original_file, times, args.image_size), rgb_input(pair.fake_file, times, args.image_size), stats, device, args.amp)
                accumulate_stream(
                    flow_model, flow_hidden,
                    flow_input(pair.original_file, times, args.image_size, args.flow_method, args.flow_bound, pair.start_sec, pair.end_sec),
                    flow_input(pair.fake_file, times, args.image_size, args.flow_method, args.flow_bound, pair.start_sec, pair.end_sec),
                    stats, device, args.amp,
                )
            except Exception as error:
                failures.append(f"{pair.pair_id}: {type(error).__name__}: {error}")
    finally:
        for handle in rgb_handles + flow_handles: handle.remove()
    merged = gather_stats(stats)
    if is_distributed():
        gathered_failures = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_failures, failures)
        failures = [value for part in gathered_failures for value in part]
    if is_main_process():
        save_probe_results(args.output_dir, "tsn", merged, args.top_ratio)
        if failures:
            (args.output_dir / "tsn_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        processed = next(iter(merged.values())).count
        print(f"Processed {processed}/{processed + len(failures)} pairs. Results: {args.output_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()

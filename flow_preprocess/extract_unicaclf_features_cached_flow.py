"""Extract UniCaCLF final/neuron features using a precomputed TV-L1 Flow cache.

This is a cache-aware entry point kept outside ``UniCaCLF/``.  It reuses the
unchanged UniCaCLF feature-extraction implementation and replaces its module
global ``flow_input`` function at runtime.  Thus every RGB, audio, encoder,
hook, feature-saving and distributed code path remains identical to the
baseline extractor; only the online OpenCV TV-L1 computation is replaced by a
read from ``<flow-cache-root>/<split>/<video_id>.npy``.

The cache must be produced by ``flow_preprocess.precompute_tvl1_flow`` with
the same stride, image size and Flow bound.  Mismatches fail before models are
loaded rather than silently changing the TSN-Flow input distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


class InMemorySubset:
    """Path-like subset object that keeps core settings pointing at the source JSON."""

    def __init__(self, source: Path, records: list[dict]):
        self.source = source
        self.payload = json.dumps(records)

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.payload

    def __str__(self) -> str:
        return str(self.source)


def select_contiguous_batch(records: list, batch_count: int, batch_index: int) -> list:
    if batch_count == 1:
        return records
    start = len(records) * batch_index // batch_count
    end = len(records) * (batch_index + 1) // batch_count
    return records[start:end]


def parse_wrapper_args() -> argparse.Namespace:
    """Consume this wrapper's arguments; leave all UniCaCLF arguments intact."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--flow-cache-root", type=Path)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Optional outer data shard count before rank sharding")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="0-based outer data shard index before rank sharding")
    parser.add_argument("--batch-count", type=int, default=1,
                        help="Split each rank-local shard into this many batches")
    parser.add_argument("--batch-index", type=int, default=0,
                        help="0-based batch index inside each rank-local shard")
    wrapper_args, remaining = parser.parse_known_args()
    if wrapper_args.flow_cache_root is None:
        if "-h" in remaining or "--help" in remaining:
            print(
                "Additional cache wrapper arguments:\n"
                "  --flow-cache-root FLOW_CACHE_ROOT\n"
                "  --batch-count BATCH_COUNT\n"
                "  --batch-index BATCH_INDEX\n"
                "  --shard-count SHARD_COUNT\n"
                "  --shard-index SHARD_INDEX"
            )
            sys.argv = [sys.argv[0], "--help"]
            import UniCaCLF.extract_unicaclf_offline_features as core
            core.main()
        parser.error("--flow-cache-root is required")
    if wrapper_args.shard_count < 1 or not 0 <= wrapper_args.shard_index < wrapper_args.shard_count:
        parser.error("--shard-count must be positive and --shard-index must satisfy 0 <= index < count")
    if wrapper_args.batch_count < 1 or not 0 <= wrapper_args.batch_index < wrapper_args.batch_count:
        parser.error("--batch-count must be positive and --batch-index must satisfy 0 <= index < count")
    sys.argv = [sys.argv[0], *remaining]
    return wrapper_args


def apply_wrapper_subset_filter(core_args, wrapper_args: argparse.Namespace) -> None:
    """Match precompute_tvl1_flow's shard -> rank -> batch record selection."""
    if wrapper_args.shard_count == 1 and wrapper_args.batch_count == 1:
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    records = json.loads(core_args.subset.read_text(encoding="utf-8"))
    selected: list[dict] = []
    for split in core_args.splits:
        split_records = [item for item in records if item.get("split") == split]
        if core_args.max_items is not None:
            split_records = split_records[:core_args.max_items]
        split_records = select_contiguous_batch(split_records, wrapper_args.shard_count, wrapper_args.shard_index)
        selected_positions: list[int] = []
        for rank in range(world_size):
            rank_positions = list(range(rank, len(split_records), world_size))
            selected_positions.extend(
                select_contiguous_batch(rank_positions, wrapper_args.batch_count, wrapper_args.batch_index)
            )
        selected.extend(split_records[index] for index in sorted(selected_positions))
    core_args.subset = InMemorySubset(core_args.subset, selected)
    core_args.max_items = None

def validate_cache(root: Path, *, stride: int, image_size: int, flow_bound: float) -> None:
    settings_path = root / "flow_cache_settings.json"
    if not settings_path.is_file():
        raise FileNotFoundError(f"Flow cache settings do not exist: {settings_path}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    expected = {
        "method": "tvl1",
        "video_stride_frames": stride,
        "image_size": image_size,
        "flow_bound": flow_bound,
        "stored_layout": "T,10,H,W",
    }
    mismatches = {key: (settings.get(key), wanted) for key, wanted in expected.items() if settings.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Incompatible TV-L1 cache settings in {settings_path}: {mismatches}")


def cached_flow_input(cache_root: Path):
    """Return a drop-in replacement for UniCaCLF's online ``flow_input``."""
    time_grids: dict[Path, np.ndarray] = {}

    def load(
        path: str, times: np.ndarray, image_size: int, method: str, bound: float,
        period_start: float, period_end: float, decode_threads: int = 2,
    ) -> torch.Tensor:
        if method != "tvl1":
            raise ValueError("The precomputed cache contains TV-L1 Flow; use --flow-method tvl1")
        video = Path(path)
        split = video.parent.name
        cache_path = cache_root / split / f"{video.stem}.npy"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing cached Flow: {cache_path}")

        # The unchanged extractor calls this function once per contiguous
        # temporal batch.  Reconstruct the cache offsets from the full video
        # grid, then use searchsorted to select precisely these positions.
        if video not in time_grids:
            import UniCaCLF.extract_unicaclf_offline_features as core
            all_times, _ = core.video_time_grid(video, stride=core_args.video_stride_frames,
                                                decode_threads=core_args.decode_threads)
            time_grids[video] = all_times
        all_times = time_grids[video]
        positions = np.searchsorted(all_times, times)
        if np.any(positions >= len(all_times)) or not np.allclose(all_times[positions], times, atol=1e-6):
            raise ValueError(f"Cached Flow time grid mismatch for {video}")

        cached = np.load(cache_path, mmap_mode="r")
        expected = (len(all_times), 10, image_size, image_size)
        if tuple(cached.shape) != expected:
            raise ValueError(f"Cached Flow shape mismatch for {cache_path}: got {cached.shape}, expected {expected}")
        # ``ascontiguousarray`` detaches the requested small temporal batch
        # from the memory map before the next video is opened.
        return torch.from_numpy(np.ascontiguousarray(cached[positions])).float()
    return load


def main() -> None:
    global core_args
    wrapper_args = parse_wrapper_args()
    cache_root = wrapper_args.flow_cache_root
    import UniCaCLF.extract_unicaclf_offline_features as core

    # Parse the original extractor's complete command line once, validate the
    # cache, then replace only its online Flow provider.  Calling core.main()
    # would parse a second time, so invoke its main body through this small
    # argument-preserving wrapper instead.
    core_args = core.parse_args()
    apply_wrapper_subset_filter(core_args, wrapper_args)
    validate_cache(cache_root, stride=core_args.video_stride_frames,
                   image_size=core_args.image_size, flow_bound=core_args.flow_bound)
    core.flow_input = cached_flow_input(cache_root)

    # Core main normally calls parse_args itself.  Temporarily substitute that
    # parser with a zero-argument closure so all remaining logic is unchanged.
    original_parse_args = core.parse_args
    core.parse_args = lambda: core_args
    try:
        core.main()
    finally:
        core.parse_args = original_parse_args


if __name__ == "__main__":
    main()

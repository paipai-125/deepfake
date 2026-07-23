"""Remove precomputed TV-L1 Flow files for one metadata shard/batch.

This deletes only direct ``<flow-cache-root>/<split>/<video_id>.npy`` files
created by ``flow_preprocess.precompute_tvl1_flow``. It does not touch extracted
``final/`` or ``neurons/`` feature directories under the same cache root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-cache-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max-items", type=int, default=None,
                        help="Debug limit per selected split, applied before shard/rank/batch selection")
    parser.add_argument("--shard-count", type=int, default=None,
                        help="Outer data shard count; defaults to flow_cache_settings.json or 1")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="0-based outer data shard index; defaults to flow_cache_settings.json or 0")
    parser.add_argument("--batch-count", type=int, default=None,
                        help="Rank-local batch count; defaults to flow_cache_settings.json or 1")
    parser.add_argument("--batch-index", type=int, default=None,
                        help="0-based rank-local batch index; defaults to flow_cache_settings.json or 0")
    parser.add_argument("--rank-shard-count", type=int, default=None,
                        help="torchrun world size used for precompute; defaults to flow_cache_settings.json or 1")
    parser.add_argument("--dry-run", action="store_true", help="Print the count without deleting files")
    parser.add_argument("--fail-on-missing", action="store_true",
                        help="Raise an error if an expected cache file is absent")
    return parser.parse_args()


def select_contiguous_batch(records: list[dict], batch_count: int, batch_index: int) -> list[dict]:
    if batch_count == 1:
        return records
    start = len(records) * batch_index // batch_count
    end = len(records) * (batch_index + 1) // batch_count
    return records[start:end]


def positive_index(count: int, index: int, name: str) -> None:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--{name}-count must be positive and --{name}-index must satisfy 0 <= index < count")


def load_settings(root: Path) -> dict:
    settings_path = root / "flow_cache_settings.json"
    if settings_path.is_file():
        return json.loads(settings_path.read_text(encoding="utf-8"))
    return {}


def resolve_int(value: int | None, settings: dict, key: str, default: int) -> int:
    return int(settings.get(key, default) if value is None else value)


def main() -> None:
    args = parse_args()
    settings = load_settings(args.flow_cache_root)
    shard_count = resolve_int(args.shard_count, settings, "shard_count", 1)
    shard_index = resolve_int(args.shard_index, settings, "shard_index", 0)
    batch_count = resolve_int(args.batch_count, settings, "batch_count", 1)
    batch_index = resolve_int(args.batch_index, settings, "batch_index", 0)
    rank_shard_count = resolve_int(args.rank_shard_count, settings, "world_size", 1)
    positive_index(shard_count, shard_index, "shard")
    positive_index(batch_count, batch_index, "batch")
    if rank_shard_count < 1:
        raise ValueError("--rank-shard-count must be positive")

    records = json.loads(args.subset.read_text(encoding="utf-8"))
    removed = 0
    missing: list[Path] = []
    for split in args.splits:
        split_records = [item for item in records if item.get("split") == split]
        if args.max_items is not None:
            split_records = split_records[:args.max_items]
        split_records = select_contiguous_batch(split_records, shard_count, shard_index)
        for rank in range(rank_shard_count):
            rank_records = split_records[rank::rank_shard_count]
            batch_records = select_contiguous_batch(rank_records, batch_count, batch_index)
            for item in batch_records:
                source = Path(item["file"])
                cache_path = args.flow_cache_root / split / f"{source.stem}.npy"
                if cache_path.is_file():
                    removed += 1
                    if not args.dry_run:
                        cache_path.unlink()
                elif args.fail_on_missing:
                    missing.append(cache_path)

    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing {len(missing)} expected Flow cache files:\n{preview}{extra}")

    verb = "Would remove" if args.dry_run else "Removed"
    shard_label = "" if shard_count == 1 else f" shard {shard_index + 1}/{shard_count}"
    batch_label = "" if batch_count == 1 else f" batch {batch_index + 1}/{batch_count}"
    print(f"{verb} {removed} cached Flow files{shard_label}{batch_label} under {args.flow_cache_root} using rank_shard_count={rank_shard_count}")


if __name__ == "__main__":
    main()

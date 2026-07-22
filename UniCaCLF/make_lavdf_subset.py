"""Create a deterministic stratified LAV-DF subset manifest for all experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_counts(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        split, count = value.split("=", 1)
        if split not in {"train", "dev", "test"} or int(count) < 1:
            raise ValueError(f"Invalid count: {value}; expected train=12000/dev=2000/test=4000")
        result[split] = int(count)
    return result


def sample(records: list[dict], count: int, rng: np.random.Generator) -> list[dict]:
    groups: dict[tuple[bool, bool], list[dict]] = {}
    for item in records:
        groups.setdefault((bool(item.get("modify_video")), bool(item.get("modify_audio"))), []).append(item)
    selected = []
    keys = sorted(groups)
    base, extra = divmod(count, len(keys))
    for i, key in enumerate(keys):
        values = groups[key]; rng.shuffle(values)
        selected.extend(values[:base + int(i < extra)])
    if len(selected) < count:
        have = {x["file"] for x in selected}
        remainder = [x for key in keys for x in groups[key] if x["file"] not in have]
        rng.shuffle(remainder); selected.extend(remainder[:count - len(selected)])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lavdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", default=["train=12000", "dev=2000", "test=4000"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    counts = parse_counts(args.counts)
    metadata = json.loads((args.lavdf_root / "metadata.min.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    output = []
    for split, count in counts.items():
        values = [x for x in metadata if x.get("split") == split]
        if count > len(values): raise ValueError(f"Requested {count} {split} samples, only {len(values)} available")
        output.extend(sample(values, count, rng))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary = {split: sum(x["split"] == split for x in output) for split in counts}
    print(json.dumps({"output": str(args.output), "counts": summary, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()

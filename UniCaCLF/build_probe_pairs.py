"""Create strict single-modality fake/original LAV-DF pair manifests.

Input `--subset` is a JSON list of selected official metadata records.  This
script never samples from the full dataset by itself, preventing accidental
train/dev/test or subset leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Support both `python file.py` and `python -m UniCaCLF.file`.
    from .probe_common import ProbePair
except ImportError:
    from probe_common import ProbePair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lavdf-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True, help="JSON list of selected LAV-DF metadata records")
    parser.add_argument("--modality", choices=("video", "audio"), required=True)
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    records = json.loads(args.subset.read_text(encoding="utf-8"))
    pairs = []
    for item in records:
        if item.get("split") != args.split:
            continue
        visual_only = bool(item.get("modify_video")) and not bool(item.get("modify_audio"))
        audio_only = bool(item.get("modify_audio")) and not bool(item.get("modify_video"))
        if not (visual_only if args.modality == "video" else audio_only):
            continue
        original = item.get("original")
        if not original:
            continue
        for period_index, period in enumerate(item.get("fake_periods") or []):
            start, end = map(float, period)
            if end <= start:
                continue
            fake = args.lavdf_root / item["file"]
            real = args.lavdf_root / original
            if not fake.is_file() or not real.is_file():
                continue
            pairs.append(ProbePair(
                pair_id=f"{args.modality}-{Path(item['file']).stem}-{period_index:02d}", modality=args.modality,
                fake_file=str(fake.resolve()), original_file=str(real.resolve()),
                start_sec=start, end_sec=end, split=args.split, fake_period=(start, end),
            ))
            if args.max_pairs and len(pairs) >= args.max_pairs:
                break
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break
    if not pairs:
        raise RuntimeError("No strict single-modality fake/original pairs found in the chosen subset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(pair.__dict__) + "\n" for pair in pairs), encoding="utf-8")
    print(f"Wrote {len(pairs)} strict {args.modality} pairs: {args.output}")


if __name__ == "__main__":
    main()

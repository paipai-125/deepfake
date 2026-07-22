"""Shared LAV-DF pair, statistic, and result utilities for encoder probing."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ProbePair:
    pair_id: str
    modality: str
    fake_file: str
    original_file: str
    start_sec: float
    end_sec: float
    split: str
    fake_period: tuple[float, float]


def read_pairs(path: Path, modality: str) -> list[ProbePair]:
    pairs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("modality") != modality:
            raise ValueError(f"{path}:{line_number} has modality={value.get('modality')!r}, expected {modality!r}")
        pairs.append(ProbePair(**value))
    if not pairs:
        raise ValueError(f"No {modality} pairs in {path}")
    ids = [pair.pair_id for pair in pairs]
    if len(ids) != len(set(ids)):
        seen, duplicate_ids = set(), set()
        for value in ids:
            if value in seen:
                duplicate_ids.add(value)
            seen.add(value)
        duplicates = sorted(duplicate_ids)
        raise ValueError(f"Duplicate pair_id values in {path}: {duplicates[:8]}")
    return pairs


class RunningVectorStats:
    """Welford accumulator for one paired delta vector per input pair."""
    def __init__(self, width: int):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.mean.shape:
            raise ValueError(f"Expected delta shape {self.mean.shape}, received {value.shape}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.full_like(self.mean, np.nan)
        return np.sqrt(self.m2 / (self.count - 1))

    def merge_state(self, count: int, mean: np.ndarray, m2: np.ndarray) -> None:
        """Merge another Welford accumulator without retaining its samples."""
        if count == 0:
            return
        mean = np.asarray(mean, dtype=np.float64)
        m2 = np.asarray(m2, dtype=np.float64)
        if mean.shape != self.mean.shape or m2.shape != self.m2.shape:
            raise ValueError(f"Cannot merge stats with shape {mean.shape} into {self.mean.shape}")
        if self.count == 0:
            self.count, self.mean, self.m2 = int(count), mean.copy(), m2.copy()
            return
        total = self.count + int(count)
        delta = mean - self.mean
        self.mean += delta * (count / total)
        self.m2 += m2 + delta * delta * (self.count * count / total)
        self.count = total

    def export_state(self) -> tuple[int, np.ndarray, np.ndarray]:
        return self.count, self.mean, self.m2


def paired_shift_score(stats: RunningVectorStats, eps: float = 1e-4, floor_ratio: float = 0.05) -> np.ndarray:
    """Stable paired standardised activation shift, |mean(delta)| / std(delta)."""
    std = stats.std()
    valid = std[np.isfinite(std) & (std > eps)]
    floor = max(eps, float(np.median(valid)) * floor_ratio) if valid.size else 1.0
    return np.abs(stats.mean) / np.sqrt(std * std + floor * floor)


def save_probe_results(output_dir: Path, prefix: str, stats: dict[str, RunningVectorStats], top_ratio: float) -> None:
    """Save every channel score, selected indices, CSV, and compact layer plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    rows, layer_names, layer_shift, top_scores = [], [], [], []
    neuron_rows = []
    for name, state in stats.items():
        score = paired_shift_score(state)
        count = max(1, int(np.ceil(top_ratio * score.size)))
        indices = np.argsort(score)[::-1][:count]
        arrays[f"{name}_score"] = score.astype(np.float32)
        arrays[f"{name}_mean_delta"] = state.mean.astype(np.float32)
        arrays[f"{name}_std_delta"] = state.std().astype(np.float32)
        arrays[f"{name}_top_indices"] = indices.astype(np.int32)
        rms = float(np.sqrt(np.mean(score * score)))
        rows.append((name, state.count, score.size, count, rms, float(np.mean(score)), float(score[indices[0]])))
        layer_names.append(name); layer_shift.append(rms); top_scores.append(float(score[indices[0]]))
        for rank, channel in enumerate(indices, 1):
            neuron_rows.append((name, rank, int(channel), float(score[channel]), float(state.mean[channel]), float(state.std()[channel])))
    np.savez_compressed(output_dir / f"{prefix}_neuron_scores.npz", **arrays)
    with (output_dir / f"{prefix}_layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "n_pairs", "channels", "top_count", "d_shift_rms", "mean_channel_score", "best_channel_score"])
        writer.writerows(rows)
    with (output_dir / f"{prefix}_top_neurons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "rank", "channel_index", "paired_shift_score", "mean_delta_fake_minus_real", "std_delta"])
        writer.writerows(neuron_rows)

    x = np.arange(1, len(layer_names) + 1)
    fig, axis = plt.subplots(figsize=(max(8, 0.52 * len(layer_names)), 4.8), dpi=180)
    axis.plot(x, layer_shift, marker="o", linewidth=2, label=r"$D_{shift}^{(l)}$ (RMS over channels)")
    axis.plot(x, top_scores, marker="s", linewidth=1.7, label="best neuron ShiftScore")
    axis.set_xticks(x, layer_names, rotation=50, ha="right")
    axis.set_ylabel("paired standardised activation shift")
    axis.set_title(f"{prefix}: layer and top-neuron shifts")
    axis.grid(alpha=0.25); axis.legend(frameon=False); fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_layer_shift.png", bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8, 0.52 * len(layer_names)), 4.8), dpi=180)
    counts = [row[3] for row in rows]
    axis.bar(x, counts, color="#1769aa")
    axis.set_xticks(x, layer_names, rotation=50, ha="right")
    axis.set_ylabel(r"selected neurons (top $\rho$%)")
    axis.set_title(f"{prefix}: selected neurons per layer (rho={top_ratio:g})")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_selected_counts.png", bbox_inches="tight"); plt.close(fig)

    settings = {"top_ratio": top_ratio, "layers": list(stats), "n_pairs": {key: value.count for key, value in stats.items()}}
    (output_dir / f"{prefix}_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

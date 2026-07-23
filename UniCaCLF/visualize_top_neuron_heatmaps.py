"""Draw per-layer top-ratio neuron heatmaps from UniCaCLF probe outputs.

The probing scripts save one score vector per encoder layer in
``*_neuron_scores.npz``. This script uses the same selection rule as
``probe_common.save_probe_results``: sort channels by paired shift score and
keep ``ceil(top_ratio * n_channels)`` channels for every layer.

Example:

  python -m UniCaCLF.visualize_top_neuron_heatmaps \
    --tsn-scores ../deepfake-data/results/unicaclf_probe/tsn/tsn_neuron_scores.npz \
    --byola-scores ../deepfake-data/results/unicaclf_probe/byola/byola_neuron_scores.npz \
    --output-dir ../deepfake-data/results/unicaclf_probe/neuron_heatmaps \
    --top-ratio 0.10
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TSN_SCORES = Path("../deepfake-data/results/unicaclf_probe/tsn/tsn_neuron_scores.npz")
DEFAULT_BYOLA_SCORES = Path("../deepfake-data/results/unicaclf_probe/byola/byola_neuron_scores.npz")
DEFAULT_OUTPUT_DIR = Path("../deepfake-data/results/unicaclf_probe/neuron_heatmaps")


@dataclass(frozen=True)
class LayerSelection:
    layer: str
    channels: int
    top_count: int
    indices: np.ndarray
    score: np.ndarray
    mean_delta: np.ndarray | None
    std_delta: np.ndarray | None


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def ratio_label(top_ratio: float) -> str:
    percent = f"{top_ratio * 100:.4f}".rstrip("0").rstrip(".")
    return f"top{percent.replace('.', 'p')}pct"


def ratio_text(top_ratio: float) -> str:
    percent = f"{top_ratio * 100:.4f}".rstrip("0").rstrip(".")
    return f"top {percent}%"


def stable_paired_shift_score(
    mean_delta: np.ndarray,
    std_delta: np.ndarray,
    eps: float = 1e-4,
    floor_ratio: float = 0.05,
) -> np.ndarray:
    """Recompute the probe score when an older NPZ lacks ``*_score`` arrays."""
    mean_delta = np.asarray(mean_delta, dtype=np.float64)
    std_delta = np.asarray(std_delta, dtype=np.float64)
    valid = std_delta[np.isfinite(std_delta) & (std_delta > eps)]
    floor = max(eps, float(np.median(valid)) * floor_ratio) if valid.size else 1.0
    return np.abs(mean_delta) / np.sqrt(std_delta * std_delta + floor * floor)


def _npz_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    return np.asarray(data[key], dtype=np.float64).copy()


def layer_names_from_npz(data: np.lib.npyio.NpzFile) -> list[str]:
    # NpzFile keeps the insertion order produced by save_probe_results, which is
    # the encoder traversal order. Preserve that instead of alphabetic sorting.
    layers = [key[: -len("_score")] for key in data.files if key.endswith("_score")]
    if layers:
        return layers
    layers = []
    for key in data.files:
        if key.endswith("_mean_delta"):
            layer = key[: -len("_mean_delta")]
            if f"{layer}_std_delta" in data.files:
                layers.append(layer)
    return layers


def load_layer_selections(score_path: Path, top_ratio: float) -> list[LayerSelection]:
    if not score_path.exists():
        raise FileNotFoundError(f"Score file does not exist: {score_path}")
    selections: list[LayerSelection] = []
    with np.load(score_path, allow_pickle=False) as data:
        layer_names = layer_names_from_npz(data)
        if not layer_names:
            raise ValueError(f"No layer score arrays found in {score_path}")
        for layer in layer_names:
            mean_delta = _npz_array(data, f"{layer}_mean_delta")
            std_delta = _npz_array(data, f"{layer}_std_delta")
            score = _npz_array(data, f"{layer}_score")
            if score is None:
                if mean_delta is None or std_delta is None:
                    raise ValueError(f"Layer {layer} has no score or mean/std arrays in {score_path}")
                score = stable_paired_shift_score(mean_delta, std_delta)
            if score.ndim != 1:
                raise ValueError(f"Layer {layer} score must be 1-D, got {score.shape}")
            channels = int(score.size)
            if channels == 0:
                raise ValueError(f"Layer {layer} has zero channels")
            for name, values in (("mean_delta", mean_delta), ("std_delta", std_delta)):
                if values is not None and values.shape != score.shape:
                    raise ValueError(f"Layer {layer} {name} shape {values.shape} != score shape {score.shape}")
            top_count = max(1, int(np.ceil(top_ratio * channels)))
            ranking = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf)
            indices = np.argsort(ranking)[::-1][:top_count].astype(np.int64)
            selections.append(
                LayerSelection(
                    layer=layer,
                    channels=channels,
                    top_count=top_count,
                    indices=indices,
                    score=score,
                    mean_delta=mean_delta,
                    std_delta=std_delta,
                )
            )
    return selections


def selected_values(selection: LayerSelection, field: str) -> np.ndarray:
    if field == "score":
        source = selection.score
    elif field == "mean_delta":
        source = selection.mean_delta
    elif field == "std_delta":
        source = selection.std_delta
    else:
        raise ValueError(f"Unknown field: {field}")
    if source is None:
        return np.full(selection.top_count, np.nan, dtype=np.float64)
    return np.asarray(source[selection.indices], dtype=np.float64)


def padded_matrix(selections: list[LayerSelection], field: str) -> np.ndarray:
    width = max(selection.top_count for selection in selections)
    matrix = np.full((len(selections), width), np.nan, dtype=np.float64)
    for row, selection in enumerate(selections):
        values = selected_values(selection, field)
        matrix[row, : values.size] = values
    return matrix


def rank_ticks(width: int) -> tuple[np.ndarray, list[str]]:
    if width <= 30:
        ticks = np.arange(width)
    else:
        step = max(1, int(np.ceil(width / 10)))
        ticks = np.arange(0, width, step)
        if ticks[-1] != width - 1:
            ticks = np.append(ticks, width - 1)
    return ticks, [str(int(value) + 1) for value in ticks]


def color_limits(matrix: np.ndarray, diverging: bool, percentile: float) -> tuple[float | None, float | None]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return None, None
    percentile = float(np.clip(percentile, 0.0, 100.0))
    if diverging:
        limit = float(np.percentile(np.abs(finite), percentile))
        if limit == 0:
            limit = max(float(np.max(np.abs(finite))), 1e-12)
        return -limit, limit
    vmax = float(np.percentile(finite, percentile))
    if vmax <= 0:
        vmax = float(np.max(finite))
    return 0.0, vmax if vmax > 0 else None


def annotate_cells(
    axis: plt.Axes,
    matrix: np.ndarray,
    selections: list[LayerSelection],
    mode: str,
    max_cols: int,
) -> None:
    if mode == "none" or matrix.shape[1] > max_cols:
        return
    for row, selection in enumerate(selections):
        values = selected_values(selection, "score")
        for col in range(selection.top_count):
            if mode == "channel":
                text = str(int(selection.indices[col]))
            elif mode == "score":
                text = f"{values[col]:.2f}"
            else:
                raise ValueError(f"Unknown annotation mode: {mode}")
            axis.text(col, row, text, ha="center", va="center", fontsize=6, color="black")


def plot_heatmap(
    matrix: np.ndarray,
    selections: list[LayerSelection],
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap_name: str,
    diverging: bool,
    dpi: int,
    color_percentile: float,
    annotate: str,
    annotate_max_cols: int,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f2f4f7")
    vmin, vmax = color_limits(matrix, diverging, color_percentile)
    width = matrix.shape[1]
    height = matrix.shape[0]
    fig_width = min(max(9.0, 3.5 + 0.075 * width), 30.0)
    fig_height = max(4.0, 1.6 + 0.36 * height)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ticks, labels = rank_ticks(width)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, rotation=0, fontsize=8)
    axis.set_yticks(np.arange(height))
    axis.set_yticklabels([selection.layer for selection in selections], fontsize=8)
    axis.set_xlabel("selected neuron rank within each layer")
    axis.set_ylabel("encoder layer")
    axis.set_title(title)
    axis.tick_params(axis="both", length=0)
    axis.set_facecolor("#f2f4f7")
    annotate_cells(axis, matrix, selections, annotate, annotate_max_cols)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_selected_csv(path: Path, selections: list[LayerSelection], top_ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer",
            "rank",
            "channel_index",
            "paired_shift_score",
            "mean_delta_fake_minus_real",
            "std_delta",
            "channels_in_layer",
            "top_ratio",
        ])
        for selection in selections:
            scores = selected_values(selection, "score")
            means = selected_values(selection, "mean_delta")
            stds = selected_values(selection, "std_delta")
            for rank, channel in enumerate(selection.indices, 1):
                writer.writerow([
                    selection.layer,
                    rank,
                    int(channel),
                    f"{scores[rank - 1]:.8g}",
                    "" if np.isnan(means[rank - 1]) else f"{means[rank - 1]:.8g}",
                    "" if np.isnan(stds[rank - 1]) else f"{stds[rank - 1]:.8g}",
                    selection.channels,
                    f"{top_ratio:.8g}",
                ])


def grouped_selections(prefix: str, selections: list[LayerSelection]) -> list[tuple[str, str, list[LayerSelection]]]:
    title = "TSN video encoder" if prefix == "tsn" else "BYOL-A audio encoder"
    groups = [(prefix, title, selections)]
    if prefix == "tsn":
        for stream in ("rgb", "flow"):
            subset = [selection for selection in selections if selection.layer.startswith(f"{stream}_")]
            if subset:
                groups.append((f"tsn_{stream}", f"TSN {stream.upper()} stream", subset))
    if prefix == "byola":
        for block in ("audio_conv", "audio_fc"):
            subset = [selection for selection in selections if selection.layer.startswith(block)]
            if subset:
                suffix = "conv" if block.endswith("conv") else "fc"
                groups.append((f"byola_{suffix}", f"BYOL-A audio {suffix.upper()} layers", subset))
    return groups


def save_outputs(
    prefix: str,
    selections: list[LayerSelection],
    output_dir: Path,
    top_ratio: float,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = ratio_label(top_ratio)
    created: list[Path] = []
    csv_path = output_dir / f"{prefix}_{label}_selected_neurons.csv"
    write_selected_csv(csv_path, selections, top_ratio)
    created.append(csv_path)

    for group_prefix, group_title, group in grouped_selections(prefix, selections):
        score_path = output_dir / f"{group_prefix}_{label}_score_heatmap.png"
        plot_heatmap(
            padded_matrix(group, "score"),
            group,
            f"{group_title}: {ratio_text(top_ratio)} neurons ranked by paired shift score",
            "paired shift score",
            score_path,
            "magma",
            False,
            args.dpi,
            args.color_percentile,
            args.annotate,
            args.annotate_max_cols,
        )
        created.append(score_path)
        if args.mean_delta_heatmap:
            mean_path = output_dir / f"{group_prefix}_{label}_mean_delta_heatmap.png"
            plot_heatmap(
                padded_matrix(group, "mean_delta"),
                group,
                f"{group_title}: signed fake-real mean delta for {ratio_text(top_ratio)} neurons",
                "mean delta (fake - real)",
                mean_path,
                "coolwarm",
                True,
                args.dpi,
                args.color_percentile,
                args.annotate,
                args.annotate_max_cols,
            )
            created.append(mean_path)
        if args.std_delta_heatmap:
            std_path = output_dir / f"{group_prefix}_{label}_std_delta_heatmap.png"
            plot_heatmap(
                padded_matrix(group, "std_delta"),
                group,
                f"{group_title}: delta std for {ratio_text(top_ratio)} neurons",
                "std delta",
                std_path,
                "viridis",
                False,
                args.dpi,
                args.color_percentile,
                args.annotate,
                args.annotate_max_cols,
            )
            created.append(std_path)
    return created


def existing_or_none(path: Path) -> Path | None:
    resolved = resolve_path(path)
    return resolved if resolved and resolved.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsn-scores", type=Path, default=None, help="TSN video probe *_neuron_scores.npz")
    parser.add_argument("--byola-scores", type=Path, default=None, help="BYOL-A audio probe *_neuron_scores.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-ratio", type=float, default=0.10, help="Per-layer fraction selected with the probe ranking rule")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--color-percentile", type=float, default=99.0, help="Robust upper color scale percentile")
    parser.add_argument("--mean-delta-heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--std-delta-heatmap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--annotate", choices=("none", "channel", "score"), default="none")
    parser.add_argument("--annotate-max-cols", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.top_ratio <= 1:
        raise ValueError("--top-ratio must be in (0, 1]")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None

    explicit_scores = args.tsn_scores is not None or args.byola_scores is not None
    if explicit_scores:
        tsn_scores = resolve_path(args.tsn_scores) if args.tsn_scores else None
        byola_scores = resolve_path(args.byola_scores) if args.byola_scores else None
    else:
        tsn_scores = existing_or_none(DEFAULT_TSN_SCORES)
        byola_scores = existing_or_none(DEFAULT_BYOLA_SCORES)
        if tsn_scores is None and byola_scores is None:
            raise FileNotFoundError(
                "No default score files were found. Provide --tsn-scores and/or --byola-scores."
            )

    jobs: list[tuple[str, Path]] = []
    if tsn_scores is not None:
        jobs.append(("tsn", tsn_scores))
    if byola_scores is not None:
        jobs.append(("byola", byola_scores))
    if not jobs:
        raise ValueError("No score files selected")

    all_created: list[Path] = []
    summary: dict[str, object] = {
        "top_ratio": args.top_ratio,
        "score_files": {},
        "outputs": [],
        "layers": {},
    }
    for prefix, score_path in jobs:
        selections = load_layer_selections(score_path, args.top_ratio)
        created = save_outputs(prefix, selections, output_dir, args.top_ratio, args)
        all_created.extend(created)
        summary["score_files"][prefix] = str(score_path)
        summary["layers"][prefix] = [
            {"layer": selection.layer, "channels": selection.channels, "top_count": selection.top_count}
            for selection in selections
        ]

    summary["outputs"] = [str(path) for path in all_created]
    settings_path = output_dir / f"{ratio_label(args.top_ratio)}_heatmap_settings.json"
    settings_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_created.append(settings_path)

    print("Saved neuron heatmaps and CSV files:")
    for path in all_created:
        print(f"  {path}")


if __name__ == "__main__":
    main()

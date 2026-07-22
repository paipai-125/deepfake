# UniCaCLF TSN / BYOL-A neuron probe

This is a pairwise, online statistic pipeline. It does **not** save every
internal activation. For each fake/original pair, it captures every TSN
ResNet-50 Bottleneck or every BYOL-A block, pools its non-temporal dimensions,
averages within the annotated fake interval, and updates the paired channel
shift statistics. It saves all per-channel scores and top-ratio indices.

## Required encoders

The probing code loads the ResNet-50 backbone from the official TSN checkpoints
directly with `torchvision`; it does **not** require the legacy
`mmaction2==0.24.1` / `mmcv-full` stack.  Thus it works in the existing
PyTorch 2.6 + CUDA 12.4 environment.  Install only:

```bash
pip install opencv-contrib-python
git clone https://github.com/nttcslab/byol-a.git external/byol-a
```

Download the two TSN checkpoints listed in `../UMMAFormer/README.md` and the
official BYOL-A weight `AudioNTT2020-BYOLA-64x96d2048.pth`.  TV-L1 optical flow
requires `opencv-contrib-python`; `--flow-method farneback` is supported as a
fallback but is not a paper-faithful replacement for TV-L1.

## Commands

Run from `/root/autodl-tmp/Deepfake`.

```bash
python -m UniCaCLF.make_lavdf_subset \
  --lavdf-root Data/LAV-DF \
  --output UniCaCLF/manifests/lavdf_12k_2k_4k.json

python -m UniCaCLF.build_probe_pairs \
  --lavdf-root Data/LAV-DF \
  --subset UniCaCLF/manifests/lavdf_12k_2k_4k.json \
  --modality video --split train \
  --output UniCaCLF/manifests/video_only_train_pairs.jsonl

python -m UniCaCLF.build_probe_pairs \
  --lavdf-root Data/LAV-DF \
  --subset UniCaCLF/manifests/lavdf_12k_2k_4k.json \
  --modality audio --split train \
  --output UniCaCLF/manifests/audio_only_train_pairs.jsonl
```

Compute normalization only from the fixed train subset, then probe audio:

```bash
python -m UniCaCLF.compute_byola_norm_stats \
  --lavdf-root Data/LAV-DF \
  --subset UniCaCLF/manifests/lavdf_12k_2k_4k.json \
  --output UniCaCLF/manifests/byola_train_stats.json

python -m UniCaCLF.probe_byola_neurons \
  --pairs UniCaCLF/manifests/audio_only_train_pairs.jsonl \
  --byola-repo external/byol-a \
  --checkpoint checkpoints/AudioNTT2020-BYOLA-64x96d2048.pth \
  --norm-stats UniCaCLF/manifests/byola_train_stats.json \
  --output-dir results/unicaclf_probe/byola --top-ratio 0.10 --device cuda
```

Probe both TSN streams:

```bash
python -m UniCaCLF.probe_tsn_neurons \
  --pairs UniCaCLF/manifests/video_only_train_pairs.jsonl \
  --rgb-checkpoint checkpoints/tsn_rgb.pth \
  --flow-checkpoint checkpoints/tsn_flow.pth \
  --output-dir results/unicaclf_probe/tsn \
  --samples-per-period 16 --flow-method tvl1 --top-ratio 0.10 --amp --device cuda
```

Each output directory contains `*_neuron_scores.npz`, `*_top_neurons.csv`,
`*_layer_summary.csv`, a layer-shift figure, and a selected-neuron-count figure.
Both probe commands use the first 100 strict pairs by default.  Set
`--max-pairs N` for another size, or `--max-pairs 0` to probe every pair in the
manifest.

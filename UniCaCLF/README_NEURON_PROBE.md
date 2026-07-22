# UniCaCLF TSN / BYOL-A pairwise neuron probe

This pipeline probes frozen encoder channels with strict LAV-DF
`fake-original` pairs.  It does **not** save dense internal activations.
For each annotated fake interval it pools a layer output into one channel
vector per pair, accumulates `fake - real` with Welford statistics, and ranks
channels using:

\[
S_{l,d}=\frac{|\operatorname{mean}(\Delta h_{l,d})|}
 {\sqrt{\operatorname{std}(\Delta h_{l,d})^2+\epsilon}}.
\]

In code, `epsilon` is a data-dependent stability floor:
`max(1e-4, 0.05 × median(valid channel std))`; this prevents an almost
constant channel from obtaining an artificial infinite score.

Video pairs are strictly visual-only fakes; audio pairs are strictly
audio-only fakes.  RGB samples, Flow source frames, and audio all remain
inside the annotated fake interval.  Original media is sampled at the same
time interval.

## Server layout and requirements

Run every command from:

```bash
cd /root/autodl-tmp/Deepfake/deepfake/deepfake-code
conda activate deepfake
```

The expected paths on the current server are:

```text
../deepfake-data/LAV-DF/                         # dataset root
../deepfake-data/models/tsn/                     # RGB and Flow TSN weights
../deepfake-data/models/byola/                   # BYOL-A 2048-d weight
byol-a/                                           # cloned official BYOL-A repo
```

The TSN probe uses the current `torchvision` ResNet-50 and loads the official
MMAction checkpoint backbone weights directly.  It does not require legacy
`mmaction2` or `mmcv`.  `decord`, `torch`, `torchvision`, `torchaudio`,
`ffmpeg`, and `opencv-contrib-python` with `cv2.optflow` TV-L1 must be
available in the active environment.

## 1. Build a fixed subset and pair manifests

Generate pair manifests from the dataset root used for the current run:

```bash
mkdir -p ../deepfake-data/manifests ../deepfake-data/results/unicaclf_probe

python -m UniCaCLF.make_lavdf_subset \
  --lavdf-root ../deepfake-data/LAV-DF \
  --output ../deepfake-data/manifests/lavdf_18k.json \
  --counts train=12000 dev=2000 test=4000 \
  --seed 2026

python -m UniCaCLF.build_probe_pairs \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --modality video --split train \
  --output ../deepfake-data/manifests/video_only_probe_pairs.jsonl

python -m UniCaCLF.build_probe_pairs \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --modality audio --split train \
  --output ../deepfake-data/manifests/audio_only_probe_pairs.jsonl
```

Each `fake_period` becomes one pair.  With a 12,000-record balanced subset,
the strict video-only and audio-only manifests each contain roughly 3,000 fake
records (more if records have multiple fake periods).

## 2. Compute BYOL-A Log-Mel statistics

Run once whenever the training subset changes:

```bash
python -m UniCaCLF.compute_byola_norm_stats \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --split train \
  --output ../deepfake-data/manifests/byola_train_stats.json \
  --device cuda:0
```

## 3. Probe on one GPU

For a quick integrity check, append `--max-pairs 20`.  Omit it for the full
manifest; the default `--max-pairs 0` means all pairs.

```bash
python -m UniCaCLF.probe_byola_neurons \
  --pairs ../deepfake-data/manifests/audio_only_probe_pairs.jsonl \
  --byola-repo byol-a \
  --checkpoint ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth \
  --norm-stats ../deepfake-data/manifests/byola_train_stats.json \
  --output-dir ../deepfake-data/results/unicaclf_probe/byola \
  --top-ratio 0.10 --amp --device cuda:0

python -m UniCaCLF.probe_tsn_neurons \
  --pairs ../deepfake-data/manifests/video_only_probe_pairs.jsonl \
  --rgb-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth \
  --flow-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth \
  --output-dir ../deepfake-data/results/unicaclf_probe/tsn \
  --samples-per-period 16 --flow-method tvl1 \
  --top-ratio 0.10 --amp --device cuda:0
```

## 4. Probe on multiple GPUs

Use this only on a machine with at least two visible GPUs.  `torchrun` assigns
each pair once using `pairs[rank::world_size]`.  Each rank saves only
`count/mean/M2`; rank 0 merges these Welford states before it computes scores.
Do **not** manually launch several independent Python commands.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m UniCaCLF.probe_byola_neurons \
  --pairs ../deepfake-data/manifests/audio_only_probe_pairs.jsonl \
  --byola-repo byol-a \
  --checkpoint ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth \
  --norm-stats ../deepfake-data/manifests/byola_train_stats.json \
  --output-dir ../deepfake-data/results/unicaclf_probe/byola_4gpu \
  --top-ratio 0.10 --amp

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m UniCaCLF.probe_tsn_neurons \
  --pairs ../deepfake-data/manifests/video_only_probe_pairs.jsonl \
  --rgb-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth \
  --flow-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth \
  --output-dir ../deepfake-data/results/unicaclf_probe/tsn_4gpu \
  --samples-per-period 16 --flow-method tvl1 \
  --top-ratio 0.10 --amp
```

## Outputs and mandatory checks

Each output directory contains:

```text
*_neuron_scores.npz       full score, delta mean/std, and selected indices
*_top_neurons.csv         top-rho channel indices for every layer
*_layer_summary.csv       layer RMS shift and per-layer pair count
*_layer_shift.png         layer/top-neuron score visualization
*_selected_counts.png     selected neuron count per layer
*_run_audit.json          exact pair coverage and per-rank counts
*_failures.txt            present only when media decoding/model execution failed
```

Accept the result only if the audit satisfies:

```text
expected_pairs == attempted_pairs
successful_pairs + failed_pairs == expected_pairs
```

For a clean run, `failed_pairs` must be zero and no `*_failures.txt` should
exist.  The probe validates pair IDs before encoder inference; decoding and
model failures are recorded in the failure file.

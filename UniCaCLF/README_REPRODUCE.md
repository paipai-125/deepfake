# UniCaCLF baseline completion (LAV-DF)

The public UniCaCLF checkout was incomplete: its original `train_lavdf.py`
imports a private `dataset.py`, `contextformer/` package and `utils.nms`, and
its launch script contains paths from the authors' machine.  This directory
adds a runnable module while preserving the released `Contextformer` / `CaPFormer`,
FPN, proposal heads and contrastive loss.

## Encoder features to use

Reuse the scripts already present in `../UMMAFormer`.

* **Video**: TSN ResNet-50 RGB (2048) + Flow (2048) = **4096** dimensions.
* **Audio**: BYOL-A AudioNTT2020Feature = **2048** dimensions.

The old UniCaCLF shell file says `input_aud_dim=1024`, but does not contain a
BYOL-A model or a compatible 1024-D checkpoint.  Do not silently truncate
UMMAFormer's 2048-D BYOL-A features.  This completed baseline therefore uses
`--audio-dim 2048`; report it as *UniCaCLF with UMMAFormer TSN/BYOL-A features*,
not as a bit-exact result from an unavailable original feature archive.

Expected feature layout:

```
<feature-root>/
  tsn/rgb/{train,dev,test}/000001.npy
  tsn/flow/{train,dev,test}/000001.npy
  byola/{train,dev,test}/000001.npy
```

Each `.npy` is `T x C`.  The runner interpolates streams to 768 positions and
maps LAV-DF `fake_periods` (seconds) to the corresponding feature grid.

## Run

From `/root/autodl-tmp/Deepfake`:

```bash
python -m UniCaCLF.run_unicaclf_baseline --mode train \
  --metadata Data/LAV-DF/metadata.json \
  --feature-root data/lavdf/feats \
  --output-dir results/unicaclf_tsn_byola \
  --epochs 10 --batch-size 16 --audio-dim 2048 --device cuda

python -m UniCaCLF.run_unicaclf_baseline --mode eval \
  --metadata Data/LAV-DF/metadata.json \
  --feature-root data/lavdf/feats \
  --output-dir results/unicaclf_tsn_byola/test \
  --checkpoint results/unicaclf_tsn_byola/best.pt \
  --audio-dim 2048 --device cuda
```

`best.pt` is selected using dev `mAP@0.50:0.95`; test results go to
`test_metrics.json`.

# UniCaCLF 实验交接说明

所有代码均在 `deepfake-code/` 目录下执行。

## 1. 环境

组合为 Python 3.10、PyTorch 2.6.0 CUDA 12.4、torchvision/torchaudio 2.6.0 和 RTX 4090。

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

## 2. 数据集与模型下载

### LAV-DF

将官方 LAV-DF 数据集完整下载到 `../deepfake-data/LAV-DF/`，必须保留官方目录

```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com
hf download ControlNet/LAV-DF --repo-type dataset --local-dir ../deepfake-data/LAV-DF
```

### TSN 权重

```bash
mkdir -p ../deepfake-data/models/tsn ../deepfake-data/models/byola

wget -c -P ../deepfake-data/models/tsn https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth

wget -c -P ../deepfake-data/models/tsn https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth
```

### BYOL-A 权重

下载官方 BYOL-A 仓库中的预训练权重`AudioNTT2020-BYOLA-64x96d2048.pth`

```bash
wget -c -O ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth
```

## 3. 创建固定 12k/2k/4k 子集与探测 pair

`make_lavdf_subset.py`、`build_probe_pairs.py` 仅处理 JSON 元数据，不使用 GPU，保留单进程运行。

```bash
mkdir -p ../deepfake-data/manifests ../deepfake-data/results

# 创新18k数据子集
python -m UniCaCLF.make_lavdf_subset \
  --lavdf-root ../deepfake-data/LAV-DF \
  --output ../deepfake-data/manifests/lavdf_18k.json \
  --counts train=12000 dev=2000 test=4000 --seed 2026

# 构造探测数据对（视频）
python -m UniCaCLF.build_probe_pairs \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --modality video --split train \
  --output ../deepfake-data/manifests/video_only_probe_pairs.jsonl

# 构造探测数据对（音频）
python -m UniCaCLF.build_probe_pairs \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --modality audio --split train \
  --output ../deepfake-data/manifests/audio_only_probe_pairs.jsonl

# 计算BYOL-A的标准化统计量
torchrun --standalone --nproc_per_node=8 \
  -m UniCaCLF.compute_byola_norm_stats \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --output ../deepfake-data/manifests/byola_train_stats.json \
  --device cuda
```

video pair 仅使用“视频伪造、音频真实”样本；audio pair 仅使用“音频伪造、视频真实”样本。每个 pair 在伪造区间取样，并在 original 中取时间一致的真实片段。


## 4. 神经元探测

`--max-pairs 100  --sample-seed 2026` 固定随机抽样探测100个pair

```bash
# 探测音频编码器神经元
torchrun --standalone --nproc_per_node=8 \
  -m UniCaCLF.probe_byola_neurons \
  --pairs ../deepfake-data/manifests/audio_only_probe_pairs.jsonl \
  --byola-repo byol-a \
  --checkpoint ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth \
  --norm-stats ../deepfake-data/manifests/byola_train_stats.json \
  --output-dir ../deepfake-data/results/unicaclf_probe/byola \
  --top-ratio 0.10 --amp --device cuda \
  --max-pairs 100  --sample-seed 2026

# 探测视频编码器神经元
torchrun --standalone --nproc_per_node=8 \
  -m UniCaCLF.probe_tsn_neurons \
  --pairs ../deepfake-data/manifests/video_only_probe_pairs.jsonl \
  --rgb-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth \
  --flow-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth \
  --output-dir ../deepfake-data/results/unicaclf_probe/tsn \
  --samples-per-period 16 --flow-method tvl1 \
  --top-ratio 0.10 --amp --device cuda \
  --max-pairs 100  --sample-seed 2026
```

各输出目录中的 `*_neuron_scores.npz` 保存各层分数和 `*_top_indices`；`*_top_neurons.csv` 及 PNG 是可视化结果。


## 5. 离线提取 final 与 neuron 表征

```bash
torchrun --standalone --nproc_per_node=8 \
  -m UniCaCLF.extract_unicaclf_offline_features \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --output-root ../deepfake-data/cache/unicaclf_lavdf_18k \
  --representation both --splits train dev test \
  --rgb-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth \
  --flow-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth \
  --byola-repo byol-a \
  --byola-checkpoint ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth \
  --byola-norm-stats ../deepfake-data/manifests/byola_train_stats.json \
  --tsn-scores ../deepfake-data/results/unicaclf_probe/tsn/tsn_neuron_scores.npz \
  --byola-scores ../deepfake-data/results/unicaclf_probe/byola/byola_neuron_scores.npz \
  --video-stride-frames 4   --video-batch-size 32  --decode-threads 2 \
  --flow-method tvl1 --amp --device cuda
```

输出路径为：

```text
../deepfake-data/cache/unicaclf_lavdf_18k/final/    # RGB 2048 + Flow 2048；BYOL-A 2048
../deepfake-data/cache/unicaclf_lavdf_18k/neurons/  # top-rho 内部神经元拼接表征
```

Neuron 表征处理为：TSN 空间平均 / BYOL-A Conv 频率平均 → top-rho 通道选择 → 线性插值到 TSN 时间轴 → 每通道时序 z-score → 层拼接。

## 6. UniCaCLF 训练与测试

论文配置为 Adam、lr=1e-3、batch size=8、betas=(0.9,0.999)、eps=1e-8、6 个 CaP/FPN 层、100 epochs。
`--batch-size` 表示**每张卡**的 batch size。为保持论文的全局 batch size=8，8 卡命令中使用 `--batch-size 1`。

### Final 表征 baseline

```bash
# 训练
torchrun --standalone --nproc_per_node=8 -m UniCaCLF.run_unicaclf_baseline \
  --mode train --metadata ../deepfake-data/manifests/lavdf_18k.json \
  --feature-root ../deepfake-data/cache/unicaclf_lavdf_18k/final \
  --output-dir ../deepfake-data/results/unicaclf_final_baseline \
  --epochs 100 --batch-size 1 --lr 1e-3 --max-seq-len 768 \
  --video-dim 4096 --audio-dim 2048 --workers 2 --device cuda

# 测试
torchrun --standalone --nproc_per_node=8 -m UniCaCLF.run_unicaclf_baseline \
  --mode eval --metadata ../deepfake-data/manifests/lavdf_18k.json \
  --feature-root ../deepfake-data/cache/unicaclf_lavdf_18k/final \
  --output-dir ../deepfake-data/results/unicaclf_final_baseline/test \
  --checkpoint ../deepfake-data/results/unicaclf_final_baseline/best.pt \
  --video-dim 4096 --audio-dim 2048 --max-seq-len 768 --workers 2 --device cuda
```

### top-rho 神经元表征

读取 `../deepfake-data/cache/unicaclf_lavdf_18k/extraction_settings.json` 中的 `neuron_video_dim`、`neuron_audio_dim`

```bash
python -c "import json; d=json.load(open('../deepfake-data/cache/unicaclf_lavdf_18k/extraction_settings.json')); print('neuron_video_dim =', d['neuron_video_dim']); print('neuron_audio_dim =', d['neuron_audio_dim'])"
```

如果结果和下列指令中的 3038、431 不一致，则替换为实际值：

```bash
torchrun --standalone --nproc_per_node=8 -m UniCaCLF.run_unicaclf_baseline \
  --mode train --metadata ../deepfake-data/manifests/lavdf_18k.json \
  --feature-root ../deepfake-data/cache/unicaclf_lavdf_18k/neurons \
  --output-dir ../deepfake-data/results/unicaclf_top10_neurons \
  --epochs 100 --batch-size 1 --lr 1e-3 --max-seq-len 768 \
  --video-dim 3038 --audio-dim 431 --workers 2 --device cuda

torchrun --standalone --nproc_per_node=8 -m UniCaCLF.run_unicaclf_baseline \
  --mode eval --metadata ../deepfake-data/manifests/lavdf_18k.json \
  --feature-root ../deepfake-data/cache/unicaclf_lavdf_18k/neurons \
  --output-dir ../deepfake-data/results/unicaclf_top10_neurons/test \
  --checkpoint ../deepfake-data/results/unicaclf_top10_neurons/best.pt \
  --video-dim 3038 --audio-dim 431 --max-seq-len 768 --workers 2 --device cuda
```

测试输出 `test_metrics.json` 包含：AP@0.50、AP@0.75、AP@0.90、AP@0.95、mAP@0.50:0.95、AR@5、AR@10、AR@20、AR@30、AR@50

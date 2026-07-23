# TV-L1 Flow 预计算

在 `deepfake-code/` 目录执行。

```bash
# cpu 预计算 TV-L1 Flow
torchrun --standalone --nproc_per_node=8 \
  -m flow_preprocess.precompute_tvl1_flow \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --output-root ../deepfake-data/cache/lavdf_tvl1_flow \
  --splits train dev test \
  --video-stride-frames 4 \
  --image-size 256 \
  --flow-bound 20 \
  --chunk-size 32 \
  --decode-threads 2

# 读取缓存并抽取 final + neuron 特征
torchrun --standalone --nproc_per_node=8 \
  -m flow_preprocess.extract_unicaclf_features_cached_flow \
  --flow-cache-root ../deepfake-data/cache/lavdf_tvl1_flow \
  --lavdf-root ../deepfake-data/LAV-DF \
  --subset ../deepfake-data/manifests/lavdf_18k.json \
  --output-root ../deepfake-data/cache/lavdf_tvl1_flow \
  --representation both --splits train dev test \
  --rgb-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth \
  --flow-checkpoint ../deepfake-data/models/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth \
  --byola-repo byol-a \
  --byola-checkpoint ../deepfake-data/models/byola/AudioNTT2020-BYOLA-64x96d2048.pth \
  --byola-norm-stats ../deepfake-data/manifests/byola_train_stats.json \
  --tsn-scores ../deepfake-data/results/unicaclf_probe/tsn/tsn_neuron_scores.npz \
  --byola-scores ../deepfake-data/results/unicaclf_probe/byola/byola_neuron_scores.npz \
  --video-stride-frames 4 \
  --video-batch-size 32 \
  --decode-threads 2 \
  --flow-method tvl1 \
  --amp --device cuda
```

输出路径为：

```text
../deepfake-data/cache/lavdf_tvl1_flow/final/    # RGB 2048 + Flow 2048；BYOL-A 2048
../deepfake-data/cache/lavdf_tvl1_flow/neurons/  # top-rho 内部神经元拼接表征
```

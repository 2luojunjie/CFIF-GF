# CFIF-GF

[中文文档](README_zh.md) | English

PyTorch speech emotion recognition (SER) project for reproducing and extending
the paper's WavLM_Att and CFIF-GF workflows on IEMOCAP and EMODB.

The current code includes dataset preprocessing, manifest loading or dataset
auto-discovery, 10-fold leave-one-speaker-out (LOSO) training, logging, metrics,
and implementations of the `WavLM_Att` and `CFIF-GF` model structures.

## Project Structure

```text
CFIF-GF/
  configs/          YAML experiment configs
  data/             Dataset discovery, preprocessing, folds, dataloaders
  models/           Model registry and placeholder models
  scripts/          Helper scripts
  utils/            Config, logging, seed, and metrics utilities
  train.py          Training and LOSO training entry point
  evaluate.py       Evaluation entry point
  requirements.txt  Python dependencies
```

## Install

```bash
pip install -r requirements.txt
```

Recommended server environment:

- Python 3.9+
- PyTorch with CUDA matching the server driver
- Single GPU is supported by default
- HuggingFace `transformers` downloads `microsoft/wavlm-base` on first use

## Audio Preprocessing

All samples are processed consistently in `data/preprocessing.py`:

- Resample audio to 16 kHz.
- Normalize each utterance to 3 seconds.
- Zero-pad utterances shorter than 3 seconds.
- Truncate utterances longer than 3 seconds.
- Extract 40-dimensional MFCC features.
- MFCC window size: 40 ms.
- MFCC hop length: 10 ms.
- MFCC uses a Hamming window.
- Extract magnitude spectrogram features.
- Spectrogram FFT length: 800.
- Use the first 200 FFT bins as spectrogram input.
- Spectrogram STFT uses a Hamming window.
- Keep the raw fixed-length waveform for WavLM input.
- Optional pre-emphasis can be enabled for handcrafted features with
  `dataset.preprocessing.pre_emphasis`.

`SpeechEmotionDataset` returns:

- `waveform`
- `mfcc`
- `spectrogram`
- `label`
- `speaker_id`
- `file_path`
- `wavlm_features`, empty by default or loaded from `wavlm_path`

## Datasets

Supported dataset names:

- `IEMOCAP`
- `EMODB`

The loader supports two modes.

### Manifest Mode

Set `dataset.mock: false` and provide `dataset.all_manifest` for LOSO training,
or `train_manifest` / `test_manifest` for ordinary training/evaluation.

CSV columns:

```csv
path,label,speaker_id,wavlm_path
/path/to/audio.wav,angry,Session1_F,
/path/to/audio2.wav,0,Session1_M,/path/to/offline_wavlm.pt
```

`label` may be either a class name from `dataset.label_names` or an integer class
index. `wavlm_path` is optional. When present, the model uses offline WavLM
sequence features instead of running WavLM during training.

### Auto-Discovery Mode

If `dataset.mock: false` and `dataset.all_manifest` is empty, the code can scan
standard dataset layouts:

- IEMOCAP: reads `Session*/dialog/EmoEvaluation/*.txt` and
  `Session*/sentences/wav/**/*.wav`.
- EMODB: scans `.wav` files and parses speaker/emotion from official file names.

For IEMOCAP, `exc` is merged into `happy`, matching the common 4-class SER setup:
`angry`, `happy`, `sad`, `neutral`.

## 10-Fold LOSO Training

Leave-one-speaker-out folds are built by `speaker_id`:

- One speaker is used as the evaluation/test set in each fold.
- All other speakers are used for training.
- The default expected number of folds is 10.
- Each fold saves `best.pt`, `last.pt`, `metrics.csv`, and `result.json`.
- If `--fold` is omitted, all 10 folds are trained automatically.
- If `--fold 0` is provided, only the first fold is trained.
- After all folds, average WA, UA, and F1 are written to
  `cross_validation_summary.json`.

Train a single fold:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml --fold 0
```

Train all folds:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml
python train.py --config configs/emodb_cfif_gf.yaml
```

Resume single-fold training:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml --fold 0 --resume outputs/IEMOCAP/CFIF-GF/fold_00_<speaker_id>/last.pt
```

Output layout:

```text
outputs/<DATASET>/<MODEL>/
  fold_00_<speaker_id>/
    best.pt
    last.pt
    metrics.csv
    result.json
  cross_validation_summary.json
```

## Mock Smoke Test

The default config uses `dataset.mock: true`, so the pipeline can run without
real audio data:

```bash
python train.py --config configs/default.yaml --fold 0
```

## Metrics

The paper reports weighted accuracy (WA), unweighted accuracy (UA), and F1.
These are computed in `utils/metrics.py` as:

- `wa`: overall accuracy.
- `ua`: mean per-class recall.
- `macro_f1`: mean per-class F1.
- `confusion_matrix`: rows are true classes and columns are predicted classes.
- `per_class`: precision, recall, F1, and support for every emotion class.

## Models

- `WavLM_Att`
- `CFIF-GF`

## WavLM_Att Model

`models/wavlm_att.py` implements the chapter 3 model:

- Raw `waveform` is passed to HuggingFace `microsoft/wavlm-base`.
- The implementation uses HuggingFace `AutoModel`, so Wav2Vec2 and HuBERT
  backbones can also be selected through `model.wavlm_name`.
- WavLM parameters are frozen by default with `model.freeze_wavlm: true`.
- Set `model.freeze_wavlm: false` to fine-tune WavLM.
- `mfcc` is encoded by a bidirectional LSTM.
- MFCC BiLSTM hidden size is 256.
- MFCC BiLSTM layer count is configurable with `model.mfcc_num_layers`.
- `spectrogram` is encoded by an AlexNet-style CNN.
- CNN channels are 64, 192, 384, 256, 256.
- CNN kernels use 11 in the first convolution and 3 in later convolutions.
- The co-attention module uses MFCC and spectrogram features to generate
  temporal attention weights over the WavLM output sequence.
- Attended WavLM, MFCC, and spectrogram features are concatenated for
  classification.

The model forward pass returns logits for `CrossEntropyLoss`. For inference
probabilities, use `model.predict_proba(...)`, which applies Softmax.

## CFIF-GF Model

`models/cfif_gf.py` implements the chapter 4 model:

- Raw `waveform` is passed to HuggingFace `microsoft/wavlm-base` to obtain
  WavLM sequence features `X_w`.
- WavLM is frozen by default with `model.freeze_wavlm: true`.
- `mfcc` is encoded by a Bi-LSTM to obtain MFCC sequence features `X_m`.
- `spectrogram` is encoded by TFCNN to obtain spectrogram sequence features
  `X_s`.
- TFCNN uses two parallel Conv2d branches:
  T-CNN with a `5 x 1` kernel and F-CNN with a `1 x 4` kernel.
- Each TFCNN branch contains Conv2d, BatchNorm2d, ReLU, and MaxPool2d.
- Branch outputs are concatenated, passed through `3 x 3` convolution blocks,
  then AdaptiveAvgPool and Linear produce spectrogram feature sequences.
- CFIF has two cross-feature interaction branches:
  `MFCC -> WavLM` and `Spectrogram -> WavLM`.
- Each CFIF branch maps source and WavLM features to a common hidden dimension,
  applies broadcast add, tanh, Linear, and softmax to compute interaction
  attention over the WavLM sequence.
- The attended WavLM features are concatenated with the source sequence and
  projected to a common dimension.
- GF is a gMLP-style global fusion block with LayerNorm, Linear, GELU, feature
  split, LayerNorm plus `1 x 1` Conv gating, elementwise multiplication, Linear,
  and residual connection.
- The pooled global fusion feature is sent to the final fully connected
  classifier.

Run CFIF-GF LOSO training with:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml
python train.py --config configs/emodb_cfif_gf.yaml
```

## Training Commands

Full workflow:

1. Check the dataset:

```bash
python scripts/check_dataset.py --config configs/iemocap_cfif_gf.yaml
```

2. Optional: cache WavLM features offline:

```bash
python scripts/extract_wavlm_features.py --config configs/iemocap_cfif_gf.yaml --output-dir features/iemocap_wavlm --output-manifest manifests/iemocap_wavlm.csv
```

3. Debug forward shapes:

```bash
python scripts/debug_forward.py --config configs/iemocap_cfif_gf.yaml --batch-size 2
```

4. Train one fold:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml --fold 0
```

5. Train all 10 folds:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml --all-folds
```

6. Summarize results manually if needed:

```bash
python scripts/summarize_results.py --results-dir outputs/IEMOCAP/CFIF-GF
```

Train WavLM_Att:

```bash
python train.py --config configs/iemocap_wavlm_att.yaml
python train.py --config configs/emodb_wavlm_att.yaml
```

Train CFIF-GF:

```bash
python train.py --config configs/iemocap_cfif_gf.yaml
python train.py --config configs/emodb_cfif_gf.yaml
```

Ablation examples:

```bash
python train.py --config configs/ablation/cfif_gf_full.yaml --all-folds
python train.py --config configs/ablation/cfif_without_gf.yaml --all-folds
python train.py --config configs/ablation/mha_fusion.yaml --all-folds
```

Available ablation configs:

Chapter 3:

- `configs/ablation_ch3/wavlm_att_full.yaml`
- `configs/ablation_ch3/wavlm_only.yaml`
- `configs/ablation_ch3/wavlm_concat.yaml`
- `configs/ablation_ch3/wavlm_mha.yaml`
- `configs/ablation_ch3/wav2vec2_att.yaml`
- `configs/ablation_ch3/hubert_att.yaml`
- `configs/ablation_ch3/wav2vec2_only.yaml`
- `configs/ablation_ch3/hubert_only.yaml`

Chapter 4:

- `configs/ablation/cfif_gf_full.yaml`
- `configs/ablation/cfif_mfcc_to_wavlm.yaml`
- `configs/ablation/cfif_spec_to_wavlm.yaml`
- `configs/ablation/cfif_wavlm_to_mfcc_spec.yaml`
- `configs/ablation/concat_fusion.yaml`
- `configs/ablation/mha_fusion.yaml`
- `configs/ablation/cfif_without_gf.yaml`
- `configs/ablation/cfif_wav2vec2.yaml`
- `configs/ablation/cfif_hubert.yaml`

Default dataset hyperparameters:

- IEMOCAP: learning rate `2e-5`, batch size `32`, epochs `100`, AdamW.
- EMODB: learning rate `3e-5`, batch size `64`, epochs `100`, AdamW.

Early stopping is configured in YAML under `train.early_stopping`. The default
monitor is `wa`, with patience `10`.

CSV logging is enabled by default. To use TensorBoard, set:

```yaml
train:
  log_backend: tensorboard
```

AMP and gradient clipping are configured in YAML:

```yaml
train:
  amp: true
  grad_clip_norm: 5.0
```

## Evaluation

Evaluate a best checkpoint:

```bash
python evaluate.py --config configs/iemocap_cfif_gf.yaml --checkpoint outputs/IEMOCAP/CFIF-GF/fold_00_<speaker_id>/best.pt
```

The evaluator prints WA, UA, Macro F1, confusion matrix, and per-class emotion
metrics. It also writes `<checkpoint>.eval.json`.

## Offline WavLM Features

If GPU memory is tight, extract WavLM sequence features before training:

```bash
python scripts/extract_wavlm_features.py --config configs/iemocap_cfif_gf.yaml --output-dir features/iemocap_wavlm --output-manifest manifests/iemocap_wavlm.csv
```

Then set:

```yaml
dataset:
  mock: false
  all_manifest: manifests/iemocap_wavlm.csv
model:
  use_offline_wavlm_features: true
  offline_wavlm_dim: 768
```

The manifest's `wavlm_path` column will be used automatically. With
`use_offline_wavlm_features: true`, the training model does not load WavLM, which
reduces GPU memory usage.

## Engineering Assumptions

- In LOSO training, the held-out speaker is used as the fold evaluation/test
  split for early stopping and final reporting.
- IEMOCAP `exc` is merged into `happy`.
- When paper implementation details are underspecified, the code favors a
  runnable PyTorch implementation with configurable dimensions in YAML.

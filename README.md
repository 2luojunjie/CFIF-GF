# CFIF-GF

PyTorch speech emotion recognition (SER) project for reproducing and extending
the paper's WavLM_Att and CFIF-GF workflows on IEMOCAP and EMODB.

The current code includes dataset preprocessing, manifest loading or dataset
auto-discovery, 10-fold leave-one-speaker-out (LOSO) training, logging, metrics,
and placeholder model classes. The real `WavLM_Att` and `CFIF-GF` model internals
can be implemented later inside `models/`.

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

## Audio Preprocessing

All samples are processed consistently in `data/preprocessing.py`:

- Resample audio to 16 kHz.
- Normalize each utterance to 3 seconds.
- Zero-pad utterances shorter than 3 seconds.
- Truncate utterances longer than 3 seconds.
- Extract 40-dimensional MFCC features.
- MFCC window size: 40 ms.
- MFCC hop length: 10 ms.
- Extract magnitude spectrogram features.
- Spectrogram FFT length: 800.
- Use the first 200 FFT bins as spectrogram input.
- Keep the raw fixed-length waveform for WavLM input.

`SpeechEmotionDataset` returns:

- `waveform`
- `mfcc`
- `spectrogram`
- `label`
- `speaker_id`
- `file_path`

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
path,label,speaker_id
/path/to/audio.wav,angry,Session1_F
/path/to/audio2.wav,0,Session1_M
```

`label` may be either a class name from `dataset.label_names` or an integer class
index.

### Auto-Discovery Mode

If `dataset.mock: false` and `dataset.all_manifest` is empty, the code can scan
standard dataset layouts:

- IEMOCAP: reads `Session*/dialog/EmoEvaluation/*.txt` and
  `Session*/sentences/wav/**/*.wav`.
- EMODB: scans `.wav` files and parses speaker/emotion from official file names.

For IEMOCAP, `exc` is merged into `happy`, matching the common 4-class SER setup:
`angry`, `happy`, `neutral`, `sad`.

## 10-Fold LOSO Training

Leave-one-speaker-out folds are built by `speaker_id`:

- One speaker is used as the test set in each fold.
- All other speakers are used for training.
- The default expected number of folds is 10.
- Each fold saves its own checkpoint and metrics.
- After all folds, average WA, UA, and F1 are written to `loso_summary.json`.

Run:

```bash
python train.py --config configs/iemocap.yaml --loso
python train.py --config configs/emodb.yaml --loso
```

Outputs:

```text
outputs/loso/
  fold_01_<speaker_id>/
    checkpoint.pt
    metrics.json
  ...
  loso_summary.json
```

## Mock Smoke Test

The default config uses `dataset.mock: true`, so the pipeline can run without
real audio data:

```bash
python train.py --config configs/default.yaml --loso
python evaluate.py --config configs/default.yaml
```

## Metrics

The paper reports weighted accuracy (WA), unweighted accuracy (UA), and F1.
These are computed in `utils/metrics.py` as:

- `wa`: overall accuracy.
- `ua`: mean per-class recall.
- `macro_f1`: mean per-class F1.

## Planned Models

- `WavLM_Att`
- `CFIF-GF`

## WavLM_Att Model

`models/wavlm_att.py` implements the chapter 3 model:

- Raw `waveform` is passed to HuggingFace `microsoft/wavlm-base`.
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

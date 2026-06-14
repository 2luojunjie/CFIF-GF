# CFIF-GF

PyTorch project skeleton for speech emotion recognition (SER), prepared for the
IEMOCAP and EMODB datasets and future implementations of `WavLM_Att` and
`CFIF-GF`.

This first version contains a runnable mock training/evaluation pipeline:
configuration loading, random seed setup, logging, basic metrics, placeholder
datasets, and placeholder model classes.

The paper describes evaluation on IEMOCAP and EMODB with weighted accuracy
(WA), unweighted accuracy (UA), and F1. These metric names are already exposed
in `utils/metrics.py`.

## Project Structure

```text
CFIF-GF/
  configs/          YAML experiment configs
  data/             Dataset builders and SER dataset placeholders
  models/           Model registry and placeholder models
  scripts/          Helper scripts
  utils/            Config, logging, seed, and metrics utilities
  train.py          Training entry point
  evaluate.py       Evaluation entry point
  requirements.txt  Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
python evaluate.py --config configs/default.yaml
```

The default config uses `dataset.mock: true`, so it does not require real audio
files yet. Set `dataset.mock: false` and provide manifests later when the real
IEMOCAP/EMODB data pipelines are implemented.

## Supported Dataset Names

- `IEMOCAP`
- `EMODB`

## Planned Model Names

- `WavLM_Att`
- `CFIF-GF`

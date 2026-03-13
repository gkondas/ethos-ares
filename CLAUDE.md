# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ETHOS-ARES is an EHR foundation model (GPT2-based, no bias) for zero-shot prediction of patient health outcomes. It trains on tokenized Patient Health Timelines (PHT) derived from MIMIC-IV and supports tasks like hospital mortality, ICU admission, readmission, and more. ARES (Adaptive Risk Estimation System) layers on top for explainability.

## Installation

```bash
conda create --name ethos python=3.12 && conda activate ethos
pip install -e .[jupyter]   # or use uv run for one-off commands
```

## CLI commands (registered entry points)

| Command | Entry point |
|---|---|
| `ethos_tokenize` | `ethos.tokenize.run_tokenization:main` |
| `ethos_train` | `ethos.train.run_training:main` |
| `ethos_infer` | `ethos.inference.run_inference:main` |

All three use **Hydra** for config. Override params directly on the CLI. Set `HYDRA_FULL_ERROR=1` to get full tracebacks.

## Running tests

```bash
pytest
```

## Key pipeline stages

### 1. Pre-tokenization (MEDS extraction)
Raw MIMIC-IV → MEDS format via `scripts/meds/run_mimic.sh`. Data is split into `train`, `tuning`, `held_out` fractions configured in `scripts/meds/mimic/configs/extract_MIMIC.yaml`.

### 2. Tokenization
```bash
# Train split (builds vocabulary)
ethos_tokenize -m worker='range(0,7)' input_dir=.../train output_dir=... out_fn=train

# Other splits (reuse train vocab)
ethos_tokenize -m worker='range(0,2)' input_dir=.../tuning vocab=.../train output_dir=... out_fn=tuning
```
Output is sharded `.safetensors` files per split, plus a `vocab_t*.csv`, `static_data.pkl`, `interval_estimates.json`, and `quantile_breaks.csv`.

### 3. Training
```bash
# Single GPU
ethos_train data_fp=.../train out_dir=.../models/my_run

# Multi-GPU (DDP)
torchrun --standalone --nproc_per_node=8 ethos_train data_fp=.../train ...
```
The SLURM script is at `ethos-scripts/train-ethos.sh`. Uses `uv run ethos_train`.

**Validation data**: configured via `val_data_fp` in `src/ethos/configs/training.yaml` (defaults to the `tuning` split). If `val_data_fp` is `null`, the training set is split at runtime using `val_size` (integer = millions of tokens; float = fraction).

Checkpoints saved to `out_dir/`: `recent_model.pt` (every eval) and `best_model.pt` (best val loss).

### 4. Inference
```bash
ethos_infer task=hospital_mortality model_fp=.../best_model.pt input_dir=.../test output_dir=results/... rep_num=32 n_gpus=8
```
Available tasks are defined in `src/ethos/inference/constants.py` (`Task` enum).

## Architecture

### Data layer
- **`ShardedData`** (`datasets/_sharded_data.py`): lazy-loads sharded `.safetensors` files. Provides token/time/patient_id access across shards transparently.
- **`TimelineDataset`** (`datasets/base.py`): wraps `ShardedData`. Each sample is `(x, y)` where `x = [patient_context | timeline[:-1]]` and `y = [masked_context | timeline[1:]]`. The patient context tokens (age, sex, etc.) are always prepended and masked in the loss (`y[:context_size] = -100`).
- **`InferenceDataset`** (also `datasets/base.py`): subclass used at inference time; indexed by token position rather than sliding window.

### Model
- **`GPT2LMNoBiasModel`** (`model.py`): thin wrapper around HuggingFace `GPT2LMHeadModel` with bias disabled. Also supports encoder-decoder (`EncoderDecoderModel`) via `model_type: enc_decoder`.
- `torch.compile` is enabled by default; disable with `no_compile=true`.

### Tokenization pipeline
- Configured by `src/ethos/configs/tokenization.yaml` + dataset-specific configs under `configs/dataset/`.
- Runs multi-stage transforms (defined in `tokenize/mimic/preprocessors.py` and `tokenize/common/`) via Hydra multirun (`-m worker='range(0,N)'`).
- Vocabulary is built once on the `train` split; all other splits reference it with `vocab=.../train`.

### Inference
- Zero-shot: model autoregressively generates future tokens from a patient's history until a stop token or time limit is reached.
- Multiple repetitions (`rep_num`) give probability estimates via sampling.
- Results written as `.parquet` files to `output_dir`.

## Config files

| File | Purpose |
|---|---|
| `src/ethos/configs/training.yaml` | All training hyperparameters; `val_data_fp` defaults to the `tuning` split |
| `src/ethos/configs/inference.yaml` | Inference defaults (rep_num, temperature, subset, etc.) |
| `src/ethos/configs/tokenization.yaml` | Tokenization pipeline config |
| `src/ethos/configs/dataset/mimic.yaml` | MIMIC-specific tokenization stages |

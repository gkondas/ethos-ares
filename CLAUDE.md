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

No test suite exists currently.

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
Output per split: sharded `.safetensors` files, `vocab_t*.csv`, `static_data.pickle`, `interval_estimates.json`, and `quantile_breaks.csv`.

### 3. Training
```bash
# Single GPU
ethos_train data_fp=.../train out_dir=.../models/my_run

# Multi-GPU DDP (single node)
torchrun --standalone --nproc_per_node=8 ethos_train data_fp=.../train ...

# Multi-GPU DDP (multi-node, run on each node)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=1234 ethos_train ...

# Resume from checkpoint
ethos_train data_fp=... out_dir=... resume=true
```
The SLURM script is at `ethos-scripts/train-ethos.sh`. Uses `uv run ethos_train`.

**Validation data**: configured via `val_data_fp` in `src/ethos/configs/training.yaml` (defaults to the `tuning` split). If `val_data_fp` is `null`, the training set is split at runtime using `val_size` (integer = millions of tokens; float = fraction).

Checkpoints saved to `out_dir/`: `recent_model.pt` (every eval) and `best_model.pt` (best val loss). Each checkpoint stores `model`, `optimizer`, `iter_num`, `best_val_loss`, `model_config`, `vocab`, `model_type`, and `wandb_path`.

### 4. Inference
```bash
ethos_infer task=hospital_mortality model_fp=.../best_model.pt input_dir=.../test output_dir=results/... rep_num=32 n_gpus=8
```
Available tasks are defined in `src/ethos/inference/constants.py` (`Task` enum). Key inference params: `rep_num` (repetitions for probability estimates), `n_gpus`, `n_jobs` (workers per GPU), `subset` (fraction or count), `temperature`, `dataset_kwargs` (passed to dataset constructor), `save_generated_tokens`.

## Architecture

### Data layer
- **`ShardedData`** (`datasets/_sharded_data.py`): lazy-loads sharded `.safetensors` files. Provides token/time/patient_id access across shards transparently.
- **`TimelineDataset`** (`datasets/base.py`): wraps `ShardedData`. Each sample is `(x, y)` where `x = [patient_context | timeline[:-1]]` and `y = [masked_context | timeline[1:]]`. The patient context tokens (age, sex, etc.) are always prepended and masked in the loss (`y[:context_size] = -100`). Context size is derived from `static_data.pickle`.
- **`InferenceDataset`** (also `datasets/base.py`): abstract subclass used at inference time; indexed by token position rather than sliding window. Concrete subclasses live in `datasets/` (e.g., `hospital_mortality.py`, `readmission.py`, `mimic_icu.py`, `ed.py`).

### Model
- **`GPT2LMNoBiasModel`** (`model.py`): custom GPT2 implementation from scratch (not a HF model wrapper). Uses `GPT2Config` only for the config dataclass. Flash Attention is used automatically when available (PyTorch >= 2.0). Pass `return_attention=True` to collect attention weights for ARES explainability. Also supports encoder-decoder (`EncoderDecoderModel` from HuggingFace) via `model_type: enc_decoder`.
- `torch.compile` is enabled by default; disable with `no_compile=true`.
- Vocabulary size is rounded up to the nearest multiple of 64 for efficiency.

### Token vocabulary
- **`Vocabulary`** (`vocabulary.py`): built from `vocab_t*.csv`; loaded via `Vocabulary.from_path(dir)` (LRU-cached). Provides `encode`/`decode`, `quantile_stokens` (age encoding tokens starting with `Q`), and `time_interval_stokens`.
- **`SpecialToken`** (`constants.py`): string enum for special tokens — `MEDS_BIRTH`, `MEDS_DEATH`, `TIMELINE_END`, `HOSPITAL_ADMISSION/DISCHARGE`, `ICU_ADMISSION/DISCHARGE`, `ED_REGISTRATION/OUT`, `SOFA`.

### Tokenization pipeline
- Configured by `src/ethos/configs/tokenization.yaml` + dataset-specific configs under `configs/dataset/`.
- Runs multi-stage transforms (defined in `tokenize/mimic/preprocessors.py` and `tokenize/common/`) via Hydra multirun (`-m worker='range(0,N)'`).
- Vocabulary is built once on the `train` split; all other splits reference it with `vocab=.../train`.

### Inference
- Zero-shot: model autoregressively generates future tokens from a patient's history until a stop token or time limit is reached.
- Multiple repetitions (`rep_num`) give probability estimates via sampling.
- Results written as `.parquet` files to `output_dir`. Output directory name is auto-suffixed with subset info, temperature, and wandb run ID.
- Workers are spawned via Python `multiprocessing` (`spawn` start method); GPU assignment is round-robin across `n_gpus`.

## Config files

| File | Purpose |
|---|---|
| `src/ethos/configs/training.yaml` | All training hyperparameters; `val_data_fp` defaults to the `tuning` split |
| `src/ethos/configs/inference.yaml` | Inference defaults (rep_num, temperature, subset, etc.) |
| `src/ethos/configs/tokenization.yaml` | Tokenization pipeline config |
| `src/ethos/configs/dataset/mimic.yaml` | MIMIC-specific tokenization stages |

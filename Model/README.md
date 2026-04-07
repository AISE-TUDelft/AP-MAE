# Model

This directory contains the AP-MAE model package, the training entrypoint used for the paper, the run configurations, and a helper script for downloading the dataset cache used by the training code.

## What is included

- `ap_mae/`: reusable AP-MAE package with Hugging Face style `APMAE` and `APMAEConfig` classes
- `train_ap_mae.py`: training script used for the paper experiments
- `download_dataset.py`: one-time helper to cache the training dataset locally
- `config_ap_mae_tiny_codegpt.py`: smallest demonstration configuration
- `config_ap_mae_sc2_3b.py`, `config_ap_mae_sc2_7b.py`, `config_ap_mae_sc2_15b.py`: paper-scale StarCoder2 configurations
- `requirements.txt`: pinned Python dependencies for the Model workflow

## Recommended working directory

Run the commands for this part of the repository from inside `Model`.

That keeps the relative paths used by the checked-in configs consistent:

- dataset cache under `./huggingface`
- outputs under `./runs/<config-name>/`
- local imports from the `ap_mae` package

## Environment setup

The pinned `torch` and `torchvision` versions in `requirements.txt` target CUDA 12.1 wheels. A typical setup on Linux is:

```bash
cd Model
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

Notes:

- the supplied configs target CUDA devices by default
- the dataset cache is about 21 GB
- `use_wandb` and `use_huggingface_login` are disabled by default in `train_ap_mae.py`

## Fastest usable path in this repo

The self-contained part of this directory is the `ap_mae` package. If you want to instantiate the model, run a smoke test, or load a released checkpoint, start there.

Minimal local example:

```bash
cd Model
python - <<'PY'
import torch
from ap_mae import APMAE, APMAEConfig

config = APMAEConfig(max_length=64, patch_size=16, mask_ratio=0.5)
model = APMAE(config).eval()

attn = torch.rand(2, 1, config.max_length, config.max_length)
loss = model(attn)
print("reconstruction loss:", float(loss))
PY
```

See `ap_mae/README.md` for the expected tensor shapes, save/load examples, and the package-level API.

## Dataset download

Before training, populate the local Hugging Face cache used by the configs:

```bash
cd Model
python download_dataset.py
```

By default this downloads:

- dataset: `LaughingLogits/Stackless_Java_V2`
- split set: `Stackless_Java_V2`
- cache root: `./huggingface`

## Training configs

`train_ap_mae.py` imports a config module by name, not by filename. The valid config names are:

- `config_ap_mae_tiny_codegpt`
- `config_ap_mae_sc2_3b`
- `config_ap_mae_sc2_7b`
- `config_ap_mae_sc2_15b`

If `RUNCONFIG` is not set, the script falls back to `config_ap_mae_tiny_codegpt`.

Outputs are written under `./runs/<config-name>/`:

- `SavedModels/`
- `TrainImages/`
- `TestImages/`

Config overview:

- `config_ap_mae_tiny_codegpt`: smallest demo run, targets `microsoft/CodeGPT-small-java-adaptedGPT2`, uses `max_length=64`
- `config_ap_mae_sc2_3b`: paper-scale StarCoder2 3B run, uses `max_length=256`
- `config_ap_mae_sc2_7b`: paper-scale StarCoder2 7B run, uses `max_length=256`
- `config_ap_mae_sc2_15b`: paper-scale StarCoder2 15B run, uses `max_length=256`

The three StarCoder2 configs were intended for large-GPU or multi-GPU environments. The tiny config is the only checked-in configuration aimed at a lightweight local demonstration.

## Important note about the training script

This reproduction package includes the AP-MAE model code and the training entrypoint, but the training entrypoint is not fully self-contained in this checkout.

`train_ap_mae.py` imports:

- `DataUtil.Common`
- `DataUtil.Scalers`
- `DataUtil.DDPDataLoader.IterableAttentionDataset`

The repository does include shared utilities under `../Clustering/DataUtil`, but `DataUtil/DDPDataLoader.py` is not present in this repository. In practice this means:

- the `ap_mae` package is directly usable from this checkout
- the paper training script documents the intended workflow, but it cannot be run exactly as-is from this checkout alone
- if you have the missing `DataUtil` training loader from the original experiment environment, it needs to be available on `PYTHONPATH`

With the full training environment available, the intended local launch pattern is:

```bash
cd Model
PYTHONPATH=../Clustering RUNCONFIG=config_ap_mae_tiny_codegpt python train_ap_mae.py
```

On machines without the cluster-specific environment variables used in the paper setup, the script falls back to a single-process, single-GPU path and disables the distributed backend by default.

## Released checkpoints

The repository root links to the released AP-MAE model collection:

<https://huggingface.co/collections/LaughingLogits/ap-mae-models-66b27a73536bb1306d55c4c4>

Because `APMAE` subclasses `transformers.PreTrainedModel`, checkpoints saved with `save_pretrained()` can be reloaded with `APMAE.from_pretrained()`.

# ap_mae

`ap_mae` is the reusable AP-MAE package shipped with this repository.

It provides:

- `APMAEConfig`: configuration class built on `transformers.PretrainedConfig`
- `APMAE`: model class built on `transformers.PreTrainedModel`
- `APMAEEncoder`: encoder used to tokenize and mask attention matrices
- `APMAEDecoder`: decoder used to reconstruct the masked patches

## Importing the package

Run examples from inside the `Model` directory, or add `Model` to `PYTHONPATH`, so `from ap_mae import ...` resolves correctly.

## Expected input

AP-MAE treats an attention matrix as a single-channel image.

- input tensor shape: `[batch, 1, max_length, max_length]`
- `max_length` must match `APMAEConfig.max_length`
- `max_length` must be divisible by `patch_size`
- the extra `1` dimension is the single channel expected by the current encoder implementation

Examples:

- tiny demo config: `max_length=64`, `patch_size=16`
- paper-scale configs: `max_length=256`, `patch_size=32`

## Minimal example

```python
import torch
from ap_mae import APMAE, APMAEConfig

config = APMAEConfig(
    max_length=64,
    patch_size=16,
    mask_ratio=0.5,
)

model = APMAE(config).eval()
attn = torch.rand(2, 1, config.max_length, config.max_length)

# Standard forward pass: returns a reconstruction loss scalar.
loss = model(attn)
print(loss)

# Visualization mode: also returns the predicted and masked patch data.
loss, pred_patches, masked_indices, masked_patches, unmasked_indices = model(
    attn,
    visualizing=True,
)

# Encoder-only use: returns the encoded token sequence.
encoded_tokens = model.encoder.encode(attn)
print(encoded_tokens.shape)
```

Keep encoder and decoder on the same device unless you are deliberately working on model sharding. The checked-in configs use the same device for both components.

## Saving and loading

`APMAE` uses the usual Hugging Face save/load interface:

```python
from ap_mae import APMAE

model.save_pretrained("./runs/example_ap_mae")
reloaded = APMAE.from_pretrained("./runs/example_ap_mae")
```

The saved directory contains both the model weights and the serialized `APMAEConfig`.

## Implementation provenance

This implementation is based on:

- <https://github.com/lucidrains/vit-pytorch>
- encoder inspiration from `vit.py`
- decoder inspiration from `mae.py`
- Hugging Face ViT-MAE model/config structure:
  - <https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py>
  - <https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/configuration_vit_mae.py>

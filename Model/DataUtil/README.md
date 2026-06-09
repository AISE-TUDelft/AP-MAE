# Model Data Utilities

`DDPDataLoader.py` converts source-code samples into batches of attention-head
images for AP-MAE training. It supports one language through
`IterableAttentionDataset` and round-robin multilingual loading through
`IterableMultilingualAttentionDataset`.

## Dataset Setup

For a Hugging Face dataset, configure the repository and optional dataset
configuration name:

```python
config.dataset_location = "LaughingLogits/Stackless_Java_V2"
config.dataset_name = "Stackless_Java_V2"
config.lang = "java"
```

For datasets saved with `DatasetDict.save_to_disk`, use a shared local prefix.
The loader appends the normalized language name:

```text
/data/ap-mae-source-java
/data/ap-mae-source-cpp
```

```python
config.dataset_location = "/data/ap-mae-source"
config.languages = ["java", "c++"]  # c++ is normalized to cpp
```

Each dataset must contain the configured train/test splits and a `content`
column.

## Training Loader

`train_ap_mae.py` uses `build_attention_loader` to select the single-language
or multilingual implementation:

```python
loader = build_attention_loader(
    config=config,
    dataset_split=config.dataset_train_split,
    max_batches=10,
    batch_size=32,
    head_selection_strategy=("layerwise", 0.25),
    rank=0,
    world_size=1,
    tokenizer=tokenizer,
    target_model=target_model,
    gpu="cuda:0",
)

attentions, query = next(iter(loader))
print(attentions.shape)  # [32, 1, max_length, max_length]
```

Head selection accepts `"all"` or `("layerwise", quantity)`, where `quantity`
is a positive head count or a fraction in `(0, 1]`.

Tree-sitter queries support Java and C++ (`cpp` and `c++` are aliases).
Query names may use spaces or underscores, for example `numeric literals` and
`numeric_literals`.

Attention extraction requires enough accelerator memory for the target
language model with `output_attentions=True`. Tests and import checks do not
download models or datasets.

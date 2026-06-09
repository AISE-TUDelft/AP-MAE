# Clustering Data Utilities

`DataLoader.py` prepares query scenarios and target-model attentions for the
clustering pipeline. `AttentionData.py` stores raw patterns or AP-MAE
encodings in the HDF5 database.

## Scenario Iteration

Pass either a loaded Hugging Face dataset or a dataset repository ID:

```python
from datasets import load_dataset
from DataUtil.DataLoader import IterableScenarioAggregator

dataset = load_dataset(
    "LaughingLogits/Stackless_Java_V2",
    "Stackless_Java_V2",
    split="test",
)
scenarios = IterableScenarioAggregator(
    dataset,
    max_samples=10,
    max_length=256,
    queries=["identifiers", "numeric_literals"],
    lang="java",
    model="bigcode/starcoder2-3b",
    split="test",
)

tokenized_sample, query = next(iter(scenarios))
```

Tree-sitter queries are used for identifiers, literals, function calls, and
function names. Regex queries are used for brackets, punctuation, keywords,
and operators. Java and C++ are supported; `cpp` and `c++` are equivalent.

## Attention Loading

```python
from DataUtil.DataLoader import IterableAttentionLoader

attention_loader = IterableAttentionLoader(
    dataset,
    max_samples=10,
    max_length=256,
    queries=["identifiers", "boolean_literals"],
    lang="java",
    model="bigcode/starcoder2-3b",
    correct_only=False,
    target_model=target_model,
    target_model_device="device_map",
    split="test",
    evaluation=True,
)

attentions, query, correctness = next(iter(attention_loader))
print(attentions.shape)  # [layers, heads, 256, 256]
print(correctness)       # "correct" or "incorrect"
```

With `evaluation=False`, attentions are flattened to
`[layers * heads, 1, max_length, max_length]`. A `correct_only=True` loader
yields `(attentions, query)`; otherwise it also yields the correctness value.

`LanguageAggregator` combines one loader per language. For direct HDF5
storage, pass either one language-specific loader or a list of loaders:

```python
attention_data.generate_and_encode(attention_loader, encoding_model, 10)
attention_data.generate_patterns([java_loader, cpp_loader])
```

The target model must return eager attention tensors. Large paper-scale runs
require substantial GPU memory and storage; the examples above are intended
for small validation runs.

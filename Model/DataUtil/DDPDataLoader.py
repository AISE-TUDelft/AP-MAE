# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as tnnf
from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node

from .LanguageParser import getLanguage, getParser, normalize_language_name
from .TreeQuery import getQueryString, normalize_query_name


class IterableMultilingualAttentionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        config,
        dataset_split,
        max_batches,
        batch_size,
        head_selection_strategy,
        rank,
        world_size,
        tokenizer,
        target_model,
        gpu,
    ):
        self.languages = [
            normalize_language_name(language) for language in config.languages
        ]
        if not self.languages:
            raise ValueError("config.languages must contain at least one language")
        self.config = config
        self.dataset_split = dataset_split
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.head_selection_strategy = head_selection_strategy
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.gpu = gpu
        self.datasets = []
        for lang in self.languages:
            self.datasets.append(
                IterableAttentionDataset(
                    config=config,
                    dataset_location=config.dataset_location,
                    dataset_split=dataset_split,
                    max_batches=max_batches,
                    min_length=config.min_length,
                    max_length=config.max_length,
                    queries=config.queries,
                    lang=lang,
                    correct_only=config.correct_only,
                    target_model_name=config.target_model_name,
                    target_model_device=gpu,
                    tokenizer=tokenizer,
                    target_model=target_model,
                    num_proc=config.iter_loader_workers,
                    reset_after_iter=False,
                    equal_query_quantities=True,
                    rank=rank,
                    world_size=world_size,
                    batch_size=batch_size,
                    head_selection_strategy=head_selection_strategy,
                )
            )
        self.max_count = len(self.languages) * len(config.queries) * self.max_batches
        self.count = 0
        self.dataset_iterators = [iter(dataset) for dataset in self.datasets]

    def __iter__(self):
        modulus = len(self.dataset_iterators)
        while self.count < self.max_count:
            dataset_index = self.count % modulus
            item = next(self.dataset_iterators[dataset_index])
            self.count += 1
            yield item
        self.reset()

    def __len__(self):
        return self.max_count - self.count

    def reset(self):
        self.count = 0
        for dataset in self.datasets:
            dataset.reset()
        self.dataset_iterators = [iter(dataset) for dataset in self.datasets]


class IterableAttentionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        config,
        dataset_location,
        dataset_split,
        max_batches,
        min_length,
        max_length,
        queries,
        lang,
        correct_only,
        target_model_name,
        target_model_device,
        tokenizer,
        target_model,
        num_proc,
        reset_after_iter=True,
        equal_query_quantities=True,
        rank=0,
        world_size=1,
        batch_size=None,
        head_selection_strategy=("all"),
    ):
        self.config = config
        self.target_model_name = target_model_name
        self.target_model_device = target_model_device
        self.model_uses_device_map = (
            getattr(config, "target_model_device", None) == "device_map"
            or target_model_device == "device_map"
        )
        self.attention_device = (
            "cpu" if target_model_device == "device_map" else target_model_device
        )
        self.correct_only = correct_only
        self.dataset_location = dataset_location
        self.max_batches = max_batches
        self.min_length = min_length
        self.max_length = max_length
        self.queries = list(queries)
        if not self.queries:
            raise ValueError("queries must contain at least one query")
        self.lang = normalize_language_name(lang)
        self.count = 0
        self.num_proc = num_proc
        self.reset_after_iter = reset_after_iter
        self.equal_query_quantities = equal_query_quantities
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.head_selection_strategy = head_selection_strategy

        print(
            f"initial seed used to shuffle before dataset split: {config.dataset_split_seed}",
            flush=True,
        )
        dataset = self._load_dataset(dataset_location, dataset_split)
        self.hf_dataset = split_dataset_by_node(
            dataset.shuffle(seed=config.dataset_split_seed),
            rank=rank,
            world_size=world_size,
        )
        if len(self.hf_dataset) == 0:
            raise ValueError(
                f"dataset split {dataset_split!r} is empty for rank {rank}"
            )
        print(
            f"ddp dataloader created {world_size} splits, this process has loaded split {rank}.",
            flush=True,
        )
        self.dataset_iterator = iter(self.hf_dataset)

        self.target_model_num_layers = self.target_model.config.num_hidden_layers
        self.target_model_num_heads = self.target_model.config.num_attention_heads
        self.max_count = len(self.queries) * self.max_batches

        self.heads_per_layer = self._heads_per_layer(head_selection_strategy)
        self.selected_heads_per_sample = (
            self.heads_per_layer * self.target_model_num_layers
        )
        print(f"head_selection_strategy: {head_selection_strategy}")
        print(f"{self.selected_heads_per_sample} heads per sample")

        self.scenario_parsers = {}
        self.query_types = {}
        self.scenario_counts = {}
        self.query_accum = {}
        self.query_accum_count = {}
        self.query_accum_index = {}
        for query in self.queries:
            if normalize_query_name(query) != "random":
                query_str = getLanguage(self.lang).query(getQueryString(self.lang, query))
                self.scenario_parsers[query] = getParser(self.lang)
            else:
                query_str = query
            self.query_types[query] = query_str
            self.scenario_counts[query] = 0
            self.query_accum[query] = torch.zeros(
                self.batch_size + self.selected_heads_per_sample,
                1,
                self.max_length,
                self.max_length,
                device=self.attention_device,
            )
            self.query_accum_count[query] = 0
            self.query_accum_index[query] = 0

    def _is_local_dataset_location(self, dataset_location):
        location = str(dataset_location)
        return (
            location.startswith((".", "/", "~"))
            or Path(location).expanduser().exists()
            or Path(self._resolve_dataset_path(location)).exists()
        )

    def _resolve_dataset_path(self, dataset_location, lang=None):
        lang = normalize_language_name(lang or self.lang)
        location = os.path.expanduser(str(dataset_location))
        return f"{location}-{lang}"

    def _load_dataset(self, dataset_location, dataset_split):
        if self._is_local_dataset_location(dataset_location):
            dataset_path = self._resolve_dataset_path(dataset_location)
            return load_from_disk(dataset_path)[dataset_split]

        dataset_name = getattr(self.config, "dataset_name", None)
        return load_dataset(
            path=dataset_location,
            name=dataset_name,
            split=dataset_split,
            num_proc=self.num_proc,
        )

    def _heads_per_layer(self, strategy):
        if strategy == "all":
            return self.target_model_num_heads
        if not (
            isinstance(strategy, (tuple, list))
            and len(strategy) == 2
            and strategy[0] == "layerwise"
        ):
            raise ValueError(f"unsupported head selection strategy: {strategy!r}")

        quantity = strategy[1]
        if isinstance(quantity, bool):
            raise ValueError("layerwise head quantity must be an int or float")
        if isinstance(quantity, int):
            heads_per_layer = quantity
        elif isinstance(quantity, float):
            if not 0 < quantity <= 1:
                raise ValueError("layerwise head fraction must be in (0, 1]")
            heads_per_layer = int(quantity * self.target_model_num_heads)
        else:
            raise ValueError("layerwise head quantity must be an int or float")

        if not 1 <= heads_per_layer <= self.target_model_num_heads:
            raise ValueError(
                "layerwise head quantity must select between 1 and "
                f"{self.target_model_num_heads} heads per layer"
            )
        return heads_per_layer

    def select_heads(self, attentions):
        if self.head_selection_strategy == "all":
            return attentions.reshape(
                attentions.shape[0] * attentions.shape[1],
                1,
                self.max_length,
                self.max_length,
            )
        if self.head_selection_strategy[0] == "layerwise":
            attentions = attentions.reshape(
                attentions.shape[0] * attentions.shape[1],
                1,
                self.max_length,
                self.max_length,
            )
            q_per_layer = self.heads_per_layer
            sel = torch.IntTensor()
            for layer in range(self.target_model_num_layers):
                sel = torch.cat(
                    (
                        sel,
                        torch.randperm(self.target_model_num_heads)[:q_per_layer]
                        + layer * self.target_model_num_heads,
                    )
                )
            return attentions[sel]

    def __iter__(self):
        while self.count < self.max_count:
            for query in self.queries:
                if self.equal_query_quantities and self.scenario_counts[query] == self.max_batches:
                    continue
                while self.query_accum_count[query] < self.batch_size:
                    try:
                        sample_file = next(self.dataset_iterator)
                        sample_result = self.process(sample_file, query)
                        attentions = self.inference(sample_result)
                        attentions = self.select_heads(attentions)
                        self.query_accum[query][
                            self.query_accum_count[query] : self.query_accum_count[query]
                            + len(attentions)
                        ] = attentions
                        self.query_accum_count[query] += len(attentions)
                    except StopIteration:
                        self.dataset_iterator = iter(self.hf_dataset)
                        continue
                    except ValueError:
                        if self.equal_query_quantities:
                            continue
                        break
                if self.query_accum_count[query] < self.batch_size:
                    continue
                batch = self.query_accum[query][
                    self.query_accum_index[query] : self.query_accum_index[query]
                    + self.batch_size
                ].clone()
                self.scenario_counts[query] += 1
                self.count += 1
                self.query_accum_index[query] += self.batch_size
                self.query_accum_count[query] -= self.batch_size
                remainder = self.query_accum_count[query]
                if remainder < self.batch_size:
                    self.query_accum[query][0:remainder] = self.query_accum[query][
                        self.query_accum_index[query] : self.query_accum_index[query]
                        + remainder
                    ]
                    self.query_accum_count[query] = remainder
                    self.query_accum_index[query] = 0
                yield (
                    batch,
                    query,
                )
        if self.reset_after_iter:
            self.reset()
            return

    def __len__(self):
        return self.max_count - self.count

    def reset(self):
        self.count = 0
        for query in self.queries:
            self.scenario_counts[query] = 0
            self.query_accum_count[query] = 0
            self.query_accum_index[query] = 0
        data_shuffle_seed = random.randint(0, 100)
        print(
            f"seed used for dataset reset shuffle (only reshuffling the subsplit used by this process): {data_shuffle_seed}",
            flush=True,
        )
        self.hf_dataset = self.hf_dataset.shuffle(seed=data_shuffle_seed)
        self.dataset_iterator = iter(self.hf_dataset)

    def inference(self, sample_result):
        inputs = sample_result["input"]
        labels = sample_result["label"]
        if not self.model_uses_device_map:
            inputs = inputs.to(self.target_model_device)
        with torch.inference_mode():
            outputs = self.target_model(
                input_ids=inputs["input_ids"],
                output_hidden_states=False,
                output_attentions=True,
                use_cache=False,
            )
        attentions = outputs["attentions"]
        attentions = torch.cat(attentions)
        attentions = tnnf.pad(
            attentions,
            (
                0,
                self.max_length - attentions.shape[-1],
                0,
                self.max_length - attentions.shape[-2],
            ),
            "constant",
            0,
        )
        if not self.correct_only:
            return attentions
        batch_num = 0
        pred = outputs.logits[batch_num][-1].argmax().item()
        target = labels["input_ids"][batch_num][0].item()
        if pred == target:
            return attentions
        raise ValueError("inference was incorrect")

    def process(self, sample, query_name):
        if "starcoder" in self.target_model_name.lower():
            return self.gen_subsample_starcoder(
                self.tokenize(*self.prep_starcoder(sample["content"], query_name))
            )
        if "gpt" in self.target_model_name.lower():
            return self.gen_subsample_gpt(
                self.tokenize(*self.prep_gpt(sample["content"], query_name))
            )
        raise ValueError

    def prep_gpt(self, content, query_name):
        if normalize_query_name(query_name) == "random":
            tokens = self.tokenizer(content)["input_ids"]
            begin = random.randint(0, len(tokens) - 15)
            selection = tokens[0:begin]
            target = tokens[begin : begin + 5]
            content = self.tokenizer.decode(selection)
            target = self.tokenizer.decode(target)
            return content, target
        content = bytes(content, "UTF-8")
        tree = self.scenario_parsers[query_name].parse(content)
        captures = self.query_types[query_name].captures(tree.root_node)
        try:
            capture = random.sample(captures, 1)[0]
        except ValueError:
            raise ValueError("No matches detected in sample")
        start = capture[0].start_byte
        finish = capture[0].end_byte
        target = content[start:finish]
        content = content[:start]
        content = content.decode("UTF-8")
        target = target.decode("UTF-8")
        return content, target

    def gen_subsample_gpt(self, content):
        ids = content["input"]["input_ids"].flatten()
        mask = content["input"]["attention_mask"].flatten()
        max_len = ids.size()[0]
        if max_len < self.max_length:
            if self.min_length and max_len < self.min_length:
                raise ValueError("input is too small")
        else:
            ids = ids[-self.max_length :]
            mask = torch.ones(ids.size()).int()
        content["input"]["input_ids"] = ids.unsqueeze(dim=0)
        content["input"]["attention_mask"] = mask.unsqueeze(dim=0)
        return content

    def prep_starcoder(self, content, query_name):
        if normalize_query_name(query_name) == "random":
            tokens = self.tokenizer(content)["input_ids"]
            span_begin = random.randint(0, len(tokens) - 15)
            span_end = span_begin + random.randint(3, 10)
            prefix = tokens[0:span_begin]
            postfix = tokens[span_end:]
            target = tokens[span_begin:span_end]
            prefix = self.tokenizer.decode(prefix)
            postfix = self.tokenizer.decode(postfix)
            content = prefix + "<fim_suffix>" + postfix
            target = self.tokenizer.decode(target)
            return content, target
        content = bytes(content, "UTF-8")
        tree = self.scenario_parsers[query_name].parse(content)
        captures = self.query_types[query_name].captures(tree.root_node)
        try:
            capture = random.sample(captures, 1)[0]
        except ValueError:
            raise ValueError("No matches detected in sample")
        start = capture[0].start_byte
        finish = capture[0].end_byte
        target = content[start:finish]
        content = content[:start] + b"<fim_suffix>" + content[finish:]
        content = content.decode("UTF-8")
        target = target.decode("UTF-8")
        return content, target

    def gen_subsample_starcoder(self, sample):
        ids = sample["input"]["input_ids"].flatten()
        mask = sample["input"]["attention_mask"].flatten()
        max_len = ids.size()[0]
        start = torch.tensor([1])
        stop = torch.tensor([2])
        if max_len + 2 < self.max_length:
            if self.min_length and max_len + 2 < self.min_length:
                raise ValueError("input is too small")
            ids = torch.cat((start, ids, stop)).int()
            mask = torch.ones(ids.size()).int()
        else:
            fim_id = (ids == 3).nonzero().item()
            if fim_id <= self.max_length // 2:
                ids = torch.cat((start, ids))
                ids = ids[: self.max_length - 1]
                ids = torch.cat((ids, stop)).int()
                mask = torch.ones(ids.size()).int()
            else:
                right = ids[fim_id : fim_id + (self.max_length - 2) // 2]
                left = ids[fim_id - (self.max_length - 2 - len(right)) : fim_id]
                ids = torch.cat((start, left, right, stop)).int()
                mask = torch.ones(ids.size()).int()
        sample["input"]["input_ids"] = ids.unsqueeze(dim=0)
        sample["input"]["attention_mask"] = mask.unsqueeze(dim=0)
        return sample

    def tokenize(self, content, label):
        input = self.tokenizer(content, return_tensors="pt")
        label = self.tokenizer(label, return_tensors="pt")
        return {"input": input, "label": label}

import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class FakeIterableDataset:
    pass


def install_dependency_stubs():
    torch = types.ModuleType("torch")
    torch.utils = SimpleNamespace(
        data=SimpleNamespace(IterableDataset=FakeIterableDataset)
    )
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None
    datasets.load_from_disk = lambda *args, **kwargs: None
    datasets_distributed = types.ModuleType("datasets.distributed")
    datasets_distributed.split_dataset_by_node = (
        lambda dataset, rank, world_size: dataset
    )
    sys.modules["datasets"] = datasets
    sys.modules["datasets.distributed"] = datasets_distributed

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: None)
    sys.modules["transformers"] = transformers

    regex = types.ModuleType("regex")
    regex.compile = re.compile
    sys.modules["regex"] = regex

    tree_sitter_languages = types.ModuleType("tree_sitter_languages")
    tree_sitter_languages.get_language = lambda language: SimpleNamespace(
        query=lambda query: (language, query)
    )
    tree_sitter_languages.get_parser = lambda language: SimpleNamespace(
        set_language=lambda lang: None
    )
    sys.modules["tree_sitter_languages"] = tree_sitter_languages


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_package(package_name, directory, module_names):
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    loaded = {}
    for module_name in module_names:
        loaded[module_name] = load_module(
            f"{package_name}.{module_name}",
            directory / f"{module_name}.py",
        )
    return loaded


install_dependency_stubs()

MODEL_MODULES = load_package(
    "model_data_util",
    ROOT / "Model" / "DataUtil",
    ["LanguageParser", "TreeQuery", "DDPDataLoader"],
)

CLUSTERING_MODULES = load_package(
    "DataUtil",
    ROOT / "Clustering" / "DataUtil",
    ["LanguageParser", "TreeQuery", "DataLoader"],
)

fake_scalers = types.ModuleType("DataUtil.Scalers")
fake_scalers.log_normalize_scaler = lambda attentions, config: attentions
sys.modules["DataUtil.Scalers"] = fake_scalers
for module_name in ("numpy", "h5py", "matplotlib", "matplotlib.pyplot"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))
tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda iterable: iterable
sys.modules["tqdm"] = tqdm_module
ATTENTION_DATA = load_module(
    "DataUtil.AttentionData",
    ROOT / "Clustering" / "DataUtil" / "AttentionData.py",
)


class FakeDataset:
    def shuffle(self, seed=None):
        self.seed = seed
        return self

    def __iter__(self):
        return iter([{"content": "sample"}])

    def __len__(self):
        return 1


class LanguageAndQueryTests(unittest.TestCase):
    def test_language_aliases_are_consistent(self):
        for module in (
            MODEL_MODULES["LanguageParser"],
            CLUSTERING_MODULES["LanguageParser"],
        ):
            self.assertEqual(module.normalize_language_name("JAVA"), "java")
            self.assertEqual(module.normalize_language_name("C++"), "cpp")
            self.assertEqual(module.normalize_language_name("CPP"), "cpp")

    def test_query_names_accept_spaces_and_cpp_alias(self):
        for module in (
            MODEL_MODULES["TreeQuery"],
            CLUSTERING_MODULES["TreeQuery"],
        ):
            query = module.getQueryString("C++", "numeric literals")
            self.assertIn("number_literal", query)


class ModelLoaderTests(unittest.TestCase):
    def setUp(self):
        self.module = MODEL_MODULES["DDPDataLoader"]
        self.loader = self.module.IterableAttentionDataset.__new__(
            self.module.IterableAttentionDataset
        )
        self.loader.lang = "cpp"
        self.loader.config = SimpleNamespace(dataset_name="dataset-config")
        self.loader.num_proc = 3

    def test_dataset_path_and_source_resolution(self):
        self.assertEqual(
            self.loader._resolve_dataset_path("./source", "C++"),
            "./source-cpp",
        )

        remote_dataset = FakeDataset()
        with patch.object(
            self.module, "load_dataset", return_value=remote_dataset
        ) as load_dataset:
            result = self.loader._load_dataset("owner/repository", "train")
        self.assertIs(result, remote_dataset)
        load_dataset.assert_called_once_with(
            path="owner/repository",
            name="dataset-config",
            split="train",
            num_proc=3,
        )

        local_split = FakeDataset()
        with patch.object(
            self.module,
            "load_from_disk",
            return_value={"test": local_split},
        ) as load_from_disk:
            result = self.loader._load_dataset("./source", "test")
        self.assertIs(result, local_split)
        load_from_disk.assert_called_once_with("./source-cpp")

    def test_head_selection_validation(self):
        self.loader.target_model_num_heads = 8
        self.assertEqual(self.loader._heads_per_layer("all"), 8)
        self.assertEqual(self.loader._heads_per_layer(("layerwise", 2)), 2)
        self.assertEqual(self.loader._heads_per_layer(("layerwise", 0.25)), 2)
        for strategy in (
            ("layerwise", 0.0),
            ("layerwise", 9),
            ("unknown", 1),
        ):
            with self.assertRaises(ValueError):
                self.loader._heads_per_layer(strategy)

    def test_device_map_uses_concrete_attention_device(self):
        module = self.module
        config = SimpleNamespace(
            dataset_split_seed=42,
            target_model_device="device_map",
        )
        target_model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=2,
                num_attention_heads=4,
            )
        )
        with (
            patch.object(
                module.IterableAttentionDataset,
                "_load_dataset",
                return_value=FakeDataset(),
            ),
            patch.object(module.torch, "zeros", return_value=object(), create=True) as zeros,
        ):
            loader = module.IterableAttentionDataset(
                config=config,
                dataset_location="owner/repository",
                dataset_split="train",
                max_batches=1,
                min_length=16,
                max_length=32,
                queries=["random"],
                lang="java",
                correct_only=False,
                target_model_name="model",
                target_model_device="cuda:0",
                tokenizer=object(),
                target_model=target_model,
                num_proc=1,
                batch_size=2,
                head_selection_strategy="all",
            )
        self.assertTrue(loader.model_uses_device_map)
        self.assertEqual(loader.attention_device, "cuda:0")
        self.assertEqual(zeros.call_args.kwargs["device"], "cuda:0")

    def test_multilingual_round_robin_resets_for_next_epoch(self):
        module = self.module

        class FakeAttentionDataset:
            def __init__(self, lang, max_batches, queries, **kwargs):
                self.lang = lang
                self.max_count = max_batches * len(queries)
                self.reset()

            def __iter__(self):
                while self.count < self.max_count:
                    item = f"{self.lang}-{self.count}"
                    self.count += 1
                    yield item

            def reset(self):
                self.count = 0

        config = SimpleNamespace(
            languages=["java", "c++"],
            dataset_location="owner/repository",
            min_length=16,
            max_length=32,
            queries=["identifiers", "random"],
            correct_only=False,
            target_model_name="model",
            iter_loader_workers=1,
        )
        with patch.object(module, "IterableAttentionDataset", FakeAttentionDataset):
            loader = module.IterableMultilingualAttentionDataset(
                config=config,
                dataset_split="train",
                max_batches=1,
                batch_size=2,
                head_selection_strategy="all",
                rank=0,
                world_size=1,
                tokenizer=object(),
                target_model=object(),
                gpu="cpu",
            )
            expected = ["java-0", "cpp-0", "java-1", "cpp-1"]
            self.assertEqual(list(loader), expected)
            self.assertEqual(len(loader), 4)
            self.assertEqual(list(loader), expected)


class ClusteringAggregatorTests(unittest.TestCase):
    def setUp(self):
        self.module = CLUSTERING_MODULES["DataLoader"]

    def test_string_dataset_input_is_loaded(self):
        dataset = FakeDataset()
        with patch.object(
            self.module, "load_dataset", return_value=dataset
        ) as load_dataset:
            result = self.module._load_dataset_input("owner/repository", "test")
        self.assertIs(result, dataset)
        load_dataset.assert_called_once_with("owner/repository", split="test")

    def test_scenario_count_and_reset(self):
        module = self.module

        class FakeQueryLoader:
            def __init__(
                self, dataset, query, max_samples, max_length, lang, model
            ):
                self.query = query
                self.max_samples = max_samples

            def __iter__(self):
                for index in range(self.max_samples):
                    yield self.query, index

        with (
            patch.object(module, "IterableQueryLoader", FakeQueryLoader),
            patch.object(module.random, "choice", side_effect=lambda values: values[0]),
        ):
            aggregator = module.IterableScenarioAggregator(
                FakeDataset(),
                max_samples=2,
                max_length=32,
                queries=["identifiers", "random"],
                lang="C++",
                model="model",
                split="test",
            )
            iterator = iter(aggregator)
            self.assertEqual(next(iterator), ("identifiers", 0))
            self.assertEqual(len(aggregator), 3)
            self.assertEqual(len(list(iterator)), 3)
            self.assertEqual(len(aggregator), 4)
            self.assertEqual(len(list(aggregator)), 4)

    def test_language_aggregator_reset_does_not_duplicate_iterators(self):
        module = self.module

        class FakeAttentionLoader:
            def __init__(self, *args, **kwargs):
                self.lang = args[4]

            def __iter__(self):
                yield self.lang

            def reset(self):
                pass

        with (
            patch.object(module, "load_dataset", return_value=FakeDataset()),
            patch.object(module, "IterableAttentionLoader", FakeAttentionLoader),
            patch.object(module.random, "choice", side_effect=lambda values: values[0]),
        ):
            aggregator = module.LanguageAggregator(
                dataset_name="dataset",
                languages=["java", "c++"],
                max_samples=1,
                max_length=32,
                queries=["random"],
                model="model",
                correct_only=False,
                target_model=object(),
                target_model_device="cpu",
                split="test",
                evaluation=True,
                min_length=16,
            )
            self.assertEqual(list(aggregator), ["java", "cpp"])
            self.assertEqual(len(aggregator.iterators), 2)
            self.assertEqual(list(aggregator), ["java", "cpp"])
            self.assertEqual(len(aggregator.iterators), 2)


class AttentionDataCompatibilityTests(unittest.TestCase):
    def test_single_loader_and_loader_lists_are_supported(self):
        attention_data = ATTENTION_DATA.AttentionData.__new__(
            ATTENTION_DATA.AttentionData
        )
        loader = SimpleNamespace(lang="java")
        self.assertEqual(
            attention_data._coerce_attention_loaders(loader),
            [loader],
        )
        loaders = [loader, SimpleNamespace(lang="cpp")]
        self.assertIs(
            attention_data._coerce_attention_loaders(loaders),
            loaders,
        )


if __name__ == "__main__":
    unittest.main()

import json
from datasets import load_dataset, load_from_disk, Dataset
import hashlib
import os
import pandas as pd
from huggingface_hub import login

login("your_huggingface_token")


def sha256_checksum_text(content):
    """
    Computes and returns the SHA-256 checksum for the given text content.
    
    Args:
        content (dict): Dictionary with a "content" key containing the text.
    
    Returns:
        dict: Original file content with an added "sha" key for the checksum.
    """
    sha256 = hashlib.sha256()
    sha256.update(content["content"].encode("utf-8"))
    return {"content": content["content"], "sha": sha256.hexdigest()}


def process_file(file_path):
    """
    Processes a JSON file if its base name is a digit and computes its SHA-256 checksum.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: File path and checksum dictionary, or None if the file does not meet the criteria.

    """
    if not file_path.endswith(".json"):
        return file_path, None
    if os.path.splitext(os.path.basename(file_path))[0].isdigit():
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                return file_path, sha256_checksum_text(data)
        except json.JSONDecodeError as e:
            return file_path, None
    else:
        return file_path, None


def process_language():
    """
    Removes exact duplicate files within our custom dataset and between our custom dataset and Java-Stack v2.

    The new dataset without exact duplicates is pushed to the huggingface hub.

    """
    stackv2 = load_from_disk("your_path_to_Java_Stackv2")
    custom_dataset = load_dataset(
        "LaughingLogits/Stackless_Java_V2",
        "Raw_Java",
        split="train",
        cache_dir="your_cache_dir",
    )

    metadata = {}

    stackv2_hashes = stackv2.map(sha256_checksum_text, num_proc=64)
    custom_hashes = custom_dataset.map(sha256_checksum_text, num_proc=64)

    metadata["stackv2_size"] = len(stackv2_hashes)
    metadata["custom_size"] = len(custom_hashes)

    stackv2_hashes_set = set(stackv2_hashes["sha"])
    custom_hashes_set = set(custom_hashes["sha"])

    metadata["stackv2_oddp"] = len(stackv2_hashes_set)
    metadata["custom_oddp"] = len(custom_hashes_set)

    common_keys = stackv2_hashes_set.intersection(custom_hashes_set)
    custom_exact = custom_hashes.filter(lambda x: x["sha"] not in common_keys)

    metadata["custom_eddp"] = len(custom_exact)

    custom_exact_df = pd.DataFrame(custom_exact)
    custom_exact_fin = custom_exact_df.drop_duplicates(subset="sha")

    custom_exact_ds = Dataset.from_pandas(custom_exact_fin)
    metadata["custom_eddp"] = len(custom_exact_ds)

    with open("/your_path/metadata.json", 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    custom_exact_ds.push_to_hub(
        "your_dataset_path", "your_config_name", data_dir="your_data_dir_name"
    )


if __name__== "__main__":
    process_language()

from huggingface_hub import login
from datasets import load_dataset

login("your_huggingface_token")


def is_autogenerated(file_content):
    """
    Removes autogenerated files based on specific keywords found in the first 5 lines.

    Args:
        file_content (string): Text representing the file content.

    Returns:
        boolean: Returns True if the file contains any of the keywords, otherwise False.
    """
    autogenerated_patterns = [
        "generated by",
        "autogenerated",
        "auto-generated",
        "this file was generated",
        "this file is generated",
        "generated automatically",
        "automatically generated",
    ]

    lines = file_content.splitlines()

    for line in lines[:5]:
        line_lower = line.lower()
        for pattern in autogenerated_patterns:
            if pattern in line_lower:
                return True
    return False


dataset = load_dataset(
    "your_dataset_path",
    "your_config_name",
    split="train",
    cache_dir="your_cache_dir",
)

filtered_dataset = dataset.filter(lambda x: is_autogenerated(x["content"]) == False)

filtered_dataset.push_to_hub(
    "your_dataset_path", "your_config_name", data_dir="your_data_dir_name"
)

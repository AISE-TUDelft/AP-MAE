"""
This script converts the list of near-duplicate IDs from our dataset into 
a list of near-duplicate IDs from Java-Stack v2 with respect to each file in our dataset
"""
from huggingface_hub import login
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

login("your_huggingface_token")

ds = load_dataset(
    "your_dataset_path",
    "your_config_name",
    split="train",
    cache_dir="your_cache_dir",
)

near_stack = load_from_disk("your_path_to_Stackv2NearDup")

dict = {i: [] for i in range(len(ds))}

for idx, elem in tqdm(
    enumerate(near_stack["duplicates"]), total=len(near_stack["duplicates"])
):
    if len(elem) > 0:
        for our_id in elem:
            dict[our_id].append(idx)

values_array = list(dict.values())


near_ds = ds.add_column("near_dups_stkv2_idx", values_array)

near_ds.push_to_hub(
    "your_dataset_path", "your_config_name", data_dir="your_data_dir_name"
)

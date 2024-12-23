"""
We recommend downloading our dataset with this script, before training.

We have recently observed issues with the map function from datasets on first
downloads or generation of splits, when using built-in multiprocessing speedups.

This single threaded version should prepare the dataset without issues. 
We observed no such issues after the dataset has been downloaded / cached once.

The total size of this dataset should be ~21GB
"""

import os
# specifcy your preferred location for downloaded / cached files
# update the setting in the training config as well if you change this.
os.environ['HF_HOME'] = "./huggingface"

from datasets import load_dataset
dataset = load_dataset(
    path = 'LaughingLogits/Stackless_Java_V2', 
    name = 'Stackless_Java_V2', 
    split = None,
    # we recently saw issues with multiprocessing on initial loading.
    # num_proc = 8
    num_proc = None,
    # optional use specific cache rather than huggingface cache
    # cache_dir="./cache"
)
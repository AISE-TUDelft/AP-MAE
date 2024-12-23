import pickle
from datasketch import MinHash
from datasets import load_dataset
import hashlib
import struct
from huggingface_hub import login

login("your_huggingface_token")

"""Load your LSH object globally to avoid sharing it across processes"""
with open("your_path_to_lsh/lsh.pkl", "rb") as file:
    LSH = pickle.load(file)


def sha256_hash128(data):
    """Generate a 128-bit hash using SHA-256.

    Args:
        data (bytes): The data to generate a hash from.

    Returns:
        int: A 128-bit integer hash value.
    """
    hash_value = hashlib.sha256(data).digest()[:16]
    return struct.unpack("<QQ", hash_value)[0]


def minhash_data(doc):
    """Calculates the minhash for each file content using 7-shingles

    Args:
        doc (dict): Dictionary containing the content of the file

    Returns:
        dict: Returns dictionary with the minhash value
    """
    minhash = MinHash(num_perm=128, hashfunc=sha256_hash128)
    text = doc["content"].lower().replace(" ", "")
    shingles = set([text[i : i + 7] for i in range(len(text) - 7 + 1)])
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return {"minhash": minhash.digest()}


def jaccard_similarity(set1, set2):
    """Computes the Jaccard similarity between two sets.

    Args:
        set1 (set): The first set.
        set2 (set): The second set.

    Returns:
        float: The Jaccard similarity coefficient.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def near_dedup(elements):
    """Performs near-deduplication between Java-Stack v2 and LSH object containg the hashes of our custom dataset.

    Args:
        elements (dict): Batch of files with their content

    Returns:
        dict: Returns dictionary with the IDs of files from our dataset which are near-duplicates to files in Java-Stack v2.
    """
    results2 = []
    for idx, doc in enumerate(elements["content"]):
        minhash = MinHash(num_perm=128, hashfunc=sha256_hash128)
        text = doc.lower().replace(" ", "")
        shingles = set([text[i : i + 7] for i in range(len(text) - 7 + 1)])
        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))
        results = LSH.query(minhash)
        if len(results) > 0:
            results2.append(results)
        else:
            results2.append([])
    return {"duplicates": results2}


if __name__ == "__main__":

    stackv2 = load_dataset(
        "your_path_to_Stackv2",
        "java_config",
        split="train",
        cache_dir="your_cache_dir",
    )

    stackv2_m = stackv2.map(
        near_dedup,
        batched=True,
        num_proc=64,
        keep_in_memory=False,
        cache_file_name="your_cache_path/cache.arrow",
    )

    stackv2_m.save_to_disk("your_saving_path")

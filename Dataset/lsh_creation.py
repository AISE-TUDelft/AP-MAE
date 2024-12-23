import pickle
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset, disable_caching
import hashlib
import struct
from tqdm import tqdm
from huggingface_hub import login

# Disabling caching is optional
disable_caching()
login("your_huggingface_token")


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


if __name__ == "__main__":
    """
    Creates and saves on disk the LSH object containing all the minhashes of the files in our custom dataset.
    """

    ds = load_dataset(
        "your_dataset_path",
        "your_config_name",
        split="train",
        cache_dir="your_cache_dir",
    )

    lsh = MinHashLSH(threshold=0.7, num_perm=128, weights=(0.4, 0.6))

    ds2 = ds.map(minhash_data, num_proc=64)
    for idx, elem in tqdm(enumerate(ds2), total=len(ds2)):
        minhash = MinHash(
            num_perm=128, hashvalues=elem["minhash"], hashfunc=sha256_hash128
        )
        lsh.insert(idx, minhash, check_duplication=False)

    with open("your_lsh_path/lsh.pkl", "wb") as file:
        pickle.dump(lsh, file)

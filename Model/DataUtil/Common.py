"""
Common variables and functions.
"""
import json
import os


def write_file(
    contents,
    filename,
    create_dirs=True,
    mode="w",
    encoding="utf-8",
    ctype=None,
    indent=4,
):
    dirname = os.path.dirname(filename)
    if create_dirs and dirname != "":
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    file = open(file=filename, mode=mode, encoding=encoding)
    if ctype == "json":
        json.dump(obj=contents, fp=file, indent=indent)
    else:
        file.write(contents)
    file.close()


def read_file(filename, ctype=None, encoding="utf-8", strip=False):
    file = open(file=filename, mode="r", encoding=encoding)
    if ctype == "json":
        res = json.load(file)
    else:
        res = file.read()
    file.close()
    if strip:
        res = res.strip()
    return res


class IndexableDict(dict):
    __getattr__ = dict.__getitem__


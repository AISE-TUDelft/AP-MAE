"""
Common variables and functions.
"""
import json
import os

def write_file(
    contents,
    filename,
    create_dirs=True,
    mode='w',
    encoding="utf-8",
    ctype=None,
    indent=4,
):
    """
    Convenience writing to files function. Supports Json.
    ctype : str to identify type of content
        None (Default) : assumes basic string
    indent: integer or None
        None : no unnecessary whitespace, compact format for space efficiency.
        integer (Default=4): file is formatted for readability. value is number of spaces for indentation.
    """
    # ensure directory for file exists if not creating in current directory
    dirname = os.path.dirname(filename)
    if (create_dirs and dirname != ''):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # open+write creates files, dont need to check file exists
    # do need to check if should overwrite or append
    file = open(file=filename, mode=mode, encoding=encoding)
    if ctype == "json":
        json.dump(obj=contents, fp=file, indent=indent)
    else:
        file.write(contents)
    file.close()

def read_file(filename, ctype=None, encoding="utf-8", strip=False):
    """
    Convenience reading from files function. Supports Json.
    strip: whether to remove leading & trailing whitespace
    """
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
    """
    Wrapper for dictionaries to enable dict.attr and dict['attr'] indexing.
    Only used for mocking a static wandb format so we can re-use the template.

    warning: manually overriding attribute tables is generally not recommended.
    """
    __getattr__ = dict.__getitem__
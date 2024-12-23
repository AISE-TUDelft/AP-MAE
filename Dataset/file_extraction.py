import os
import json
import re
from datasets import Dataset
from huggingface_hub import login

login("your_huggingface_token")


def processFile(filename, basedir, extension, repo):
    """ Extracts metadata about a repository and its files with the specified extension. 

    Args:
        filename (str): the name of the code file
        basedir (str): the base directory of the code file from its GitHub repo
        extension (str): extension of the code file
        repo (dict): repository metadata

    Returns:
        dict: Repository and code file metadata
    """
    file_type = re.search(".*" + extension, filename)
    try:
        if (file_type and os.path.getsize(basedir + "/" + filename) < 50000000):
            data = {}
            try:
                with open(basedir + "/" + filename, "r", encoding = 'utf-8') as lang_file:
                    file_content = lang_file.read()
                    if len(file_content.split()) >= 10:
                        data["file_name"] = filename
                        data["file_path"] = basedir.replace("\\", "/") + "/" + filename
                        data["content"] = file_content
                        data["file_size"] = os.path.getsize(basedir + "/" + filename)
                        data["language"] = repo["language"]
                        data["extension"] = extension
                        data["repo_name"] = repo["full_name"]
                        data["repo_stars"] = repo["stargazers_count"]
                        data["repo_forks"] = repo["forks_count"]
                        data["repo_open_issues"] = repo["open_issues_count"]
                        data["repo_created_at"] = repo["created_at"]
                        data["repo_pushed_at"] = repo["pushed_at"]
                        return data
                    else:
                        return None
            except:
                return None
        else:
            return None
    except:
        return None


def fileExtraction():
    """ Extracts metadata from a list of repositories and their code files with the specified extension.

    Yields:
        dict: Yields a dictionary with repo and code file metadata for each entry in the list. 
    """
    language = "Java" # Hardcoded for Java, however, you can use it for any language
    lang_extensions = []
    folder = "/your_saving_location/" + language + "_unseen"
    repos = open(f"./{language}StrongCopyLeft10500.json", "r")
    content = repos.read()
    repos_data = json.loads(content)

    with open("./langs_extension.json", "r") as extensions:
        content = json.load(extensions)

    for lang in content:
        if lang["name"] == language:
            lang_extensions = lang["extensions"]

    for subdir in os.listdir(folder):
        subdir_data = next(
            (
                item
                for item in repos_data
                if item["full_name"].replace(".", "_").replace("/", "_") == subdir
            ),
            None,
        )
        if subdir_data is not None:
            for ext in lang_extensions:
                for current_subdir, sub_subdirs, sub_files in os.walk(
                    folder + "/" + subdir
                ):
                    for file in sub_files:
                        result = processFile(file, current_subdir, ext, subdir_data)
                        if result is not None:
                            yield result
                        else:
                            continue
        else:
            continue


if __name__ == "__main__":
    dataset = Dataset.from_generator(fileExtraction, cache_dir="/your_cache_dir/")
    dataset.push_to_hub("your_dataset_path", "your_config_name", data_dir="your_data_dir")

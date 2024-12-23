# -*- coding: utf-8 -*-
# DataUtil/LanguageParser.py
from tree_sitter import Language,  Parser
from tree_sitter_languages import get_language, get_parser
import os

def getParser(language):
    lang = get_language(language)
    parser = get_parser(language)
    parser.set_language(lang)
    return parser
    
def getLanguage(lang):
    return get_language(lang)
    try:
        return Language("./build/my-languages.so",  lang)
    except OSError:
        try:
            Language.build_library(
                "./build/my-languages.so",
                ["tree-sitter-java", "tree-sitter-python"],
                )
        except OSError:
            for repo_url in ['https://github.com/tree-sitter/tree-sitter-python',  'https://github.com/tree-sitter/tree-sitter-java']:
                os.system("git clone " + repo_url)
        finally:
            Language.build_library(
                "./build/my-languages.so",
                ["tree-sitter-java", "tree-sitter-python"],
                )
    except Exception as e:
        raise e
    finally:
        return Language("./build/my-languages.so",  lang)
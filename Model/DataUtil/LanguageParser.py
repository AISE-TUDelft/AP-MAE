# -*- coding: utf-8 -*-
from tree_sitter_languages import get_language, get_parser


def normalize_language_name(language):
    language = language.lower()
    if language == "c++":
        return "cpp"
    return language


def getParser(language):
    language = normalize_language_name(language)
    lang = get_language(language)
    parser = get_parser(language)
    parser.set_language(lang)
    return parser


def getLanguage(lang):
    return get_language(normalize_language_name(lang))

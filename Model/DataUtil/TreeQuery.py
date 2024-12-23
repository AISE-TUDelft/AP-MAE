# -*- coding: utf-8 -*-
# DataUtil/TreeQuery.py
import regex as re
import random
from .LanguageParser import getParser,  getLanguage

class Query():
    def __init__(self, query_name, query_string, lang, tokenizer):
        self.query_name = query_name
        self.query_string = query_string
        self.lang = lang
        self.tokenizer = tokenizer

    def get_span(self, content):
        raise Exception("Not implemented")

    def tokenize(self, content, label):
        input = self.tokenizer(content, return_tensors = 'pt')
        label = self.tokenizer(label, return_tensors = 'pt')

        return {"input": input, "label": label}


class RandomQuery(Query):
    def __init__(self, query_name, lang, tokenizer):
        super(RandomQuery).__init__()
        self.query_name = query_name
        self.lang = lang
        self.tokenizer = tokenizer


    def get_span(self, content):
        tokens = self.tokenizer(content)['input_ids']
        span_begin = random.randint(0, len(tokens)-15)
        span_end = span_begin + random.randint(3, 10)
        prefix = tokens[0:span_begin]
        postfix = tokens[span_end:]
        target = tokens[span_begin:span_end]
        
        prefix = self.tokenizer.decode(prefix)
        postfix = self.tokenizer.decode(postfix)

        context = prefix + "<fim_suffix>" + postfix
        target = self.tokenizer.decode(target)

        return self.tokenize(*(context, target))


class TreeSitterQuery(Query):
    def __init__(self, query_name, lang, tokenizer):
        super(TreeSitterQuery).__init__()
        self.lang = lang
        self.query = getLanguage(self.lang).query(getQueryString(self.lang, query_name))
        self.parser = getParser(self.lang)
        self.tokenizer = tokenizer

    def get_span(self, content):
        content = bytes(content, "UTF-8")
        tree = self.parser.parse(content)
        captures = self.query.captures(tree.root_node)

        try:
            capture = random.sample(captures, 1)[0]
        except ValueError:
            raise ValueError("No matches detected in sample")
        start = capture[0].start_byte
        finish = capture[0].end_byte
        
        target = content[start:finish]
        context = content[:start] + b"<fim_suffix>" + content[finish:]

        context = context.decode("UTF-8")
        target = target.decode("UTF-8")


        return self.tokenize(*(context, target))



class RegexQuery(Query):
    def __init__(self, query_name, lang, tokenizer):
        super(RegexQuery).__init__()
        self.tokenizer = tokenizer 
        self.query = getQueryString(lang, query_name)
        self.pattern = re.compile(self.query)


    def get_span(self, content):
        matches = len(self.pattern.findall(content))

        num = random.randint(0, matches-1)
        
        for m in self.pattern.finditer(content):
            if num == 0:
                break
            num-=1

        prefix = content[0:m.start()]
        postfix = content[m.end():]
        target = content[m.start():m.end()]

        context = prefix + "<fim_suffix>" + postfix

        return self.tokenize(*(context, target))



def getQuery(name, lang, tokenizer):
    treesitter = {"identifiers", "string_literals", "boolean_literals", "numeric_literals", "function_call", "function_name"}
    regex = {"closing_bracket", "stop", "eol", "keywords", "mathematical_operators", "boolean_operators", "assignment_operators"}
    if name in treesitter:
        return TreeSitterQuery(name, lang, tokenizer)
    elif name in regex:
        return RegexQuery(name, lang, tokenizer)
    elif name == "random":
        return RandomQuery("random", lang, tokenizer)
    else:
        raise ValueError("Query type not known " + name)


def getQueryString(lang, name):
    if name =='random':
        return ""
    if lang == 'java':
        return getJavaQuery(name)
    else:
        raise ValueError("Language not implemented")

def getJavaQuery(name):
    if name == 'identifiers':
        return """
                (identifier) @id
               """
    elif name == 'string_literals':
        return """
                (string_literal) @String_literal
                (character_literal) @String_literal
               """
    elif name =="boolean_literals":
        return """
               (true) @boolean
               (false) @boolean
               """
    elif name == "numeric_literals":
        return """
               (decimal_integer_literal) @number
               (decimal_floating_point_literal) @number
               (hex_integer_literal) @number
               (binary_integer_literal) @number

               """
    elif name == "function_call":
        return """
                (method_invocation
                    name: (identifier) @func_call
                )
               """
    elif name == "function_name":
        return """
                   (method_declaration
                       name: (identifier) @func_name
                   )

               """
    elif name == "closing_bracket":
        return "}|\)|]"
    elif name == "eol":
        return ";\n"
    elif name == "keywords":
        return "abstract|assert|break|case|catch|continue|default|do|else|enum|exports|extends|final|finally|for|if|implements|import|instanceof|interface|module|native|new|package|private|protected|public|requires|return|static|super|switch|synchronized|this|throws|throw|transient|try|void|volatile|while"
    elif name == "mathematical_operators":
        return "\+|-|\*|/|>|<|>=|<=|%|\+\+|--"
    elif name == "boolean_operators":
        return "!|&&|\|\|==|!="
    elif name =="assignment_operators":
        return "\+=|-=|\*=|/=|%=|&=|\|=|\^=|>>=|<<="
    elif name == "stop":
        return "\."
    else:
        raise ValueError("Query not implemented: " + str(name))

from nltk.tokenize.regexp import WordPunctTokenizer
from translator_constants.global_constant import *


class WordPunctTokenizerWrapper:

    def __init__(self, with_SOS=False):
        self.tokenizer = WordPunctTokenizer()
        self.with_SOS = with_SOS

    def __call__(self, text):
        token_list = self.tokenizer.tokenize(text)
        if self.with_SOS:
            token_list.insert(0, SOS_LABEL)
        token_list.append(EOS_LABEL)
        return token_list


def tokenizer_factory(factory_name):
    tokenizer = None
    if factory_name == WORD_PUNCT_TOKENIZER_WITH_SOS:
        tokenizer = WordPunctTokenizerWrapper(with_SOS=True)
    elif factory_name == WORD_PUNCT_TOKENIZER_WITHOUT_SOS:
        tokenizer = WordPunctTokenizerWrapper()
    return tokenizer

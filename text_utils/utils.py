import torch
import pandas as pd
import translator_constants.global_constant as glc
from tokenizer.word_punct_tokenizer import tokenizer_factory


def tokenize_corpus(corpus_df: pd.DataFrame, lang: str) -> pd.Series:
    if lang == glc.EN_LABEL:
        tokenizer = tokenizer_factory(glc.WORD_PUNCT_TOKENIZER_WITH_SOS)
    elif lang == glc.RU_LABEL:
        tokenizer = tokenizer_factory(glc.WORD_PUNCT_TOKENIZER_WITHOUT_SOS)
    else:
        raise ValueError(f"Unknown language {lang}")
    tokens = corpus_df.apply(lambda x: tokenizer(x[lang]), axis=1)
    return tokens


def add_padding(vector, size):
    if vector.shape[0] < size:
        pad_end_of_sentence = size - vector.shape[0]
        matrix_list = [vector[-1]] * pad_end_of_sentence
        paddin_tensor = torch.stack(matrix_list, dim=0)
        vector = torch.cat((vector, paddin_tensor), dim=0)
    elif vector.shape[0] > size:
        vector = vector[:size]
    return vector

import torch
from typing import List
from torch.utils.data import Dataset
import translator_constants.global_constant as glc


class RUENDataset(Dataset):

    def __init__(self, ru_data_list, en_data_list, ru_encoder, en_encoder, device,
                 ru_sentence_len=41, en_sentence_len=47):
        self.ru_data: List[List[int]] = ru_data_list
        self.en_data: List[List[int]] = en_data_list
        self.en_encoder = en_encoder
        self.ru_encoder = ru_encoder
        self.device = device
        self.ru_sentence_len = ru_sentence_len
        self.en_sentence_len = en_sentence_len

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, pos):
        en_vector = self._get_en_sentence(pos)
        ru_vector = self._get_ru_sentence(pos)

        return {glc.RU_DS_LABEL: ru_vector, glc.EN_DS_LABEL: en_vector}

    def _get_ru_sentence(self, pos):
        ru_sentence_list = self.ru_data[pos]
        vector = self.ru_encoder.transform(ru_sentence_list)
        vector = self.__add_padding(vector, self.ru_sentence_len)
        return vector

    def _get_en_sentence(self, pos):
        en_sentence_list = self.en_data[pos]
        vector = self.en_encoder.transform(en_sentence_list)
        # vector.shape (word_in_sentence, features)
        vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        vector = self.__add_padding(vector, self.en_sentence_len)
        return vector

    def __add_padding(self, vector, size):
        if vector.shape[0] < size:
            pad_end_of_sentence = size - vector.shape[0]
            matrix_list = [vector[-1]] * pad_end_of_sentence
            paddin_tensor = torch.stack(matrix_list, dim=0)
            vector = torch.cat((vector, paddin_tensor), dim=0)
        elif vector.shape[0] > size:
            vector = vector[:size]
        return vector

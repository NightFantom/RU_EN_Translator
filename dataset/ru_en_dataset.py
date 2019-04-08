import numpy as np
import torch
from torch.utils.data import Dataset
from translator_constants.global_constant import *


class RUENDataset(Dataset):

    def __init__(self, ru_data_list, en_data_list, ru_encoder, en_encoder, device,
                 ru_sentence_len=50, en_sentence_len=50):
        self.ru_data = ru_data_list
        self.en_data = en_data_list
        self.en_encoder = en_encoder
        self.ru_encoder = ru_encoder
        self.device = device
        self.ru_sentence_len = ru_sentence_len
        self.en_sentence_len = en_sentence_len

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, pos):
        en_vector = self._get_en_sentence(pos)
        en_vector = torch.tensor(en_vector, dtype=torch.float32, device=self.device)
        ru_vector = self._get_ru_sentence(pos)
        ru_vector = torch.tensor(ru_vector, dtype=torch.float32, device=self.device)

        return {RU_DS_LABEL: ru_vector, EN_DS_LABEL: en_vector}

    def _get_ru_sentence(self, pos):
        ru_sentence_list = self.ru_data[pos]
        ru_sentence_list = [ru_sentence_list]
        vector = self.ru_encoder.transform(ru_sentence_list)
        vector = vector[0]
        if vector.shape[0] < self.ru_sentence_len:
            pad_right = self.ru_sentence_len - vector.shape[0]
            vector = np.pad(vector, [(0, pad_right), (0, 0)], mode="constant", constant_values=0)
        elif vector.shape[0] > self.ru_sentence_len:
            vector = vector[:self.ru_sentence_len]
        return vector

    def _get_en_sentence(self, pos):
        en_sentence_list = self.en_data[pos]
        vector = self.en_encoder.transform(en_sentence_list)
        if vector.shape[0] < self.en_sentence_len:
            pad_right = self.en_sentence_len - vector.shape[0]
            vector = np.pad(vector, [(0, pad_right), (0, 0)], mode="constant", constant_values=0)
        elif vector.shape[0] > self.en_sentence_len:
            vector = vector[:self.en_sentence_len]
        return vector

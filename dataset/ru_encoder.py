from typing import List
import torch


class RUEncoderVoc:

    def __init__(self, vectorized_voc, device):
        self.vectorized_voc = vectorized_voc
        self.device = device
        self.dim_size = self.vectorized_voc.shape[1]

    def transform(self, sentence_list: List[int]) -> torch.Tensor:
        sentence_torch = torch.zeros(len(sentence_list), self.dim_size, dtype=torch.float32, device=self.device)
        for index, word_int in enumerate(sentence_list):
            vector_tensor = self.vectorized_voc[word_int]
            sentence_torch[index] = vector_tensor

        return sentence_torch

import json
from typing import List


class Vocabulary:

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.max_index = -1

    def fit(self, X: List[List[str]], Y=None):
        index = -1
        for sample in X:
            for feature in sample:
                if feature not in self.word_to_index:
                    index += 1
                    self.word_to_index[feature] = index

        self.__convert_index_to_word()

    def __convert_index_to_word(self):
        for word, index in self.word_to_index.items():
            self.index_to_word[index] = word
            if self.max_index < index:
                self.max_index = index

    def transform(self, X: List[List[str]]) -> List[List[int]]:
        sentense_list = []
        for sample in X:
            sentence = []
            sentense_list.append(sentence)
            for feature in sample:
                index = self.word_to_index[feature]
                sentence.append(index)
        return sentense_list

    def inverse_transform(self, X: List[List[int]]) -> List[List[str]]:
        sentense_list = []
        for sample in X:
            sentence = []
            sentense_list.append(sentence)
            for feature in sample:
                word = self.index_to_word[feature]
                sentence.append(word)
        return sentense_list

    def save(self, path):
        with open(path, mode="w") as file:
            json.dump(self.word_to_index, file, ensure_ascii=False)

    def load(self, path):
        with open(path, mode="r") as file:
            self.word_to_index = json.load(file)
        self.__convert_index_to_word()

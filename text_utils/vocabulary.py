class Vocabulary:

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.max_index = -1

    def fit(self, X, Y=None):
        for sample in X:
            for feature in sample:
                if feature not in self.word_to_index:
                    self.max_index += 1
                    self.word_to_index[feature] = self.max_index

        for word, index in self.word_to_index.items():
            self.index_to_word[index] = word

    def transform(self, X):
        sentense_list = []
        for sample in X:
            sentence = []
            sentense_list.append(sentence)
            for feature in sample:
                index = self.word_to_index[feature]
                sentence.append(index)
        return sentense_list

    def inverse_transform(self, X):
        sentense_list = []
        for sample in X:
            sentence = []
            sentense_list.append(sentence)
            for feature in sample:
                word = self.index_to_word[feature]
                sentence.append(word)
        return sentense_list
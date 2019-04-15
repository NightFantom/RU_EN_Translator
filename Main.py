# coding: utf-8

# # Local variable

# In[ ]:


BASE_PATH = "."

# # Imports

# In[ ]:


import os
import random
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook, tqdm

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch import nn

from gensim.models import FastText

from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from translator_constants.global_constant import *
from text_utils.vocabulary import Vocabulary
from text_utils.fast_text import FastTextWrapper
from tokenizer.word_punct_tokenizer import tokenizer_factory
from dataset.ru_en_dataset import RUENDataset

# In[ ]:


TENSORBOARD_LOG = os.path.join(BASE_PATH, "tensorboard_log")

if not os.path.exists(TENSORBOARD_LOG):
    os.makedirs(TENSORBOARD_LOG)

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chosen {device.type}:{device.index} device")


# In[ ]:


def del_tensorboard_log():
    import shutil
    shutil.rmtree(TENSORBOARD_LOG)


def load_data(word_dir: str) -> pd.DataFrame:
    path = os.path.join(word_dir, "data/corpus.en_ru.1m.en")
    data_en = load_corpus(path)

    path = os.path.join(word_dir, "data/corpus.en_ru.1m.ru")
    data_ru = load_corpus(path)

    df = pd.DataFrame({RU_LABEL: data_ru, EN_LABEL: data_en})
    return df


def load_corpus(path: str) -> list:
    with open(path, mode="r") as file:
        data = file.readlines()
    data = [s.strip().lower() for s in data]
    return data


def show_translation(ru_list, en_list):
    index = random.randint(0, len(ru_list) - 1)
    ru_sent = " ".join(ru_list[index])
    en_sent = " ".join(en_list[index])
    print(ru_sent)
    print(en_sent)


# # Load data

# In[ ]:


corpus_df = load_data(BASE_PATH)

# In[ ]:


corpus_df.head()

# In[ ]:


corpus_df = corpus_df.iloc[:10000]

# # Convert English tokens in one hot vectors

# In[ ]:


tokenizer = tokenizer_factory(WORD_PUNCT_TOKENIZER_WITH_SOS)

# In[ ]:


english_vocab = Vocabulary()
english_tokens = corpus_df.apply(lambda x: tokenizer(x[EN_LABEL]), axis=1)
english_vocab.fit(english_tokens)

# In[ ]:


en_encoder = LabelBinarizer(sparse_output=False)
en_encoder.fit(range(english_vocab.max_index + 1))

# In[ ]:


en_sentence_list = english_vocab.transform(english_tokens)

# In[ ]:



max_lenth = np.max([len(x) for x in en_sentence_list])
max_lenth += 1
print(f"Max sequence lenth is {max_lenth}")


# # Convert Russian tokens in vectors

# In[ ]:


path = os.path.join(BASE_PATH, "embeddings/skipgram_fasttext/araneum_none_fasttextskipgram_300_5_2018.model")
model = FastText.load(path)
ru_embedder = FastTextWrapper(model)
del path

# In[ ]:


ru_tokenizer = tokenizer_factory(WORD_PUNCT_TOKENIZER_WITHOUT_SOS)

# In[ ]:


russian_tokens = corpus_df.apply(lambda x: ru_tokenizer(x[RU_LABEL]), axis=1)

# # Wrap into Dataset

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(russian_tokens, en_sentence_list)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

# In[ ]:


train_data_set = RUENDataset(X_train, Y_train, ru_embedder, en_encoder, device=device)
train_dataloader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

# In[ ]:


test_data_set = RUENDataset(X_test, Y_test, ru_embedder, en_encoder, device=device)
test_dataloader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)

# In[ ]:


del corpus_df
del russian_tokens
del en_sentence_list


# # NN

# In[ ]:


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_vector_size):
        super().__init__()
        self.encoder = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_vector_size)

    def forward(self, X):
        """
        X.shape [batch_size, seq_len, fast_text_vect]
        return: output.shape[1,hidden_size]
                hidden_states - typle
        """
        output, hidden_states = self.encoder(X)

        return output, hidden_states


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_vector_size, vocabular_size):
        super().__init__()
        self.dence_in = nn.Linear(vocabular_size, input_size)
        self.decoder = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_vector_size)
        self.dence_out = nn.Linear(hidden_vector_size, vocabular_size)
        self.log_soft_max = nn.LogSoftmax(dim=-1)

    def forward(self, X, hidden_state):
        """
        Input: X.shape (batch_size, 1, vocabular_size)
        Return: X.shape (batch_size, 1, vocabular_size)
        """
        X = self.dence_in(X)
        X = torch.tanh(X)
        X, hidden_state = self.decoder(X, hidden_state)
        X = self.dence_out(X)
        #         X = torch.sigmoid(X)
        X = self.log_soft_max(X)
        return X, hidden_state


# In[ ]:


LOSS_VAL = "LossVal"
BLEU_SCORE = "BLEU"
BATCH_SIZE_INDEX = 0

class Trainer:
    def __init__(self,
                 log_writer,
                 encoder,
                 decoder,
                 encoder_optimizer,
                 decoder_optimizer,
                 loss,
                 input_size,
                 hidden_size,
                 EOS,
                 SOS,
                 epoch,
                 device,
                 verbose=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.loss = loss
        self.log_writer = log_writer
        self.EOS = EOS
        self.SOS = SOS
        self.epoch = epoch
        self.verbose = verbose
        self.device = device

    def train(self, dataloader, validation_dataloader):
        dataloader = self._wrap_dataloader(dataloader)

        for current_epoch in range(1, self.epoch + 1):
            if self.verbose:
                print(f"Epoch {current_epoch}")
            metric_dict = {LOSS_VAL: 0}
            for batch in dataloader:
                # ru_vector shape [1, seq_len_1, fast_text_vect]
                ru_vector = batch[RU_DS_LABEL]
                # eng_vector shape [1, seq_len_2, vocab_size]
                eng_vector = batch[EN_DS_LABEL]
                temp_metrics = self.process_one_pair(ru_vector, eng_vector)
                metric_dict[LOSS_VAL] += temp_metrics[LOSS_VAL]

            for key, val in metric_dict.items():
                self.log_writer.add_scalar(f"train/{key}", val, current_epoch)

            temp_metrics = self.validate(validation_dataloader)

            for key, val in temp_metrics.items():
                self.log_writer.add_scalar(f"validation/{key}", val,
                                           current_epoch)

    def process_one_pair(self, ru_vector, eng_vector):
        """
        ru_vector shape [batch_size, seq_len_1, fast_text_vect]
        eng_vector shape [batch_size, seq_len_2, vocab_size]
        """
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        temp_metrics = {}

        X, hidden_state = self.encoder(ru_vector)

        loss_torch = 0

        Y = self.SOS

        # Start from second token because we will compare prediction from token 'SOS' and next to him token
        for i in range(1, eng_vector.shape[1]):
            token = eng_vector[:, i]
            class_index = torch.argmax(token, dim=-1)

            Y, hidden_state = self.decoder(Y, hidden_state)

            # Check prediction of first tokens in batch
            Y = Y.view(BATCH_SIZE, -1)
            temp_loss = self.loss(Y, class_index)
            loss_torch += temp_loss

            # Y, word_index = self._get_pred_vect(Y)
            batch_size = eng_vector.shape[BATCH_SIZE_INDEX]
            # Teaching force mode. Using next token from true sequence
            Y = token.view(batch_size, 1, -1)

        temp_metrics[LOSS_VAL] = loss_torch.item()
        loss_torch.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return temp_metrics

    def validate(self, dataloader):
        bleu = 0
        loss_torch = 0
        with torch.no_grad():
            for batch in dataloader:

                # ru_vector shape [batch_size, seq_len_1, fast_text_vect]
                ru_vector = batch[RU_DS_LABEL]
                # eng_vector shape [batch_size, seq_len_2, vocab_size]
                eng_vector = batch[EN_DS_LABEL]

                X, hidden_state = self.encoder(ru_vector)
                Y = self.SOS

                sentence = []
                target_sentence_list = []
                for i in range(1, eng_vector.shape[1]):
                    token = eng_vector[:, i]
                    class_index = torch.argmax(token, dim=-1)

                    Y, hidden_state = self.decoder(Y, hidden_state)

                    # Check prediction of first tokens in batch
                    Y = Y.view(BATCH_SIZE, -1)
                    temp_loss = self.loss(Y, class_index)
                    loss_torch += temp_loss

                    Y, word_index = self._get_pred_vect(Y)

                    target_sentence_list.append(class_index[0].item())
                    sentence.append(word_index)
                    if Y is None:
                        break
                bleu += sentence_bleu([sentence], target_sentence_list)
            bleu = bleu / len(dataloader.dataset)

        return {LOSS_VAL: loss_torch.item(), BLEU_SCORE: bleu}

    def predict(self, dataloader):
        dataloader = self._wrap_dataloader(dataloader)

        with torch.no_grad():
            result = []
            for batch in dataloader:
                sentence = []
                result.append(sentence)

                ru_vector = batch[RU_DS_LABEL]

                X, hidden_state = self.encoder(ru_vector)
                Y = self.SOS
                for i in range(1, ru_vector.shape[1]):
                    Y, hidden_state = self.decoder(Y, hidden_state)

                    Y, word_index = self._get_pred_vect(Y)
                    sentence.append(word_index)
                    if Y is None:
                        break
            return result

    def _wrap_dataloader(self, dataloader):
        if self.verbose:
            dataloader = tqdm(dataloader)
        return dataloader

    def _get_pred_vect(self, Y):
        """
        Y.shape (batch_size,1,vocabular_size)
        """
        _, word_index = Y.topk(1)
        sequence_len = Y.size()[2]
        batch_size = Y.size()[BATCH_SIZE_INDEX]
        res = torch.zeros((batch_size, 1, sequence_len), device=self.device)
        for i in range(0, word_index.shape[0]):
            res[i, 0, word_index[i]] = 1
        return res, word_index


# In[ ]:


input_size = model.vector_size
vocabular_input_size = english_vocab.max_index + 1
hidden_size = 300

encoder = Encoder(input_size, hidden_size).to(device)
decoder = Decoder(hidden_size, hidden_size, vocabular_input_size).to(device)

encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)

loss_function = nn.NLLLoss()

# In[ ]:


EOS_vector = [[EOS_LABEL]]
EOS_vector = english_vocab.transform(EOS_vector)
EOS_vector = EOS_vector[0][0]

SOS_vector = [[SOS_LABEL]] * BATCH_SIZE
SOS_vector = english_vocab.transform(SOS_vector)
SOS_vector = en_encoder.transform(SOS_vector)
SOS_vector = torch.tensor(SOS_vector, dtype=torch.float32, device=device)
SOS_vector = SOS_vector.view(BATCH_SIZE, -1, SOS_vector.size()[1])

# In[ ]:


experiment_number = 0

# In[ ]:


experiment_number += 1
# log_path = os.path.join(TENSORBOARD_LOG, f"test_{experiment_number}")
log_path = os.path.join(TENSORBOARD_LOG, f"test_tanh")

log_writer = SummaryWriter(log_path)

trainer = Trainer(log_writer=log_writer,
                  encoder=encoder,
                  decoder=decoder,
                  encoder_optimizer=encoder_optimizer,
                  decoder_optimizer=decoder_optimizer,
                  loss=loss_function,
                  input_size=input_size,
                  hidden_size=hidden_size,
                  EOS=EOS_vector,
                  SOS=SOS_vector,
                  epoch=5,
                  device=device,
                  verbose=False)

# In[ ]:


try:
    trainer.train(train_dataloader, test_dataloader)
except Exception as e:
    traceback.print_tb(e)
finally:
    log_writer.close()

# In[ ]:


prediction = trainer.predict(test_dataloader)
predicted_sentences_list = english_vocab.inverse_transform(prediction)

# In[ ]:


show_translation(X_test, predicted_sentences_list)


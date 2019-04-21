import json
import os
import sys
import traceback
from typing import List

import torch
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from translator_constants.global_constant import *
from nn_models.decoder import Decoder
from nn_models.encoder import Encoder

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
                 verbose=False,
                 model_save_path=None,
                 english_vocab=None,
                 runtime_config_path=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.loss = loss
        self.log_writer = log_writer
        self.EOS = EOS
        self.SOS = SOS
        self.epoch = epoch
        self.verbose = verbose
        self.device = device
        self.model_save_path = model_save_path
        self.best_loss = sys.float_info.max
        self.english_vocab = english_vocab
        self.chencherry = SmoothingFunction()
        self.runtime_config_path = runtime_config_path
        self.runtime_config_dict = {}

    def train(self, dataloader, validation_dataloader):
        for current_epoch in range(1, self.epoch + 1):
            self.reload_config()
            epoch_dataloader = self._wrap_dataloader(dataloader)

            if self.verbose:
                print(f"Epoch {current_epoch}")
            metric_dict = {LOSS_VAL: 0}
            for batch in epoch_dataloader:
                # ru_vector shape [1, seq_len_1, fast_text_vect]
                ru_vector = batch[RU_DS_LABEL]
                # eng_vector shape [1, seq_len_2, vocab_size]
                eng_vector = batch[EN_DS_LABEL]
                temp_metrics = self.process_batch(ru_vector, eng_vector)
                metric_dict[LOSS_VAL] += temp_metrics[LOSS_VAL]

            for key, val in metric_dict.items():
                self.log_writer.add_scalar(f"train/{key}", val, current_epoch)

            # Because tqdm fix initial time when it creats we have to put it here
            epoch_validation_dataloader = self._wrap_dataloader(validation_dataloader)
            temp_metrics = self.validate(epoch_validation_dataloader)

            for key, val in temp_metrics.items():
                self.log_writer.add_scalar(f"validation/{key}", val,
                                           current_epoch)

            epoch_loss = temp_metrics[LOSS_VAL]
            if current_epoch > 1 and epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_model(current_epoch)

            if self.runtime_config_dict.get(STOP_COMMAND_KEY, False):
                if self.verbose:
                    print("Stoping traning")
                if self.best_loss != epoch_loss:
                    print(f"Save the last model: model_{current_epoch}")
                    self.save_model(current_epoch)
                break

    def process_batch(self, ru_vector, eng_vector):
        """
        ru_vector shape [batch_size, seq_len_1, fast_text_vect]
        eng_vector shape [batch_size, seq_len_2, vocab_size]
        """
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        temp_metrics = {}

        X, hidden_state = self.encoder(ru_vector)

        loss_torch = 0

        batch_size = X.shape[0]
        Y = self.get_SOS_vector(batch_size)

        # Start from second token because we will compare prediction from token 'SOS' and next to him token
        for token_pos in range(1, eng_vector.shape[1]):
            token = eng_vector[:, token_pos]
            class_index = torch.argmax(token, dim=-1)

            Y, hidden_state = self.decoder(Y, hidden_state)

            # Check prediction of first tokens in batch
            Y = Y.view(batch_size, -1)
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
        batch_amount = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_amount += 1
                # ru_vector shape [batch_size, seq_len_1, fast_text_vect]
                ru_vector = batch[RU_DS_LABEL]
                # eng_vector shape [batch_size, seq_len_2, vocab_size]
                eng_vector = batch[EN_DS_LABEL]

                X, hidden_state = self.encoder(ru_vector)
                batch_size = X.shape[0]
                Y = self.get_SOS_vector(batch_size)

                sentence = []
                target_sentence_list = []
                for i in range(1, eng_vector.shape[1]):
                    token = eng_vector[:, i]
                    class_index = torch.argmax(token, dim=-1)

                    Y, hidden_state = self.decoder(Y, hidden_state)

                    # Check prediction of first tokens in batch
                    Y = Y.view(batch_size, -1)
                    temp_loss = self.loss(Y, class_index)
                    loss_torch += temp_loss

                    Y, word_index = self._get_pred_vect(Y)

                    target_sentence_list.append(class_index)
                    sentence.append(word_index)

                bleu += self.calculate_bleu(sentence, target_sentence_list)
            bleu = bleu / batch_amount

        return {LOSS_VAL: loss_torch.item(), BLEU_SCORE: bleu}

    def predict(self, dataloader):
        dataloader = self._wrap_dataloader(dataloader)
        result = []
        with torch.no_grad():
            for batch in dataloader:
                sentence = []
                ru_vector = batch[RU_DS_LABEL]

                X, hidden_state = self.encoder(ru_vector)
                batch_size = X.shape[0]
                Y = self.get_SOS_vector(batch_size)
                for i in range(1, ru_vector.shape[1]):
                    Y, hidden_state = self.decoder(Y, hidden_state)
                    Y = Y.view(batch_size, -1)
                    Y, word_index = self._get_pred_vect(Y)
                    sentence.append(word_index)
                sentence = self.normilize(sentence)
                result.extend(sentence)
        result = torch.stack(result, dim=0)
        if torch.cuda.is_available():
            result = result.cpu()
        result = result.numpy()
        return result

    def _wrap_dataloader(self, dataloader):
        if self.verbose:
            dataloader = tqdm(dataloader)
        return dataloader

    def _get_pred_vect(self, Y):
        """
        :param Y - Y.shape (batch_size,vocabular_size)
        :return res.shape (batch_size, 1, vocabular_size)
                word_index.shape (batch_size, 1)
        """
        _, word_index = Y.topk(1)
        vocabular_size = Y.size()[1]
        batch_size = Y.size()[BATCH_SIZE_INDEX]
        res = torch.zeros((batch_size, 1, vocabular_size), device=self.device)
        for i in range(0, word_index.shape[0]):
            res[i, 0, word_index[i]] = 1
        word_index = word_index.view(-1)
        return res, word_index

    def calculate_bleu(self, predicted_sentence:List[torch.Tensor], target_sentence:List[torch.Tensor]) -> float:
        target_sentence_torch = self.normilize(target_sentence)
        target_sentence_list= self.normilize_translation(target_sentence_torch)
        predicted_sentence_torch = self.normilize(predicted_sentence)
        predicted_sentence_list = self.normilize_translation(predicted_sentence_torch)
        referenses = []
        for row in target_sentence_list:
            referenses.append([row])
        bleu = corpus_bleu(referenses, predicted_sentence_list, smoothing_function=self.chencherry.method1)
        return bleu

    def normilize(self, matrix_list):
        """
        Because of rows are position of word in sentence and columns are sentence we transpose matrix_list
        :param matrix_list: matrix, column - sentence, row - word in sentence
        :return: matrix where column - word in sentence, row - sentence
        """
        matrix_tensor = torch.stack(matrix_list, dim=0)
        matrix_tensor = torch.transpose(matrix_tensor, 0, 1)
        return matrix_tensor

    def save_model(self, epoch):
        directory = os.path.join(self.model_save_path, f"model_{epoch}")
        os.makedirs(directory)
        encoder_path = os.path.join(directory, "encoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)

        decoder_path = os.path.join(directory, "decoder.pt")
        torch.save(self.decoder.state_dict(), decoder_path)

    def get_SOS_vector(self, batch_size):
        vector = self.SOS
        if vector.shape[0] > batch_size:
            vector = vector[:batch_size]
        return vector

    def normilize_translation(self, sentences_torch: torch.Tensor) -> List[List[int]]:
        corpus_list = []
        for sentence_torch in sentences_torch:
            sentence_list = []
            corpus_list.append(sentence_list)
            for word_torch in sentence_torch:
                if len(sentence_list) == 0:
                    sentence_list.append(word_torch.item())
                elif sentence_list[-1] != word_torch:
                    sentence_list.append(word_torch.item())
        return corpus_list


    def reload_config(self):
        try:
            with open(self.runtime_config_path, mode="r") as file:
                new_config = json.load(file)
            self.runtime_config_dict = new_config
        except Exception as e:
            traceback.print_tb(e)

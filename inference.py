import os

import torch
from sklearn.preprocessing import LabelBinarizer

import translator_constants.global_constant as glc
from dataset.ru_encoder import RUEncoderVoc
from gpu_utils.gpu_utils import get_device
from nn_models.decoder import Decoder
from nn_models.decoder_with_attention import AttentionDecoder
from nn_models.encoder import Encoder
from nn_models.seq2seq_model import Trainer
from text_utils.utils import add_padding
from text_utils.vocabulary import Vocabulary
from tokenizer.word_punct_tokenizer import tokenizer_factory
from train import find_the_last_model, get_EOS, get_SOS

if __name__ == "__main__":

    device = get_device()
    print(f"Chosen {device.type}:{device.index} device")

    path = os.path.join(glc.BASE_PATH, "data/russian_meta/vectorized_vocabulary.trch")
    vocabulary_torch = torch.load(path)

    path = os.path.join(glc.BASE_PATH, "data/russian_meta/russian_vocab.json")
    russian_vocab = Vocabulary()
    russian_vocab.load(path)

    path = os.path.join(glc.BASE_PATH, "data/english_meta/english_vocab.json")
    english_vocab = Vocabulary()
    english_vocab.load(path)

    ru_encoder = RUEncoderVoc(vocabulary_torch, device)

    en_encoder = LabelBinarizer(sparse_output=False)
    en_encoder.fit(range(english_vocab.max_index + 1))

    input_size = glc.FAST_TEXT_VECTOR_SIZE
    vocabular_input_size = english_vocab.max_index + 1
    hidden_size = 300

    model_save_path = os.path.join(glc.BASE_PATH, "models")

    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = AttentionDecoder(hidden_size, hidden_size, vocabular_input_size, device).to(device)

    model_path = find_the_last_model(model_save_path)
    if model_path is not None:
        print(f"Using model {model_path}")
        model_dict = torch.load(model_path)
        encoder.load_state_dict(model_dict[glc.ENCODER_STATE_DICT])
        decoder.load_state_dict(model_dict[glc.DECODER_STATE_DICT])

        encoder.eval()
        decoder.eval()

        EOS_vector = get_EOS(english_vocab)
        SOS_vector = get_SOS(device, english_vocab, en_encoder)

        trainer = Trainer(log_writer=None,
                          encoder=encoder,
                          decoder=decoder,
                          encoder_optimizer=None,
                          decoder_optimizer=None,
                          loss=None,
                          input_size=input_size,
                          hidden_size=hidden_size,
                          EOS=EOS_vector,
                          SOS=SOS_vector,
                          epoch=None,
                          device=device,
                          verbose=True)

        ru_encoder = RUEncoderVoc(vocabulary_torch, device)
        tokenizer = tokenizer_factory(glc.WORD_PUNCT_TOKENIZER_WITHOUT_SOS)
        message = ""
        while message != ":q":
            message = input(">")
            russian_tokens = tokenizer(message)
            ru_sentence_list = russian_vocab.transform([russian_tokens])[0]
            vector = ru_encoder.transform(ru_sentence_list)
            vector = add_padding(vector, glc.MAX_RUSSIAN_SEQUENCE_LEN)
            vector = vector.view(1, vector.shape[0], vector.shape[1])
            prediction = trainer.predict_batch(vector)
            prediction = english_vocab.inverse_transform(prediction)[0]
            prediction = " ".join(prediction)
            print(prediction)

import glob
import os
import pickle
import traceback

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import translator_constants.global_constant as glc
from dataset.ru_en_dataset import RUENDataset
from dataset.ru_encoder import RUEncoderVoc
from gpu_utils.gpu_utils import get_device
from nn_models.decoder import Decoder
from nn_models.decoder_with_attention import AttentionDecoder
from nn_models.encoder import Encoder
from nn_models.seq2seq_model import Trainer
from text_utils.vocabulary import Vocabulary

TENSORBOARD_LOG = os.path.join(glc.BASE_PATH, "tensorboard_log")


def get_SOS(device, english_vocab, en_encoder):
    SOS_vector = [[glc.SOS_LABEL]] * glc.BATCH_SIZE
    SOS_vector = english_vocab.transform(SOS_vector)
    SOS_vector = en_encoder.transform(SOS_vector)
    SOS_vector = torch.tensor(SOS_vector, dtype=torch.float32, device=device)
    SOS_vector = SOS_vector.view(glc.BATCH_SIZE, -1, SOS_vector.size()[1])
    return SOS_vector


def get_EOS(english_vocab):
    EOS_vector = [[glc.EOS_LABEL]]
    EOS_vector = english_vocab.transform(EOS_vector)
    EOS_vector = EOS_vector[0][0]
    return EOS_vector


def shrink(seq: list):
    return seq[:glc.AMOUNT_OF_SAMPLES]


def get_experiment_folder_name():
    pattern = os.path.join(TENSORBOARD_LOG, f"test_*")
    max_index = 0
    folder_list = glob.glob(pattern)
    if len(folder_list) > 0:
        for folder_name in folder_list:
            folder_name = os.path.basename(folder_name)
            index = int(folder_name.split("_")[-1])
            if index > max_index:
                max_index = index

    experiment_number = max_index + 1
    log_path = os.path.join(TENSORBOARD_LOG, f"test_{experiment_number}")
    return log_path


def find_the_last_model(path):
    path = os.path.join(path, "*.pt")
    model_list = glob.glob(path)
    max_epoch = -1
    the_last_model_path = None
    for model_path in model_list:
        model = os.path.basename(model_path)
        name, _ = os.path.splitext(model)
        epoch = int(name.split("_")[-1])
        if max_epoch < epoch:
            max_epoch = epoch
            the_last_model_path = model_path
    return the_last_model_path


def main():

    if not os.path.exists(TENSORBOARD_LOG):
        os.makedirs(TENSORBOARD_LOG)

    device = get_device()
    print(f"Chosen {device.type}:{device.index} device")

    path = os.path.join(glc.BASE_PATH, "data/russian_meta/russian_token.pkl")
    with open(path, mode="rb") as file:
        ru_sentence_list = pickle.load(file)
    ru_sentence_list = shrink(ru_sentence_list)

    path = os.path.join(glc.BASE_PATH, "data/russian_meta/vectorized_vocabulary.trch")
    vocabulary_torch = torch.load(path)

    path = os.path.join(glc.BASE_PATH, "data/english_meta/english_token.pkl")
    with open(path, mode="rb") as file:
        en_sentence_list = pickle.load(file)
    en_sentence_list = shrink(en_sentence_list)

    path = os.path.join(glc.BASE_PATH, "data/english_meta/english_vocab.json")
    english_vocab = Vocabulary()
    english_vocab.load(path)

    ru_encoder = RUEncoderVoc(vocabulary_torch, device)

    en_encoder = LabelBinarizer(sparse_output=False)
    en_encoder.fit(range(english_vocab.max_index + 1))

    X_train, X_test, Y_train, Y_test = train_test_split(ru_sentence_list, en_sentence_list, random_state=42)

    train_data_set = RUENDataset(X_train, Y_train, ru_encoder, en_encoder, device=device)
    train_dataloader = DataLoader(train_data_set, batch_size=glc.BATCH_SIZE, shuffle=True)

    test_data_set = RUENDataset(X_test, Y_test, ru_encoder, en_encoder, device=device)
    test_dataloader = DataLoader(test_data_set, batch_size=glc.BATCH_SIZE, shuffle=False)

    input_size = glc.FAST_TEXT_VECTOR_SIZE
    vocabular_input_size = english_vocab.max_index + 1
    hidden_size = 300

    model_save_path = os.path.join(glc.BASE_PATH, "models")

    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = Decoder(hidden_size, hidden_size, vocabular_input_size, device).to(device)

    encoder_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {encoder_total_params}")
    print(f"Decoder parameters: {decoder_total_params}")
    print(f"Total parameters: {encoder_total_params + decoder_total_params}")

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)

    start_epoch = 1
    best_loss = None
    model_path = find_the_last_model(model_save_path)
    if glc.CONTINUE_LEARNING and model_path is not None:

        model_dict = torch.load(model_path)
        encoder.load_state_dict(model_dict[glc.ENCODER_STATE_DICT])
        decoder.load_state_dict(model_dict[glc.DECODER_STATE_DICT])

        encoder_optimizer.load_state_dict(model_dict[glc.ENCODER_OPTIMIZER_STATE_DICT])
        decoder_optimizer.load_state_dict(model_dict[glc.DECODER_OPTIMIZER_STATE_DICT])
        start_epoch = model_dict[glc.EPOCH] + 1
        best_loss = model_dict.get(glc.THE_LOWEST_LOSS)
        print(f"Continue learning from epoch {start_epoch}")
    else:
        print("Start learning from the beginning")

    loss_function = torch.nn.NLLLoss()

    EOS_vector = get_EOS(english_vocab)
    SOS_vector = get_SOS(device, english_vocab, en_encoder)

    log_path = get_experiment_folder_name()
    log_writer = SummaryWriter(log_path)

    runtime_config_path = os.path.join(glc.BASE_PATH, "runtime_config.json")

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
                      epoch=100,
                      device=device,
                      verbose=True,
                      model_save_path=model_save_path,
                      english_vocab=english_vocab,
                      runtime_config_path=runtime_config_path,
                      start_epoch=start_epoch,
                      best_loss=best_loss)

    try:
        trainer.train(train_dataloader, test_dataloader)
    except Exception as e:
        traceback.print_exc()
    finally:
        log_writer.close()


if __name__ == "__main__":
    main()

import json
import string
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_wrapper import DataWrapper

ERC_DATASET_PATHS = {
    "train": "data/erc/MaSaC_train_erc.json",
    "val": "data/erc/MaSaC_val_erc.json",
    "test": "data/erc/MaSaC_test_erc.json"
}

EMOTIONS_ERC = {
    "disgust": 0,
    "neutral": 1,
    "contempt": 2,
    "anger": 3,
    "sadness": 4,
    "joy": 5,
    "fear": 6,
    "surprise": 7
}


def get_dataloaders(args):
    train_dataset_path, val_dataset_path, test_dataset_path = _get_dataset_paths(args.experiment_name)
    train_dataloader = _create_dataloader(train_dataset_path, args.batch_size, args.shuffle)
    val_dataloader = _create_dataloader(val_dataset_path, args.batch_size, args.shuffle)
    test_dataloader = _create_dataloader(test_dataset_path, args.batch_size, args.shuffle)
    return train_dataloader, val_dataloader, test_dataloader


def _tokenize(text):
    return text.split()


def _get_dataset_paths(experiment_name):
    if experiment_name == "erc":
        return ERC_DATASET_PATHS["train"], ERC_DATASET_PATHS["val"], ERC_DATASET_PATHS["test"]


def _create_dataloader(test_dataset_path, batch_size, shuffle):
    with open(test_dataset_path) as f:
        dataset = json.load(f)

    # Extract all utterances and emotions
    utterance_list = []
    emotions_list = []
    for dialogue in dataset:
        utterance_list = utterance_list + dialogue["utterances"]
        emotions_list = emotions_list + dialogue["emotions"]

    # Tokenize utterances and remove punctuation
    tokenized_utterances = [_tokenize(utt.lower()) for utt in utterance_list]
    tokenized_utterances = [[word.strip(string.punctuation) for word in sublist] for sublist in tokenized_utterances]

    # Encode emotions
    encoded_emotions = [EMOTIONS_ERC[emotion] for emotion in emotions_list]

    # Build the vocabulary
    vocab = Counter(word for utterance in tokenized_utterances for word in utterance)

    # Assign an index to each word in the vocabulary starting from 2
    # We reserve '0' for padding and '1' for unknown words
    word_to_index = {word: i + 2 for i, (word, _) in enumerate(vocab.items())}
    vocab_size = len(word_to_index) + 2  # Including padding and unknown word tokens

    # Convert the tokenized utterances to integer sequences
    sequences = [[word_to_index.get(word, 1) for word in utterance] for utterance in tokenized_utterances]

    # Pad the sequences to have equal length
    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)

    # Concatenate utterances with emotions
    assert len(padded_sequences) == len(encoded_emotions)

    wrapped_dataset = DataWrapper(data=padded_sequences, labels=encoded_emotions)
    dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

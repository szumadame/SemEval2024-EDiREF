import json
import string
from collections import Counter
from googletrans import Translator

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, AutoTokenizer

from data_wrapper import DialogueDatasetWrapperForBERT, DialogueDatasetWrapperForLSTM

ERC_DATASET_PATHS = {
    "train": "data/erc/MaSaC_train_erc.json",
    "val": "data/erc/MaSaC_val_erc.json",
    "test": "data/erc/MaSaC_test_erc.json"
}

EFR_DATASET_PATHS = {
    "train": "data/efr_hin_eng/MaSaC_train_efr.json",
    "val": "data/efr_hin_eng/MaSaC_val_efr.json",
    "test": "data/efr_hin_eng/MaSaC_test_efr.json"
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
    train_dataset_path, val_dataset_path, _ = _get_dataset_paths(args.experiment_name)
    if args.model == "lstm":
        return _create_dataloader_lstm(train_dataset_path, val_dataset_path, args.batch_size)
    elif args.model == "bert":
        return _create_dataloader_bert(train_dataset_path, val_dataset_path, args.batch_size)


def _get_dataset_paths(experiment_name):
    if experiment_name == "erc":
        return ERC_DATASET_PATHS["train"], ERC_DATASET_PATHS["val"], ERC_DATASET_PATHS["test"]


def _tokenize(text):
    return text.split()


def _extract_relevant_data(dataset):
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

    return encoded_emotions, tokenized_utterances


def _create_dataloader_lstm(train_dataset_path, test_dataset_path, batch_size):
    with open(train_dataset_path) as f:
        train_dataset = json.load(f)

    with open(test_dataset_path) as f:
        test_dataset = json.load(f)

    train_encoded_emotions, train_tokenized_utterances = _extract_relevant_data(train_dataset)
    test_encoded_emotions, test_tokenized_utterances = _extract_relevant_data(test_dataset)

    #train_utterences_augmentated = augmentation(train_tokenized_utterances, train_encoded_emotions)

    translated_lists, selected_labels = augmentation(train_tokenized_utterances, train_encoded_emotions)

    train_tokenized_utterances = train_tokenized_utterances + translated_lists
    train_encoded_emotions = train_encoded_emotions + selected_labels

    tokenizer = AutoTokenizer.from_pretrained("obaidtambo/hinglish_bert_tokenizer")

    train_utterances_as_strings = [" ".join(sublist) for sublist in train_tokenized_utterances]
    test_utterances_as_strings = [" ".join(sublist) for sublist in test_tokenized_utterances]

    # Tokenize the utterances using the BERT tokenizer
    train_sequences = [tokenizer.encode(utterance, add_special_tokens=True) for utterance in
                       train_utterances_as_strings]
    test_sequences = [tokenizer.encode(utterance, add_special_tokens=True) for utterance in test_utterances_as_strings]

    # Pad the sequences to have equal length
    train_padded_sequences = pad_sequence([torch.tensor(seq) for seq in train_sequences], batch_first=True,
                                          padding_value=tokenizer.pad_token_id)
    test_padded_sequences = pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True,
                                         padding_value=tokenizer.pad_token_id)

    # # Build the vocabulary based on the training dataset
    # vocab = Counter(word for utterance in train_tokenized_utterances for word in utterance)
    #
    # # Assign an index to each word in the vocabulary starting from 2
    # # We reserve '0' for padding and '1' for unknown words
    # word_to_index = {word: i + 2 for i, (word, _) in enumerate(vocab.items())}
    # vocab_size = len(word_to_index) + 2  # Including padding and unknown word tokens
    #
    # # Convert the tokenized utterances to integer sequences
    # train_sequences = [[word_to_index.get(word, 1) for word in utterance] for utterance in train_tokenized_utterances]
    # test_sequences = [[word_to_index.get(word, 1) for word in utterance] for utterance in test_tokenized_utterances]
    #
    # # Pad the sequences to have equal length
    # train_padded_sequences = pad_sequence([torch.tensor(seq) for seq in train_sequences], batch_first=True,
    #                                       padding_value=0)
    # test_padded_sequences = pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True,
    #                                      padding_value=0)

    assert len(train_padded_sequences) == len(train_encoded_emotions)
    assert len(test_padded_sequences) == len(test_encoded_emotions)

    vocab_size = tokenizer.vocab_size

    # Concatenate utterances with emotions
    train_wrapped_dataset = DialogueDatasetWrapperForLSTM(data=train_padded_sequences, labels=train_encoded_emotions,
                                                          vocab_size=vocab_size)
    test_wrapped_dataset = DialogueDatasetWrapperForLSTM(data=test_padded_sequences, labels=test_encoded_emotions,
                                                         vocab_size=vocab_size)

    # Create weighted random sampler to counteract the imbalanced dataset
    weighted_random_sampler = WeightedRandomSampler(weights=train_wrapped_dataset.class_weights,
                                                    num_samples=len(train_wrapped_dataset))

    # train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, sampler=weighted_random_sampler)
    test_dataloader = DataLoader(test_wrapped_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def _extract_relevant_data_bert(dataset):
    # Extract all utterances and emotions
    utterance_list = []
    emotions_list = []
    for dialogue in dataset:
        utterance_list = utterance_list + dialogue["utterances"]
        emotions_list = emotions_list + dialogue["emotions"]

    # Encode emotions
    encoded_emotions_list = [EMOTIONS_ERC[emotion] for emotion in emotions_list]
    return encoded_emotions_list, utterance_list


def _create_dataloader_bert(train_dataset_path, test_dataset_path, batch_size):
    with open(train_dataset_path) as f:
        train_dataset = json.load(f)

    with open(test_dataset_path) as f:
        test_dataset = json.load(f)

    train_encoded_emotions, train_utterances = _extract_relevant_data_bert(train_dataset)
    test_encoded_emotions, test_utterances = _extract_relevant_data_bert(test_dataset)

    # Get pretrained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Concatenate utterances with emotions
    train_wrapped_dataset = DialogueDatasetWrapperForBERT(data=train_utterances, labels=train_encoded_emotions,
                                                          vocab_size=None, max_length=None, tokenizer=tokenizer)
    test_wrapped_dataset = DialogueDatasetWrapperForBERT(data=test_utterances, labels=test_encoded_emotions,
                                                         vocab_size=None, max_length=None, tokenizer=tokenizer)

    # Create weighted random sampler to counteract the imbalanced dataset
    # weighted_random_sampler = WeightedRandomSampler(weights=train_wrapped_dataset.class_weights,
    #                                                 num_samples=len(train_wrapped_dataset))

    # train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, sampler=weighted_random_sampler)
    train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_wrapped_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def augmentation(utterances, labels):
    # Pick part of dataset that will be augmentated
    num_to_pick = len(utterances) * 50 // 100
    random.seed(42)

    # Shuffle indices
    indices = list(range(len(utterances)))
    random.shuffle(indices)
    selected_indices = indices[:num_to_pick]

    selected_utterances = [utterances[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]
    print(len(utterances))
    random_deletion_lists = []
    #translated_lists = []
    for sublist in selected_utterances:
        #translated_sublist = [translate_word(word) for word in sublist]
        #translated_lists.append(translated_sublist)
        if random.random() < 0.5:
            modified_sublist = random_deletion(sublist)
        else:
            modified_sublist = random_swap(sublist)

        random_deletion_lists.append(modified_sublist)

    return random_deletion_lists, selected_labels


def translate_word(word, src='hi', dest='en'):
    try:
        translator = Translator()
        if translator.detect(word).lang == 'en':
            # Translate to English
            translation = translator.translate(word, src='en', dest='hi')
            return translation.text
        else:
            # If the string is not in Hindi, keep it as is
            return word
    except TypeError as e:
        # Handle NoneType error
        print("Error:", e, word)
        return word


def random_deletion(words_in_list):
    n = len(words_in_list)

    if n == 1:  # return if single word
        return words_in_list

    filtered_list = [word for word in words_in_list if random.random() <= 0.9]
    return filtered_list


def random_swap(words_in_list, n_swaps=1):
    words = words_in_list[:]
    length = len(words)

    if length == 1:  # return if single word
        return words_in_list

    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

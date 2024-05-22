import json

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from data_wrapper import DialogueDatasetWrapper

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
    train_dataset_path, val_dataset_path, _ = _get_dataset_paths(args.experiment_name)
    return _create_dataloaders(train_dataset_path, val_dataset_path, args.batch_size)


def _get_dataset_paths(experiment_name):
    if experiment_name == "erc":
        return ERC_DATASET_PATHS["train"], ERC_DATASET_PATHS["val"], ERC_DATASET_PATHS["test"]


def _extract_relevant_data(dataset):
    # Extract all utterances and emotions
    utterance_list = []
    emotions_list = []
    for dialogue in dataset:
        utterance_list = utterance_list + dialogue["utterances"]
        emotions_list = emotions_list + dialogue["emotions"]

    # Encode emotions
    encoded_emotions_list = [EMOTIONS_ERC[emotion] for emotion in emotions_list]
    return encoded_emotions_list, utterance_list


def _create_dataloaders(train_dataset_path, test_dataset_path, batch_size):
    with open(train_dataset_path) as f:
        train_dataset = json.load(f)

    with open(test_dataset_path) as f:
        test_dataset = json.load(f)

    train_encoded_emotions, train_utterances = _extract_relevant_data(train_dataset)
    test_encoded_emotions, test_utterances = _extract_relevant_data(test_dataset)

    # Get pretrained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Concatenate utterances with emotions
    train_wrapped_dataset = DialogueDatasetWrapper(data=train_utterances,
                                                   labels=train_encoded_emotions,
                                                   tokenizer=tokenizer)
    test_wrapped_dataset = DialogueDatasetWrapper(data=test_utterances,
                                                  labels=test_encoded_emotions,
                                                  tokenizer=tokenizer)

    # Create weighted random sampler to counteract the imbalanced dataset
    # weighted_random_sampler = WeightedRandomSampler(weights=train_wrapped_dataset.class_weights,
    #                                                 num_samples=len(train_wrapped_dataset))

    # train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, sampler=weighted_random_sampler)
    train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_wrapped_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

import torch
from torch.utils.data import Dataset


class DialogueDatasetWrapper(Dataset):
    def __init__(self, data, labels, vocab_size):
        self.data = data
        self.labels = labels
        self.vocab_size = vocab_size
        self.distinct_labels_count = len(set(labels))
        self.class_distribution = torch.unique(torch.tensor(labels), return_counts=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

import torch
from torch.utils.data import Dataset


class DialogueDatasetWrapper(Dataset):
    def __init__(self, data, labels, vocab_size):
        self.data = data
        self.labels = labels
        self.vocab_size = vocab_size
        self.distinct_labels_count = len(set(labels))
        self.class_distribution = torch.unique(torch.tensor(labels), return_counts=True)
        self.class_weights = self.__compute_class_weights()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __compute_class_weights(self):
        weights_idx = []
        for i in range(self.distinct_labels_count):
            weights_idx.append(1 / self.class_distribution[1][i])
            # weights.append(len(self.labels) / self.class_distribution[1][i])

        weights = []
        for _, label in enumerate(self.labels):
            weights.append(weights_idx[label])

        return weights

    def get_class_weights(self):
        return self.class_weights

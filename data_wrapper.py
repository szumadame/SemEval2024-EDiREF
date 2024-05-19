import torch
from torch.utils.data import Dataset


class DialogueDatasetWrapper(Dataset):
    def __init__(self, data, labels, vocab_size, max_length, tokenizer):
        self.data = data
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.distinct_labels_count = len(set(labels))
        self.class_distribution = torch.unique(torch.tensor(labels), return_counts=True)
        self.class_weights = self.__compute_class_weights()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(sequence, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                  truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}

    def __compute_class_weights(self):
        weights_idx = []
        for i in range(self.distinct_labels_count):
            weights_idx.append(1 / self.class_distribution[1][i])
            # weights.append(len(self.labels) / self.class_distribution[1][i])

        weights = []
        for _, label in enumerate(self.labels):
            weights.append(weights_idx[label])

        return weights

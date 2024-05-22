import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset


class DialogueDatasetWrapper(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.vocab_size = tokenizer.vocab_size
        self.max_length = tokenizer.model_max_length
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
        weights_idx = compute_class_weight(class_weight="balanced", classes=np.unique(self.labels), y=self.labels)

        weights = []
        for _, label in enumerate(self.labels):
            weights.append(weights_idx[label])

        return weights

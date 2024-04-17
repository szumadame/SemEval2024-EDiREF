import json
import string
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Embedding
import torch

def tokenize(text):
    return text.split()


with open('data/erc/MaSaC_train_erc.json') as f:
    dataset = json.load(f)

# Build the vocabulary
tokenized_utterances = [tokenize(utt.lower()) for utt in dataset[0]["utterances"]]  # Lowercasing for uniformity
vocab = Counter(word.strip(string.punctuation) for utterance in tokenized_utterances for word in utterance)

# Assign an index to each word in the vocabulary starting from 2
# We reserve '0' for padding and '1' for unknown words
word_to_index = {word: i + 2 for i, (word, _) in enumerate(vocab.items())}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(word_to_index) + 2  # Including padding and unknown word tokens

# Convert the tokenized utterances to integer sequences
sequences = [[word_to_index.get(word, 1) for word in utterance] for utterance in tokenized_utterances]

# Pad the sequences to have equal length
sequence_tensor = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)

# Create a simple embedding layer using PyTorch, uninitialized
embedding_dim = 300  # Example embedding size
embedding_layer = Embedding(vocab_size, embedding_dim)

# Example of how to get embeddings for the first sequence
embedded_sequence = embedding_layer(sequence_tensor[0].unsqueeze(0))  # Unsqueezing to add batch dimension

# word_to_index, sequence_tensor, embedded_sequence.shape
x = 1

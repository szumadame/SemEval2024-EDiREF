import torch.nn as nn

# Assuming `data` is a preprocessed dataset of tokenized conversations with corresponding emotion labels
# and `vocab_size` is the number of unique tokens in your dataset.

# Hyperparameters
vocab_size = 10000  # Example vocabulary size
embedding_dim = 300
hidden_dim = 256
output_dim = 8  # Number of emotions


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]  # Get the last layer's hidden state
        out = self.fc(hidden.squeeze(0))
        return out

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]  # Get the last layer's hidden state
        out = self.fc(hidden.squeeze(0))
        return out


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, output_dim):
        super(EncoderClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.emb(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = x.max(dim=1)[0]
        out = self.linear(x)
        return out

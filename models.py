import torch.nn as nn
from transformers import BertModel


def create_model(model_name, vocab_size, max_length, output_dim):
    if model_name == "lstm":
        return LSTM(vocab_size=vocab_size, embedding_dim=max_length, hidden_dim=256, output_dim=output_dim)
    elif model_name == "bert":
        return BERTClassifier("bert-base-multilingual-cased", output_dim=output_dim)
    elif model_name == "transformer":
        return EncoderClassifier(vocab_size=vocab_size, embedding_dim=300, num_layers=3, num_heads=4,
                                 output_dim=output_dim)
    else:
        raise NotImplemented


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=4, bidirectional=True, dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]  # Get the last layer's hidden state
        out = self.fc(hidden.squeeze(0))
        return out


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, output_dim):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)


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

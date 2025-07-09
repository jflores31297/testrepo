# =========================
# Model Definitions
# =========================

class RNNClassifier(nn.Module):
    """
    Basic RNN model for sentiment classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        # Use last hidden state for classification
        out = self.fc(hidden[-1])
        return out

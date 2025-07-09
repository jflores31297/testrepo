# =========================
# Data Preparation Section
# =========================

# --- For Model 2: Remove 'neutral' examples and use separate fields ---
TEXT_bin = data.Field(sequential=True, batch_first=True, lower=True)
LABEL_bin = data.LabelField()

fields_bin = {'text': TEXT_bin, 'label': LABEL_bin}

train_data_bin = Dataset([ex for ex in train_data.examples if ex.label != 'neutral'], fields_bin)
val_data_bin   = Dataset([ex for ex in val_data.examples if ex.label != 'neutral'], fields_bin)
test_data_bin  = Dataset([ex for ex in test_data.examples if ex.label != 'neutral'], fields_bin)

# Build vocabulary for Model 2 with GloVe
TEXT_bin.build_vocab(train_data_bin, vectors="glove.6B.100d")
LABEL_bin.build_vocab(train_data_bin)

# =========================
# Model Definitions
# =========================

class BiLSTMAttnClassifier(nn.Module):
    """
    Bidirectional LSTM with Attention for sentiment classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, padding_idx, num_layers=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, label_size)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # (batch, seq, emb)
        lstm_out, _ = self.lstm(embedded)              # (batch, seq, hidden*2)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)       # (batch, hidden*2)
        output = self.fc(self.dropout(context))                   # (batch, label_size)
        return output

# Early stopping utility for Model 2
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0

# =========================
# Training and Evaluation
# =========================

def run_bilstm_attention():
    """
    Train and evaluate the improved BiLSTM+Attention model (expected >70% accuracy).
    """
    print("\\n==== Training BiLSTM+Attention Model ====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    vocab_size = len(TEXT_bin.vocab)
    label_size = len(LABEL_bin.vocab)
    padding_idx = TEXT_bin.vocab.stoi['<pad>']
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 3
    dropout = 0.5
    batch_size = 64
    learning_rate = 0.005
    num_epochs = 20

    # Iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data_bin, val_data_bin, test_data_bin),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device
    )

    # Model, loss, optimizer
    model = BiLSTMAttnClassifier(vocab_size, embedding_dim, hidden_dim, label_size, padding_idx, num_layers, dropout).to(device)
    # Load pretrained GloVe embeddings
    model.embedding.weight.data.copy_(TEXT_bin.vocab.vectors)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    early_stopping = EarlyStopping(patience=3)
    best_val_acc = 0
    best_model = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_iter, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_iter, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Test
    test_loss, test_acc = evaluate_model(best_model, test_iter, criterion, device)
    print(f'BiLSTM+Attention Test Accuracy: {test_acc:.2f}%')      

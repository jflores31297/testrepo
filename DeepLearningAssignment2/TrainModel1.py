# =========================
# Training and Evaluation
# =========================

def run_basic_rnn():
    """
    Train and evaluate the basic RNN model
    """
    print("\\n==== Training Basic RNN Model ====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    vocab_size = len(TEXT.vocab)
    label_size = len(LABEL.vocab)
    padding_idx = TEXT.vocab.stoi['<pad>']
    embedding_dim = 128
    hidden_dim = 128
    batch_size = 32
    num_epochs = 10

    # Iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device
    )

    # Model, loss, optimizer
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, label_size, padding_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        print()

    # Test
    test_loss, test_acc = evaluate_model(best_model, test_iter, criterion, device)
    print(f'Basic RNN Test Accuracy: {test_acc:.2f}%')

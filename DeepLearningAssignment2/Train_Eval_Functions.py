# =========================
# Utility Functions
# =========================

def train_model(model, train_iter, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_iter:
        text, label = batch.text.to(device), batch.label.to(device)
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    return total_loss / len(train_iter), 100 * correct / total

def evaluate_model(model, data_iter, criterion, device):
    """
    Evaluate the model on validation or test set.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in data_iter:
            text, label = batch.text.to(device), batch.label.to(device)
            outputs = model(text)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return total_loss / len(data_iter), 100 * correct / total

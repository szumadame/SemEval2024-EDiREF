import torch
from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, test_dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(y.cpu().tolist())

    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

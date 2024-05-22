import torch
from sklearn.metrics import accuracy_score, classification_report

from models import LSTM, BERTClassifier


def evaluate(model, test_dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if isinstance(model, LSTM):
                outputs = model(input_ids)
            elif isinstance(model, BERTClassifier):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                raise NotImplemented

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions,
                                                                             zero_division=0.0)

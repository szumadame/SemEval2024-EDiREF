import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch import optim

from evaluate import evaluate
from models import LSTM, BERTClassifier


def train(model, train_dataloader, test_dataloader, device, args):
    model.train()
    # class_weights = train_dataloader.dataset.get_class_weights()
    # class_weights_tensor = torch.FloatTensor(class_weights).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        model.train()
        predictions = []
        actual_labels = []
        losses = []
        start = time.time()

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            if isinstance(model, LSTM):
                outputs = model(input_ids)
            elif isinstance(model, BERTClassifier):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                raise NotImplemented

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
                losses.append(loss.item())

        train_results = classification_report(actual_labels, predictions, zero_division=0.0, output_dict=True)
        val_results = evaluate(model=model, test_dataloader=test_dataloader, device=device, output_dict=True)

        print("\nEpoch: {}/{} [{} s]"
              .format(epoch,
                      args.n_epochs,
                      np.round(time.time() - start), 3))

        print("Training   |   accuracy: {}, weighted F1-score: {}, loss: {}"
              .format(np.round(train_results["accuracy"], 3),
                      np.round(train_results["weighted avg"]["f1-score"], 3),
                      np.round(np.mean(losses), 3)))

        print("Validation |   accuracy: {}, weighted F1-score: {}"
              .format(np.round(val_results["accuracy"], 3),
                      np.round(val_results["weighted avg"]["f1-score"], 3)))

    return model

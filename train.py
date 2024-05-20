import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import optim

from evaluate import evaluate


def lstm(model, train_dataloader, test_dataloader, device, args):
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
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(y.cpu().tolist())
                losses.append(loss.item())

        print("\nEpoch: {}/{} [{} s]"
              .format(epoch,
                      args.n_epochs,
                      np.round(time.time() - start), 3))

        print("Training loss: {}, accuracy: {} %"
              .format(np.round(np.mean(losses), 3),
                      np.round(accuracy_score(actual_labels, predictions) * 100, 3)))

        val_accuracy, _ = evaluate(model=model, test_dataloader=test_dataloader, device=device)
        print("Validation accuracy: {} %"
              .format(np.round(val_accuracy * 100, 3)))

    return model


def bert(model, train_dataloader, test_dataloader, device, args):
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
                losses.append(loss.item())

        print("\nEpoch: {}/{} [{} s]"
              .format(epoch,
                      args.n_epochs,
                      np.round(time.time() - start), 3))

        print("Training loss: {}, accuracy: {} %"
              .format(np.round(np.mean(losses), 3),
                      np.round(accuracy_score(actual_labels, predictions) * 100, 3)))

        val_accuracy, _ = evaluate(model=model, test_dataloader=test_dataloader, device=device)
        print("Validation accuracy: {} %"
              .format(np.round(val_accuracy * 100, 3)))

    return model

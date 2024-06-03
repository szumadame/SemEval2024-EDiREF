import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch import optim

import wandb
from evaluate import evaluate
from models import LSTMClassifier, BERTClassifier, EncoderClassifier


def train(model, train_dataloader, test_dataloader, device, args):
    if args.log_wandb:
        wandb.watch(model)
    best_model = copy.deepcopy(model)
    best_f1 = 0

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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

            if isinstance(model, LSTMClassifier):
                outputs = model(input_ids)
            elif isinstance(model, EncoderClassifier):
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

        scheduler.step()

        train_results = classification_report(actual_labels, predictions, zero_division=0.0, output_dict=True)
        val_results = evaluate(model=model, test_dataloader=test_dataloader, device=device, output_dict=True)

        if val_results["weighted avg"]["f1-score"] > best_f1:
            best_model = copy.deepcopy(model)
            best_f1 = val_results["weighted avg"]["f1-score"]
            # torch.save(model, f'models/net_latest')

        if args.log_wandb:
            wandb.log({"Training loss": np.round(np.mean(losses), 3)})
            wandb.log({"Training accuracy": np.round(train_results["accuracy"], 3)})
            wandb.log({"Training weighted f1-score": np.round(train_results["weighted avg"]["f1-score"], 3)})

            wandb.log({"Validation accuracy": np.round(val_results["accuracy"], 3)})
            wandb.log({"Validation weighted f1-score": np.round(val_results["weighted avg"]["f1-score"], 3)})

        print("\nEpoch: {}/{} [{} s]"
              .format(epoch,
                      args.n_epochs,
                      np.round(time.time() - start), 3))

        print("Training   |   accuracy: {}, weighted f1-score: {}, loss: {}"
              .format(np.round(train_results["accuracy"], 3),
                      np.round(train_results["weighted avg"]["f1-score"], 3),
                      np.round(np.mean(losses), 3)))

        print("Validation |   accuracy: {}, weighted f1-score: {}"
              .format(np.round(val_results["accuracy"], 3),
                      np.round(val_results["weighted avg"]["f1-score"], 3)))

    return best_model

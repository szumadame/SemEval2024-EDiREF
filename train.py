import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim


def train(model, train_dataloader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 100
    for epoch in range(n_epochs):
        losses = []
        accuracy = 0
        total = 0
        start = time.time()

        for iteration, batch in enumerate(train_dataloader):
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = get_correct_sum(outputs, y)
                accuracy += acc.item()
                total += len(y)
                losses.append(loss.item())

        print("Epoch: {}/{}, loss: {}, Acc: {} %, took: {} s".format(epoch, n_epochs,
                                                                     np.round(np.mean(losses), 3),
                                                                     np.round(accuracy * 100 / total, 3),
                                                                     np.round(time.time() - start), 3))


def get_correct_sum(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum

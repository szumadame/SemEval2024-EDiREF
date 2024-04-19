import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim


def train(model, train_dataloader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 100
    for epoch in range(n_epochs):
        losses = []
        correctly_predicted = 0
        total_samples = 0
        start = time.time()

        for iteration, batch in enumerate(train_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correctly_predicted += (get_correct_sum(outputs, y)).item()
                total_samples += len(y)
                losses.append(loss.item())

        print("Epoch: {}/{}, loss: {}, Acc: {} %, in: {} s"
              .format(epoch, n_epochs,
                      np.round(np.mean(losses), 3),
                      np.round(correctly_predicted * 100 / total_samples, 3),
                      np.round(time.time() - start), 3))


def get_correct_sum(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum

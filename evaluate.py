import numpy as np
import torch


def evaluate(model, test_dataloader, device):
    model.eval()
    correctly_predicted = 0
    total_samples = 0

    for iteration, batch in enumerate(test_dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        correctly_predicted += (get_correct_sum(outputs, y)).item()
        total_samples += len(y)

    print("Test accuracy: {} %".format(np.round(correctly_predicted * 100 / total_samples, 3)))


def get_correct_sum(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum

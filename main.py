import sys

from options import get_args
from setup import setup_devices
from dataset_generation import get_dataloaders
from models import LSTM
from train import train


def run(args):
    train_dataloader, val_dataloader = get_dataloaders(args)
    vocab_size = train_dataloader.dataset.vocab_size
    distinct_labels_count = train_dataloader.dataset.distinct_labels_count
    model = LSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=256, output_dim=distinct_labels_count)
    train(model=model, train_dataloader=train_dataloader)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    device = setup_devices(args)
    run(args)

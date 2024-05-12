import sys

from dataset_generation import get_dataloaders
from evaluate import evaluate
from models import LSTM, EncoderClassifier
from options import get_args
from setup import setup_devices
from train import train


def run(args):
    device = setup_devices(args)
    train_dataloader, val_dataloader = get_dataloaders(args)
    vocab_size = train_dataloader.dataset.vocab_size
    distinct_labels_count = train_dataloader.dataset.distinct_labels_count
    # model = LSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=256, output_dim=distinct_labels_count).to(device)
    model = EncoderClassifier(vocab_size=vocab_size, embedding_dim=300, num_layers=4, num_heads=10,
                              output_dim=distinct_labels_count).to(device)
    model = train(model=model, train_dataloader=train_dataloader, n_epochs=args.n_epochs, device=device, lr=args.lr)
    evaluate(model=model, test_dataloader=val_dataloader, device=device)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    run(args)

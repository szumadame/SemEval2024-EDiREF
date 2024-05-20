import sys

import train
from dataset_generation import get_dataloaders
from evaluate import evaluate
from models import create_model
from options import get_args
from setup import setup_devices


def run(args):
    device = setup_devices(args)
    train_dataloader, val_dataloader = get_dataloaders(args)
    vocab_size = train_dataloader.dataset.vocab_size
    output_dim = train_dataloader.dataset.distinct_labels_count
    model = create_model(args.model, vocab_size=vocab_size, output_dim=output_dim).to(device)

    model = train.__dict__[args.model](model=model,
                                       train_dataloader=train_dataloader,
                                       test_dataloader=val_dataloader,
                                       device=device,
                                       args=args)

    _, classification_report = evaluate(model=model, test_dataloader=val_dataloader, device=device)
    print(f'\nFinal evaluation')
    print(f'Classification report:\n {classification_report}')


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    run(args)

import sys

import wandb
from dataset_generation import get_dataloaders
from evaluate import evaluate
from models import create_model
from options import get_args
from setup import setup_devices
from train import train


def run(args):

    if args.log_wandb:
        wandb.init(project=f"SemEval2024_{args.experiment_name}")

    device = setup_devices(args)
    train_dataloader, val_dataloader = get_dataloaders(args)

    model = create_model(model_name=args.model,
                         vocab_size=train_dataloader.dataset.vocab_size,
                         max_length=train_dataloader.dataset.max_length,
                         dropout=args.dropout,
                         output_dim=train_dataloader.dataset.distinct_labels_count).to(device)

    model = train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=val_dataloader,
                  device=device,
                  args=args)

    classification_report = evaluate(model=model, test_dataloader=val_dataloader, device=device, output_dict=False)
    print(f'\nFinal evaluation')
    print(f'Classification report:\n {classification_report}')


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    run(args)

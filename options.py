import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--model', type=str, default='lstm', help='Network model')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weighted_sampler', type=bool, default=False,
                        help='Use weighted random sampler to counteract the imbalanced dataset')
    parser.add_argument('--create_classification_report', type=bool, default=False,
                        help='Generate a complete classification metrics report at the end of each epoch')
    parser.add_argument('--seed', type=int, required=False,
                        help='Random seed. If defined all random operations will be reproducible')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help='The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only')

    return parser.parse_args(argv)

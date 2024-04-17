import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_wandb', default=False, action='store_true', help="Log training process on wandb")

    return parser.parse_args(argv)

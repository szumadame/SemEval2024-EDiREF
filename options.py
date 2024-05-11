import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")

    return parser.parse_args(argv)

import sys

from options import get_args
from setup import setup_devices


def run(args):
    pass


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    device = setup_devices(args)
    run(args)

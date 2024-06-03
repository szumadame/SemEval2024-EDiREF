import random

import numpy as np
import torch


def setup_devices(args):
    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: No seed used")

    return device

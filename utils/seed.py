import torch
import numpy as np
import random

from utils.dist import get_rank

def setup_seed(args):
    if args.seed is None: return
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
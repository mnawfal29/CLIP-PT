import argparse
import random, os
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L_g", type=int, default=4, help="Sequence Length of general prompts to be inserted per layer")
    parser.add_argument("--L_s", type=int, default=4, help="Sequence Length of shared prompts to be inserted per layer")
    parser.add_argument("--D_g", type=int, default=8, help="Number of layers (Depth) in which the general prompts will be inserted")
    parser.add_argument("--D_s", type=int, default=4, help="Number of layers (Depth) in which the shared prompts will be inserted")
    parser.add_argument("--text_replace_method", type=str, default="replace", choices=["replace", "accumalate"], help= "Method to replace the text prompts in the encoders. Options: replace, accumulate")
    parser.add_argument("--vision_replace_method", type=str, default="accumulate", choices=["replace", "accumulate"], help="Method to replace the vision prompts in the encoders. Options: replace, accumulate")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cub200", "miniimagenet"], help="Name of the dataset to be used. Options: cifar100, cub200, miniimagenet")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to be executed", required=False)
    parser.add_argument("--txt_beta", type=float, default=0, help="Weight for text dispersion loss", required=False)
    parser.add_argument("--seed", type=int, default=42, help="Seed to be used in the run", required=False)

    args = parser.parse_args()
    return args

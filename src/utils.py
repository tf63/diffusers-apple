import os
import random
from functools import wraps

import click
import numpy as np
import torch


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class ExperimentalContext:
    def __init__(self, seed, device, root_dir='out'):
        self.seed = seed
        self.device = device
        torch.set_default_device(device)

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

        fix_seed(seed)

    def save_image(self, image, exp_name, image_name):
        out_dir = os.path.join(self.root_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f'seed{self.seed}_{image_name}.png')
        image.save(out_path)

        print(f'Image has been saved successfully to {out_path}')


def options(func):
    @click.command()
    @click.option('--seed', type=int, default=42, help='Random seed for reproducibility.')
    @click.option('--device', type=str, default='mps', help='Device to run the computation on.')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

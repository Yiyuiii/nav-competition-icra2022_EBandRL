import numpy as np
import torch

def set_seed(seed=None, torch_deterministic=False):
    import os
    import random

    if (seed is None) and torch_deterministic:
        seed = 42
    elif seed is None:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def time_dhms(s):
    s = int(s)
    day = s // 86400
    s %= 86400
    hour = s // 3600
    s %= 3600
    minute = s // 60
    s %= 60
    return day, hour, minute, s


def time_text(s):
    d, h, m, s = time_dhms(s)
    return f'{d}d {h:>2d}:{m:>2d}:{s:>2d}'


def do_state_norm(state, mean, var):
    if isinstance(var, torch.Tensor):
        return (state - mean) / (var.sqrt() + 1e-5)
    else:
        return (state - mean) / (np.sqrt(var) + 1e-5)


class StateNorm:
    def __init__(self, change_ratio_min=0):
        self.reset()
        self.change_ratio_min = change_ratio_min

    def reset(self):
        self.mean = 0
        self.var = 1
        self.n = 0

    def update(self, n, mean, var):
        assert n>0
        if self.n==0:
            self.mean = mean
            self.var = var
            self.n = n
            return mean, var
        else:
            self.n += n
            change_ratio = max(n / self.n, self.change_ratio_min)
            self.mean = (1 - change_ratio) * self.mean + change_ratio * mean
            self.var = (1 - change_ratio) * self.var + change_ratio * var
            return self.mean, self.var

    def get(self):
        return self.mean, self.var

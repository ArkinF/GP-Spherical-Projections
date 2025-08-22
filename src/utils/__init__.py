from .metrics import compute_metrics, extract_hyp, compute_cov95_and_piwidth
from .plotting import plot_posteriors_auto
from .mean_functions import create_mean_function

def set_seed(seed):
    """set seed for reproducibility """
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

__all__ = [
    'compute_metrics',
    'extract_hyp',
    'compute_cov95_and_piwidth',
    'plot_posteriors_auto',
    'create_mean_function',
    'set_seed'
]

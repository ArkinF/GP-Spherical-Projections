"""
Functions to generate various types of synthetic datasets for testing
"""

import torch
import numpy as np
from .utils import set_seed


def generate_toy_1d(N=1000):
    """Generate 1D toy data with periodic structure"""
    set_seed(0)
    x = 100 * np.random.rand(N) + 0.1 * np.random.randn(N)
    y = 20 * np.cos(1.5 * x) * np.cos(0.1 * x) + 0.1 * np.random.randn(N)
    x_plot = np.linspace(0, 100, 200)

    n_train = int(0.7*N)

    x_train = torch.tensor(x[:n_train], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    x_test = torch.tensor(x[n_train:], dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y[n_train:], dtype=torch.float32)

    return x_train, y_train, x_test, y_test, x_plot


def generate_toy_2d(N=1200):
    """Generate 2D toy data with complex surface."""
    def f_true_xy(x):
        x1, x2 = x[..., 0], x[..., 1]
        return (2.5*torch.sin(1.2*x1)*torch.cos(0.7*x2)
                + 0.3*x1 - 0.2*x2 + 0.8*torch.sin(0.3*(x1+x2)))

    def make_toy2d(N, noise=0.2, seed=0):
        rng = np.random.RandomState(seed)
        X = np.column_stack([rng.uniform(0.0, 10.0, N),
                             rng.uniform(0.0, 10.0, N)])
        X_t = torch.tensor(X, dtype=torch.float32)
        y_clean = f_true_xy(X_t)
        y = y_clean + noise*torch.randn_like(y_clean)
        idx = rng.permutation(N)
        n_tr = int(0.8*N)
        tr, te = idx[:n_tr], idx[n_tr:]
        x_train = X_t[tr]; y_train = y[tr]
        x_test  = X_t[te]; y_test  = y[te]
        return x_train, y_train, x_test, y_test

    def make_grid_2d(xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, G=120):
        x1 = np.linspace(xmin, xmax, G)
        x2 = np.linspace(ymin, ymax, G)
        X1g, X2g = np.meshgrid(x1, x2)
        Xg = np.c_[X1g.ravel(), X2g.ravel()]
        return torch.tensor(Xg, dtype=torch.float32), (G, G), (xmin, xmax, ymin, ymax)

    set_seed(0)

    x_train, y_train, x_test, y_test = make_toy2d(N=N, noise=0.2, seed=0)
    x_grid, grid_shape, extent = make_grid_2d(0.0, 10.0, 0.0, 10.0, G=120)

    return x_train, y_train, x_test, y_test, x_grid, grid_shape, extent


def generate_piecewise_data(N=100):
    """Generate piecewise smooth function data"""
    torch.manual_seed(0)
    np.random.seed(0)
    x = np.linspace(0, 10, N)
    
    y = np.zeros(N)
    # Different functions in different regions
    mask1 = x < 3
    mask2 = (x >= 3) & (x < 7)
    mask3 = x >= 7
    
    y[mask1] = 2 * np.sin(2 * x[mask1])
    y[mask2] = 1.5 * np.cos(3 * x[mask2]) + 0.5
    y[mask3] = 0.5 * np.exp(-(x[mask3] - 7) / 2)
    
    y += 0.1 * np.random.randn(N)
    # Randomly shuffle the data
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    x = x[perm]
    y = y[perm]

    n_train = int(0.7 * N)
    x_train = torch.tensor(x[:n_train], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    x_test = torch.tensor(x[n_train:], dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y[n_train:], dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test, x


def generate_piecewise_linear_data(N=1000):
    """Generate piecewise linear data with different slopes and intercepts"""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Define piecewise linear function
    def piecewise_linear(x):
        if x < 2:
            return 2 * x + 1      # slope=2, intercept=1
        elif x < 5:
            return -1.5 * x + 8   # slope=-1.5, intercept=8  
        elif x < 8:
            return 0.5 * x - 2    # slope=0.5, intercept=-2
        else:
            return -x + 10        # slope=-1, intercept=10
    
    # Generate x values
    x = np.linspace(0, 10, N)
    y = np.array([piecewise_linear(xi) for xi in x])
    
    # Add noise
    y += 0.1 * np.random.randn(N)
    
    # Randomly shuffle the data
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    x = x[perm]
    y = y[perm]
    
    # Split into train/test
    n_train = int(0.7 * N)
    x_train = torch.tensor(x[:n_train], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    x_test = torch.tensor(x[n_train:], dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y[n_train:], dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test, x


def generate_multiscale_data(N=1000):
    """Generate data with multiple length scales"""
    set_seed(0)
    x = np.linspace(0, 20, N)
    
    # Combine different frequency components
    y = (2.0 * np.sin(0.5 * x) +           # long period
         0.5 * np.sin(3.0 * x) +           # medium period  
         0.1 * np.sin(10.0 * x) +          # short period
         0.05 * np.random.randn(N))        # noise
    
    # Randomly shuffle
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    x = x[perm]
    y = y[perm]
    
    n_train = int(0.7 * N)
    x_train = torch.tensor(x[:n_train], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    x_test = torch.tensor(x[n_train:], dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y[n_train:], dtype=torch.float32)
    
    x_plot = np.linspace(0, 20, 200)
    
    return x_train, y_train, x_test, y_test, x_plot

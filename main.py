#!/usr/bin/env python3
"""
Main script for running Gaussian Process experiments.

Usage:
    python main.py --experiment toy_1d
    python main.py --experiment toy_2d
    python main.py --experiment co2
    python main.py --experiment d_sweep
"""

import argparse
import sys
import torch
import numpy as np

# add src to path
sys.path.append('src')

from src.experiments import run_exact_proj_svgp, run_sweep_d_agg
from src.config import TrainCfg, SVGPcfg
from src.kernels import SM_kernel

def generate_toy_1d():
    """generate 1D toy data"""
    torch.manual_seed(0)
    np.random.seed(0)
    N = 1000
    x = 100 * np.random.rand(N) + 0.1 * np.random.randn(N)
    y = 20 * np.cos(1.5 * x) * np.cos(0.1 * x) + 0.1 * np.random.randn(N)
    x_plot = np.linspace(0, 100, 200)

    n_train = int(0.7*N)

    x_train = torch.tensor(x[:n_train], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    x_test = torch.tensor(x[n_train:], dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y[n_train:], dtype=torch.float32)

    return x_train, y_train, x_test, y_test, x_plot

def generate_toy_2d():
    """generate 2D toy data"""
    def f_true_xy(x):
        x1, x2 = x[..., 0], x[..., 1]
        return (2.5*torch.sin(1.2*x1)*torch.cos(0.7*x2)
                + 0.3*x1 - 0.2*x2 + 0.8*torch.sin(0.3*(x1+x2)))

    def make_toy2d(N=1200, noise=0.2, seed=0):
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

    torch.manual_seed(0)
    np.random.seed(0)

    x_train, y_train, x_test, y_test = make_toy2d(N=1200, noise=0.2, seed=0)
    x_grid, grid_shape, extent = make_grid_2d(0.0, 10.0, 0.0, 10.0, G=120)

    return x_train, y_train, x_test, y_test, x_grid, grid_shape, extent

def run_toy_1d_experiment():
    """run 1D toy experiment"""
    print("running 1D toy experiment...")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    d_proj = 300
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=d_proj, num_mixtures=1,
        x_plot=x_plot, add_svgp=False, 
        init_hyps_data=False,
        savepath='results/toy_1d_single_mixture.png'
    )

def run_toy_2d_experiment():
    """run 2D toy experiment"""
    print("running 2D toy experiment...")
    x_train, y_train, x_test, y_test, x_grid, grid_shape, extent = generate_toy_2d()
    
    d_proj = 400
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        x_grid=x_grid, d_proj=d_proj, num_mixtures=4,
        add_svgp=True, init_hyps_data=False, 
        grid_shape=grid_shape, extent=extent,
        savepath='results/toy_2d_four_mixtures.png'
    )

def run_d_sweep_experiment():
    """run d parameter sweep experiment"""
    print("running d sweep experiment...")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    # run exact GP baseline
    from src.training import run_exact
    res_exact = run_exact(
        x_train, y_train, x_test, y_test,
        kernel=SM_kernel(1, 1, y_std=float(torch.std(y_train))),
        cfg=TrainCfg(verbose=False),
        init_from_data=False
    )

    # sweep over d values
    d_list = [10, 20, 50, 100, 200, 300, 400, 500]
    df_agg, reps_for_best, runs_raw = run_sweep_d_agg(
        x_train, y_train, x_test, y_test,
        d_list=d_list,
        n_repeats=5,  # reduced for faster execution
        num_mixtures=1,
        cfg=TrainCfg(max_iter=500, patience=40, lr=0.05, verbose=False),
        init_hyps_data=False,
        res_exact=res_exact,
        res_svgp=None,
        savepath='results/d_sweep_analysis.png',
        xscale_log=False
    )

def main():
    parser = argparse.ArgumentParser(description='Run GP experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['toy_1d', 'toy_2d', 'd_sweep'],
                       help='Experiment to run')
    
    args = parser.parse_args()
    
    if args.experiment == 'toy_1d':
        run_toy_1d_experiment()
    elif args.experiment == 'toy_2d':
        run_toy_2d_experiment()
    elif args.experiment == 'd_sweep':
        run_d_sweep_experiment()
    else:
        print(f"unknown experiment: {args.experiment}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main script for running Gaussian Process experiments.

Usage:
    python main.py --experiment toy_1d
    python main.py --experiment toy_2d
    python main.py --experiment co2
    python main.py --experiment d_sweep
    python main.py --experiment multiscale_1d
    python main.py --experiment multiscale_comparison
    python main.py --experiment multiscale_sweep
"""

import argparse
import sys
import torch
import numpy as np

# add src to path
sys.path.append('src')

from src.experiments import (
    run_exact_proj_svgp, run_sweep_d_agg
)
from src.config import TrainCfg, SVGPcfg
from src.kernels import SM_kernel

# Import data generation functions from src module
from src.data_generation import generate_toy_1d, generate_toy_2d

def run_toy_1d_experiment():
    print("running 1D toy experiment")
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
    print("running 2D toy experiment")
    x_train, y_train, x_test, y_test, x_grid, grid_shape, extent = generate_toy_2d()
    
    d_proj = 400
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        x_grid=x_grid, d_proj=d_proj, num_mixtures=4,
        add_svgp=True, init_hyps_data=False, 
        grid_shape=grid_shape, extent=extent,
        savepath='results/toy_2d_four_mixtures.png'
    )

def run_matern_experiment():
    print("running Matern kernel experiment")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    from src.kernels import stable_matern_kernel
    
    # Create Matern kernels
    kernel_exact = stable_matern_kernel(init_lengthscale=1.0, nu=2.5)
    kernel_proj = stable_matern_kernel(init_lengthscale=1.0, nu=2.5)
    
    d_proj = 300
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=d_proj,
        kernel_exact=kernel_exact,
        kernel_proj=kernel_proj,
        x_plot=x_plot, add_svgp=False, 
        init_hyps_data=False,
        savepath='results/toy_1d_matern.png'
    )

def run_periodic_plus_rbf_experiment():
    print("running periodic + RBF kernel experiment")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    from src.kernels import long_periodic_plus_short_rbf
    
    # Create periodic + RBF kernels
    kernel_exact = long_periodic_plus_short_rbf(init_period=1.0, init_rbf_lengthscale=0.1)
    kernel_proj = long_periodic_plus_short_rbf(init_period=1.0, init_rbf_lengthscale=0.1)
    
    d_proj = 300
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=d_proj,
        kernel_exact=kernel_exact,
        kernel_proj=kernel_proj,
        x_plot=x_plot, add_svgp=False, 
        init_hyps_data=False,
        savepath='results/toy_1d_periodic_plus_rbf.png'
    )

def run_rbf_experiment():
    print("running RBF kernel experiment")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    from src.kernels import stable_rbf_kernel
    
    # Create RBF kernels
    kernel_exact = stable_rbf_kernel(init_lengthscale=1.0)
    kernel_proj = stable_rbf_kernel(init_lengthscale=1.0)
    
    d_proj = 300
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=d_proj,
        kernel_exact=kernel_exact,
        kernel_proj=kernel_proj,
        x_plot=x_plot, add_svgp=False, 
        init_hyps_data=False,
        savepath='results/toy_1d_rbf.png'
    )

def run_d_sweep_experiment():
    print("running d sweep experiment")
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

def run_d_sweep_matern_experiment():
    print("running d sweep experiment with Matern kernel")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    from src.kernels import stable_matern_kernel
    from src.training import run_exact
    
    # Create Matern kernel
    kernel_matern = stable_matern_kernel(init_lengthscale=1.0, nu=2.5)
    
    # run exact GP baseline
    res_exact = run_exact(
        x_train, y_train, x_test, y_test,
        kernel=kernel_matern,
        cfg=TrainCfg(verbose=False),
        init_from_data=False
    )

    # sweep over d values
    d_list = [10, 20, 50, 100, 200, 300, 400, 500]
    df_agg, reps_for_best, runs_raw = run_sweep_d_agg(
        x_train, y_train, x_test, y_test,
        d_list=d_list,
        n_repeats=5,  # reduced for faster execution
        kernel_proj=kernel_matern,
        cfg=TrainCfg(max_iter=500, patience=40, lr=0.05, verbose=False),
        init_hyps_data=False,
        res_exact=res_exact,
        res_svgp=None,
        savepath='results/d_sweep_matern_analysis.png',
        xscale_log=False
    )

def run_multiscale_1d_experiment():
    print("running 1D multi-scale projection experiment")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    from src.experiments import run_exact_proj_svgp
    
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=300, num_mixtures=1,
        x_plot=x_plot, add_svgp=False, 
        init_hyps_data=False,
        use_multiscale=True,
        d_list=[50, 100, 150],
        savepath='results/toy_1d_multiscale.png'
    )

def run_multiscale_2d_experiment():
    print("running 2D multi-scale projection experiment")
    x_train, y_train, x_test, y_test, x_grid, grid_shape, extent = generate_toy_2d()
    
    from src.experiments import run_exact_proj_svgp
    
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=600, num_mixtures=2,
        x_grid=x_grid, grid_shape=grid_shape, extent=extent,
        add_svgp=False, init_hyps_data=False,
        use_multiscale=True,
        d_list=[100, 200, 300],  # 2D multi-scale configuration
        savepath='results/toy_2d_multiscale.png'
    )

def run_multiscale_matern_experiment():
    print("running multi-scale Matern kernel experiment")
    x_train, y_train, x_test, y_test, x_plot = generate_toy_1d()
    
    from src.kernels import stable_matern_kernel
    from src.experiments import run_exact_proj_svgp
    
    # create Matern kernel
    kernel_matern = stable_matern_kernel(init_lengthscale=1.0, nu=2.5)
    
    run_exact_proj_svgp(
        x_train, y_train, x_test, y_test, 
        d_proj=300, num_mixtures=1,
        kernel_exact=kernel_matern, kernel_proj=kernel_matern,
        x_plot=x_plot, add_svgp=False, 
        init_hyps_data=False,
        use_multiscale=True,
        d_list=[50, 100, 150],
        savepath='results/toy_1d_multiscale_matern.png'
    )

def main():
    parser = argparse.ArgumentParser(description='Run GP experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=[
                           'toy_1d', 'toy_2d', 'd_sweep', 
                           'matern', 'periodic_plus_rbf', 'rbf', 'd_sweep_matern',
                           'multiscale_1d', 'multiscale_1d_weighted', 'multiscale_2d', 'multiscale_matern'
                       ],
                       help='Experiment to run')
    
    args = parser.parse_args()
    
    # Original experiments
    if args.experiment == 'toy_1d':
        run_toy_1d_experiment()
    elif args.experiment == 'toy_2d':
        run_toy_2d_experiment()
    elif args.experiment == 'd_sweep':
        run_d_sweep_experiment()
    elif args.experiment == 'matern':
        run_matern_experiment()
    elif args.experiment == 'periodic_plus_rbf':
        run_periodic_plus_rbf_experiment()
    elif args.experiment == 'rbf':
        run_rbf_experiment()
    elif args.experiment == 'd_sweep_matern':
        run_d_sweep_matern_experiment()
    
    # Multi-scale experiments
    elif args.experiment == 'multiscale_1d':
        run_multiscale_1d_experiment()
    elif args.experiment == 'multiscale_2d':
        run_multiscale_2d_experiment()
    elif args.experiment == 'multiscale_matern':
        run_multiscale_matern_experiment()
    
    else:
        print(f"unknown experiment: {args.experiment}")

if __name__ == "__main__":
    main()
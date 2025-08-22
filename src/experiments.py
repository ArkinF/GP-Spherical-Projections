import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy # to avoid parameters being shared between kernels
from .config import TrainCfg, SVGPcfg
from .training import run_exact, run_proj, run_svgp, predict_svgp, run_proj_multiscale
from .utils import plot_posteriors_auto
from .kernels import SM_kernel


def run_exact_proj_svgp(
    x_train, y_train, x_test, y_test, d_proj=200,
    num_mixtures=1, kernel_exact=None, kernel_proj=None, kernel_svgp=None,
    cfg: TrainCfg=TrainCfg(), svgp_cfg: SVGPcfg=SVGPcfg(),
    add_svgp=False, init_hyps_data=False,
    # plotting inputs:
    x_plot=None,
    x_grid=None, grid_shape=None, extent=None,
    savepath=None, add_svgp_heatmap=True,
    flag=True,
    # multi-scale toggles:
    use_multiscale=False, d_list=None
):
    """main experiment orchestrator"""
    from .utils import set_seed
    set_seed(cfg.seed)
    
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))

    # default SM kernels only if none provided
    if kernel_exact is None:
        k_exact = SM_kernel(num_mixtures, D, y_std=y_std_train)
    else:
        k_exact = copy.deepcopy(kernel_exact)
        
    if kernel_proj is None:
        k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
    else:
        k_proj = copy.deepcopy(kernel_proj)

    if add_svgp and kernel_svgp is None:
        k_svgp = SM_kernel(num_mixtures, D, y_std=y_std_train)
    elif add_svgp:
        k_svgp = copy.deepcopy(kernel_svgp)

    # exact GP
    res_exact = run_exact(x_train, y_train, x_test, y_test, k_exact, cfg, init_from_data=init_hyps_data, flag=flag)
    
    # projection GP (single-scale or multi-scale)
    if use_multiscale:
        if d_list is None:
            d_list = [d_proj//4, d_proj//2, d_proj//4]  # default split
            current_total = sum(d_list)
            if current_total != d_proj:
                d_list[-1] += (d_proj - current_total)
        
        print(f"Running multi-scale projection with d_list = {d_list} (total d = {sum(d_list)})")
        res_proj = run_proj_multiscale(
            x_train, y_train, x_test, y_test, k_proj, cfg,
            d_list=d_list, init_from_data=init_hyps_data, 
            flag=flag
        )
        
        res_proj['d_list'] = d_list
        res_proj['total_d'] = sum(d_list)
    else:
        res_proj = run_proj(x_train, y_train, x_test, y_test, k_proj, d_proj, cfg, init_from_data=init_hyps_data, flag=flag)
    res_svgp = None
    if add_svgp:
        res_svgp = run_svgp(x_train, y_train, x_test, y_test, k_svgp, svgp_cfg, device=device, flag=flag)

    # results
    method_name = 'multiscale' if use_multiscale else 'spherical'
    df = pd.DataFrame({
        'exact'     : [res_exact['metrics']['RMSE'], res_exact['metrics']['R2'], res_exact['metrics']['NLPD'], res_exact['metrics']['COV95'], res_exact['metrics']['PIW95'], res_exact['train_time']],
        method_name : [res_proj ['metrics']['RMSE'], res_proj ['metrics']['R2'], res_proj ['metrics']['NLPD'], res_proj ['metrics']['COV95'], res_proj ['metrics']['PIW95'], res_proj ['train_time']],
    }, index=['RMSE_test','R2_test','NLPD_test', 'COV95_test', 'PIW95_test', 'time_sec'])

    if add_svgp and res_svgp is not None:
        df['SVGP'] = [res_svgp['metrics']['RMSE'], res_svgp['metrics']['R2'], res_svgp['metrics']['NLPD'], res_svgp['metrics']['COV95'], res_svgp['metrics']['PIW95'], res_svgp['train_time']]

    if flag:
        exact_params = res_exact['params']
        proj_params = res_proj['params']
        
        # parameter names based on kernel
        if kernel_exact is None or hasattr(kernel_exact, 'mixture_weights'):
            param_names = ['weight', 'scale', 'mean', 'noise_var']
        elif hasattr(kernel_exact, 'base_kernel'):
            base_kernel = kernel_exact.base_kernel
            if hasattr(base_kernel, 'period_length'):
                param_names = ['lengthscale', 'period', 'outputscale', 'noise_var']
            elif hasattr(base_kernel, 'kernels'):
                # for composite kernels like periodic * rbf, we have: [lengthscale1, period, lengthscale2, outputscale, noise_var]
                param_names = ['lengthscale1', 'period', 'lengthscale2', 'outputscale', 'noise_var']
            else:
                param_names = ['lengthscale', 'outputscale', 'noise_var'] # rbf, matern
        
        df_hyp = pd.DataFrame({
            'GPyTorch (exact)': exact_params,
            f'GPyTorch ({method_name})' : proj_params,
        }, index=param_names)
        
        if add_svgp:
            df_hyp['GPyTorch (SVGP)'] = res_svgp['params']

    if flag:
        print(df_hyp)
    print(df)
    
    # plotting
    d_proj_for_plot = res_proj.get('total_d', d_proj) if use_multiscale else d_proj
    _ = plot_posteriors_auto(
        D=x_train.shape[1],
        res_exact=res_exact, res_proj=res_proj, res_svgp=res_svgp,
        x_test=x_test, y_test=y_test, x_plot=x_plot,
        x_train=x_train, x_grid=x_grid, grid_shape=grid_shape, extent=extent,
        d_proj=d_proj_for_plot,
        predict_svgp_fn=predict_svgp,
        plot1d_fn=None,
        add_svgp_heatmap=add_svgp_heatmap,
        savepath=savepath 
    )

    return res_exact, res_proj, res_svgp

def run_sweep_d_agg(
    x_train, y_train, x_test, y_test,
    d_list,
    n_repeats: int = 5,
    num_mixtures: int = 1,
    kernel_proj=None,
    cfg: TrainCfg = TrainCfg(),
    init_hyps_data: bool = False,
    res_exact: dict | None = None,
    res_svgp: dict | None = None,
    savepath: str | None = None,
    xscale_log: bool = False
):
    """sweep over d values and aggregate results"""
    # ensure deterministic list, ints
    d_list = sorted(int(d) for d in d_list)

    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))

    # storage
    rows = []
    runs_raw = {}
    results_by_d_first = {}

    for i in tqdm(range(len(d_list))):
        d = d_list[i]
        metrics_runs = []  # each row: [RMSE, R2, NLPD, time_sec]
        res_first = None

        for r in range(n_repeats):
            # Use provided kernel or create default SM kernel
            if kernel_proj is None:
                k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
            else:
                # Create a fresh copy of the provided kernel to avoid parameter sharing
                import copy
                k_proj = copy.deepcopy(kernel_proj)

            # change seed so ProjectionObjective gets different Omega and optimiser noise differs
            seed_shift = 1000 * i + r
            cfg_i = TrainCfg(**{**vars(cfg), 'seed': cfg.seed + seed_shift})

            res_d = run_proj(
                x_train, y_train, x_test, y_test,
                kernel=k_proj, d_proj=d,
                cfg=cfg_i, init_from_data=init_hyps_data
            )
            if res_first is None:
                res_first = res_d

            metrics_runs.append([
                res_d['metrics']['RMSE'],
                res_d['metrics']['R2'],
                res_d['metrics']['NLPD'],
                res_d['train_time'],
            ])

        metrics_runs = np.asarray(metrics_runs)  # shape (n_repeats, 4)
        runs_raw[d] = metrics_runs
        results_by_d_first[d] = res_first

        mean = metrics_runs.mean(axis=0)
        std  = metrics_runs.std(axis=0, ddof=1) if n_repeats > 1 else np.zeros_like(mean)

        rows.append([
            d, mean[0], std[0],  # RMSE
            mean[1], std[1],     # R2
            mean[2], std[2],     # NLPD
            mean[3], std[3],     # time_sec
        ])

    results_df = pd.DataFrame(
        rows,
        columns=[
            'd',
            'RMSE_mean','RMSE_std',
            'R2_mean','R2_std',
            'NLPD_mean','NLPD_std',
            'time_sec_mean','time_sec_std'
        ]
    ).set_index('d')

    print(results_df)

    # plot: mean +/- std vs d with optional baselines
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_rmse, ax_r2, ax_nlpd, ax_time = axes.ravel()

    x = results_df.index.values

    def plot_band(ax, y_mean, y_std, title, ylabel):
        ax.plot(x, y_mean, marker='o')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax.set_title(title); ax.set_xlabel('d'); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
        if xscale_log: ax.set_xscale('log')

    def maybe_baseline(ax, res, key, label, ls='--'):
        if res is not None and ('metrics' in res) and (key in res['metrics']):
            ax.axhline(res['metrics'][key], ls=ls, alpha=0.7, label=label)

    # RMSE
    plot_band(ax_rmse, results_df['RMSE_mean'].values, results_df['RMSE_std'].values, 'RMSE vs d', 'RMSE')
    maybe_baseline(ax_rmse, res_exact, 'RMSE', 'exact')
    maybe_baseline(ax_rmse, res_svgp,  'RMSE', 'SVGP', ls=':')
    if len(ax_rmse.get_legend_handles_labels()[0]) > 0: ax_rmse.legend(frameon=False)

    # R2
    plot_band(ax_r2, results_df['R2_mean'].values, results_df['R2_std'].values, 'R2 vs d', 'R2')
    maybe_baseline(ax_r2, res_exact, 'R2', 'exact')
    maybe_baseline(ax_r2, res_svgp,  'R2', 'SVGP', ls=':')
    if len(ax_r2.get_legend_handles_labels()[0]) > 0: ax_r2.legend(frameon=False)

    # NLPD
    plot_band(ax_nlpd, results_df['NLPD_mean'].values, results_df['NLPD_std'].values, 'NLPD vs d', 'NLPD')
    maybe_baseline(ax_nlpd, res_exact, 'NLPD', 'exact')
    maybe_baseline(ax_nlpd, res_svgp,  'NLPD', 'SVGP', ls=':')
    if len(ax_nlpd.get_legend_handles_labels()[0]) > 0: ax_nlpd.legend(frameon=False)

    # time
    plot_band(ax_time, results_df['time_sec_mean'].values, results_df['time_sec_std'].values, 'time vs d', 'time (sec)')
    maybe_baseline(ax_time, res_exact, 'time_sec', 'exact', ls='--')
    maybe_baseline(ax_time, res_svgp,  'time_sec', 'SVGP', ls=':')
    if len(ax_time.get_legend_handles_labels()[0]) > 0: ax_time.legend(frameon=False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=220, bbox_inches='tight')
        print(f"saved plot to {savepath}")
    plt.show()

    # pick best d by RMSE
    best_d = results_df['RMSE_mean'].idxmin()
    print(f"\nbest d={best_d} " + 
        f"(RMSE={results_df.loc[best_d,'RMSE_mean']:.4f}+/-{results_df.loc[best_d,'RMSE_std']:.4f}, "
        f"R2={results_df.loc[best_d,'R2_mean']:.4f}+/-{results_df.loc[best_d,'R2_std']:.4f}, "
        f"NLPD={results_df.loc[best_d,'NLPD_mean']:.4f}+/-{results_df.loc[best_d,'NLPD_std']:.4f}, "
        f"time={results_df.loc[best_d,'time_sec_mean']:.2f}+/-{results_df.loc[best_d,'time_sec_std']:.2f}s)"
    )

    return results_df, results_by_d_first, runs_raw

def repeat_spherical_runs_for_d(
    x_train, y_train, x_test, y_test,
    d: int,
    k: int = 10,
    num_mixtures: int = 1,
    kernel_proj=None,               
    cfg: TrainCfg = TrainCfg(),
    init_hyps_data: bool = False,
    savepath_hyp: str | None = None,
    savepath_nlpd: str | None = None,
    res_exact: dict | None = None
):
    """run spherical projection objective 'k' times at fixed d"""
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))

    weights, scales, means, noises = [], [], [], []
    rmses, r2s, nlpds, times = [], [], [], []
    res_list = []

    for r in tqdm(range(k)):
        # Use provided kernel or create default SM kernel
        if kernel_proj is None:
            k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
        else:
            # Create a fresh copy of the provided kernel to avoid parameter sharing
            import copy
            k_proj = copy.deepcopy(kernel_proj)

        # change seed so ProjectionObjective gets different Omega
        cfg_r = TrainCfg(**{**vars(cfg), 'seed': cfg.seed + r})

        res = run_proj(
            x_train, y_train, x_test, y_test,
            kernel=k_proj, d_proj=int(d),
            cfg=cfg_r, init_from_data=init_hyps_data
        )
        res_list.append(res)

        # hyperparameters (handle different kernel types)
        params = res['params']
        if len(params) == 4:  # SM kernel: [weight, scale, mean, noise_var]
            w, s, m, nv = params
            if isinstance(w, np.ndarray):   # Q>1
                w_val = float(np.mean(w)); s_val = float(np.mean(s)); m_val = float(np.mean(m))
            else:                           # Q=1
                w_val = float(w); s_val = float(s); m_val = float(m)
            weights.append(w_val); scales.append(s_val); means.append(m_val); noises.append(float(nv))
        elif len(params) == 3:  # RBF/Matern: [lengthscale, outputscale, noise_var]
            ls, os, nv = params
            weights.append(float(ls)); scales.append(float(os)); means.append(0.0); noises.append(float(nv))
        elif len(params) == 4 and kernel_proj is not None and hasattr(kernel_proj.base_kernel, 'period_length'):  # Periodic: [lengthscale, period, outputscale, noise_var]
            ls, p, os, nv = params
            weights.append(float(ls)); scales.append(float(p)); means.append(float(os)); noises.append(float(nv))
        elif len(params) == 5 and kernel_proj is not None and hasattr(kernel_proj.base_kernel, 'kernels'):  # Composite: [lengthscale1, period, lengthscale2, outputscale, noise_var]
            ls1, p, ls2, os, nv = params
            weights.append(float(ls1)); scales.append(float(p)); means.append(float(ls2)); noises.append(float(os))
        else:  # Fallback
            weights.append(0.0); scales.append(0.0); means.append(0.0); noises.append(float(params[-1]))

        # metrics
        rmses.append(float(res['metrics']['RMSE']))
        r2s.append(float(res['metrics']['R2']))
        nlpds.append(float(res['metrics']['NLPD']))
        times.append(float(res['train_time']))

    # parameter names based on kernel
    if kernel_proj is None or hasattr(kernel_proj, 'mixture_weights'):
        param_names = ['weight', 'scale', 'mean', 'noise_var']
    elif hasattr(kernel_proj, 'base_kernel'):
        base_kernel = kernel_proj.base_kernel
        if hasattr(base_kernel, 'period_length'):
            param_names = ['lengthscale', 'period', 'outputscale', 'noise_var']
        elif hasattr(base_kernel, 'kernels'):
            # for composite kernels like periodic * rbf, we have: [lengthscale1, period, lengthscale2, outputscale, noise_var]
            param_names = ['lengthscale1', 'period', 'lengthscale2', 'outputscale', 'noise_var']
        else:
            param_names = ['lengthscale', 'outputscale', 'noise_var'] # rbf, matern

    # dataframes
    df_hyp = pd.DataFrame({
        param_names[0]: weights,
        param_names[1]: scales,
        param_names[2]: means,
        param_names[3]: noises
    })
    df_metric = pd.DataFrame({
        'RMSE': rmses,
        'R2': r2s,
        'NLPD': nlpds,
        'time_sec': times
    })

    # pre-compute exact GP scalars (if provided)
    exact_hyp_vals = None
    exact_nlpd_val = None
    if res_exact is not None:
        try:
            w, s, m, nv = res_exact['params']
            to_scalar = lambda x: float(np.mean(x)) if isinstance(x, np.ndarray) else float(x)
            exact_hyp_vals = [to_scalar(w), to_scalar(s), to_scalar(m), float(nv)]
        except Exception:
            exact_hyp_vals = None
        if 'metrics' in res_exact and 'NLPD' in res_exact['metrics']:
            exact_nlpd_val = float(res_exact['metrics']['NLPD'])

    # plot 1: hyperparameter boxplots (+ exact cross)
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(9, 4))
    plt.boxplot([df_hyp[param_names[0]], df_hyp[param_names[1]], df_hyp[param_names[2]], df_hyp[param_names[3]]],
                showfliers=False)
    plt.xticks([1, 2, 3, 4], param_names)
    plt.ylabel('value')
    plt.title(f'spherical GP hyperparameters over {k} runs (d={d})')
    plt.grid(alpha=0.25, axis='y')

    # overlay 'x' for exact GP (single point per param)
    if exact_hyp_vals is not None:
        x_positions = [1, 2, 3, 4]
        plt.scatter(x_positions, list(exact_hyp_vals), marker='x', s=70, linewidths=2, label='exact')

    plt.tight_layout()
    if savepath_hyp:
        plt.savefig(savepath_hyp, dpi=220, bbox_inches='tight')
    plt.show()

    # plot 2: metrics boxplots (RMSE, R2, NLPD, time) + exact 'x' overlays
    fig2 = plt.figure(figsize=(11, 4))
    metrics_for_box = [df_metric['RMSE'], df_metric['R2'], df_metric['NLPD'], df_metric['time_sec']]
    plt.boxplot(metrics_for_box, showfliers=False)
    plt.xticks([1, 2, 3, 4], ['RMSE', 'R2', 'NLPD', 'time_sec'])
    plt.ylabel('value')
    plt.title(f'metrics over {k} runs (d={d})')
    plt.grid(alpha=0.25, axis='y')

    # overlay 'x' for exact GP (single point per metric)
    if res_exact is not None and isinstance(res_exact, dict):
        x_positions = [1, 2, 3, 4]
        ex_rmse = res_exact.get('metrics', {}).get('RMSE', None)
        ex_r2   = res_exact.get('metrics', {}).get('R2', None)
        ex_nlpd = exact_nlpd_val if 'exact_nlpd_val' in locals() and exact_nlpd_val is not None \
                  else res_exact.get('metrics', {}).get('NLPD', None)
        ex_time = res_exact.get('train_time', None)
        ex_vals = [ex_rmse, ex_r2, ex_nlpd, ex_time]

        added_label = False
        for xp, val in zip(x_positions, ex_vals):
            if val is not None:
                plt.scatter([xp], [val], marker='x', s=70, linewidths=2,
                            label='exact' if not added_label else None)
                added_label = True
        if added_label:
            plt.legend(frameon=False)

    plt.tight_layout()
    if savepath_nlpd:
        plt.savefig(savepath_nlpd, dpi=220, bbox_inches='tight')
    plt.show()
    return df_hyp, df_metric, res_list



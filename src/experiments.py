import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy
from .config import TrainCfg, SVGPcfg
from .training import run_exact, run_proj, run_svgp, predict_svgp, run_proj_multiscale
from .utils import plot_posteriors_auto
from .kernels import SM_kernel
import matplotlib.pyplot as plt


def run_exact_proj_svgp(
    x_train, y_train, x_test, y_test, d_proj=200,
    num_mixtures=1, kernel_exact=None, kernel_proj=None, kernel_svgp=None,
    cfg: TrainCfg=TrainCfg(), svgp_cfg: SVGPcfg=SVGPcfg(),
    add_svgp=False, init_hyps_data=False,
    # plotting inputs:
    x_plot=None,
    x_grid=None, grid_shape=None, extent=None,
    savepath=None, add_svgp_heatmap=True,
    flag=True, verbose=True,
    use_multiscale=False, d_list=None,
    keep_exact=True,
    show_plots=True, show_dataframes=True
):
    """main experiment orchestrator"""
    from .utils import set_seed
    set_seed(cfg.seed)
    
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))

    # default SM kernels only if none provided
    if keep_exact:
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

    # exact GP (conditional)
    if keep_exact:
        res_exact = run_exact(x_train, y_train, x_test, y_test, k_exact, cfg, init_from_data=init_hyps_data, flag=flag)
    else:
        res_exact = None
    
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
    
    df_data = {
        method_name : [res_proj ['metrics']['RMSE'], res_proj ['metrics']['R2'], res_proj ['metrics']['NLPD'], res_proj ['metrics']['COV95'], res_proj ['metrics']['PIW95'], res_proj ['train_time']],
    }
    
    if keep_exact and res_exact is not None:
        df_data['exact'] = [res_exact['metrics']['RMSE'], res_exact['metrics']['R2'], res_exact['metrics']['NLPD'], res_exact['metrics']['COV95'], res_exact['metrics']['PIW95'], res_exact['train_time']]
    
    df = pd.DataFrame(df_data, index=['RMSE_test','R2_test','NLPD_test', 'COV95_test', 'PIW95_test', 'time_sec'])

    if add_svgp and res_svgp is not None:
        df['SVGP'] = [res_svgp['metrics']['RMSE'], res_svgp['metrics']['R2'], res_svgp['metrics']['NLPD'], res_svgp['metrics']['COV95'], res_svgp['metrics']['PIW95'], res_svgp['train_time']]

    if flag:
        proj_params = res_proj['params']
        
        # parameter names based on kernel (use projection kernel if exact is not available)
        kernel_for_params = kernel_exact if keep_exact and kernel_exact is not None else kernel_proj
        if kernel_for_params is None or hasattr(kernel_for_params, 'mixture_weights'):
            param_names = ['weight', 'scale', 'mean', 'noise_var']
        elif hasattr(kernel_for_params, 'base_kernel'):
            base_kernel = kernel_for_params.base_kernel
            if hasattr(base_kernel, 'period_length'):
                param_names = ['lengthscale', 'period', 'outputscale', 'noise_var']
            elif hasattr(base_kernel, 'kernels'):
                # for composite kernels like periodic * rbf, we have: [lengthscale1, period, lengthscale2, outputscale, noise_var]
                param_names = ['lengthscale1', 'period', 'lengthscale2', 'outputscale', 'noise_var']
            else:
                param_names = ['lengthscale', 'outputscale', 'noise_var'] # rbf, matern
        
        # build hyperparameter dataframe conditionally
        df_hyp_data = {f'GPyTorch ({method_name})' : proj_params}
        
        if keep_exact and res_exact is not None:
            exact_params = res_exact['params']
            df_hyp_data['GPyTorch (exact)'] = exact_params
        
        df_hyp = pd.DataFrame(df_hyp_data, index=param_names)
        
        if add_svgp:
            df_hyp['GPyTorch (SVGP)'] = res_svgp['params']

    if flag and show_dataframes:
        print(df_hyp)
    if show_dataframes:
        print(df)
    
    # plotting
    if show_plots:
        d_proj_for_plot = res_proj.get('total_d', d_proj) if use_multiscale else d_proj
        _ = plot_posteriors_auto(
            D=x_train.shape[1],
            res_exact=res_exact if keep_exact else None, res_proj=res_proj, res_svgp=res_svgp,
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
    xscale_log: bool = False,
    n_seeds: int = 5,
    base_seed: int = 0
):
    """sweep over d values and aggregate results across multiple seeds"""
    d_list = sorted(int(d) for d in d_list)

    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))

    # storage
    rows = []
    runs_raw = {}
    results_by_d_first = {}

    for i in tqdm(range(len(d_list)), desc="Sweeping d values"):
        d = d_list[i]
        metrics_runs = []  # each row: [RMSE, R2, NLPD, time_sec]
        res_first = None

        # run multiple seeds for each d value
        for seed_idx in range(n_seeds):
            # use provided kernel or create default sm kernel
            if kernel_proj is None:
                k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
            else:
                # create a fresh copy of the provided kernel to avoid parameter sharing
                import copy
                k_proj = copy.deepcopy(kernel_proj)

            # use widely spaced seeds to ensure different random states
            current_seed = base_seed + seed_idx * 1000 + i * 10000
            cfg_i = TrainCfg(**{**vars(cfg), 'seed': current_seed})

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

        metrics_runs = np.asarray(metrics_runs)  # shape (n_seeds, 4)
        runs_raw[d] = metrics_runs
        results_by_d_first[d] = res_first

        mean = metrics_runs.mean(axis=0)
        std  = metrics_runs.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)

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

    print(f"\nResults averaged over {n_seeds} seeds:")
    print(results_df)

    # plot: mean +/- std vs d with optional baselines
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_rmse, ax_r2, ax_nlpd, ax_time = axes.ravel()

    x = results_df.index.values

    def plot_band(ax, y_mean, y_std, title, ylabel, use_log_scale=False):
        ax.plot(x, y_mean, marker='o')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax.set_title(title); ax.set_xlabel('d'); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
        if xscale_log: ax.set_xscale('log')
        if use_log_scale: ax.set_yscale('log')

    def maybe_baseline(ax, res, key, label, ls='--'):
        if res is not None and ('metrics' in res) and (key in res['metrics']):
            ax.axhline(res['metrics'][key], ls=ls, alpha=0.7, label=label)

    # RMSE (log scale)
    plot_band(ax_rmse, results_df['RMSE_mean'].values, results_df['RMSE_std'].values, 'RMSE vs d', 'RMSE', use_log_scale=True)
    maybe_baseline(ax_rmse, res_exact, 'RMSE', 'exact')
    maybe_baseline(ax_rmse, res_svgp,  'RMSE', 'SVGP', ls=':')
    if len(ax_rmse.get_legend_handles_labels()[0]) > 0: ax_rmse.legend(frameon=False)

    # R2 (log scale)
    plot_band(ax_r2, results_df['R2_mean'].values, results_df['R2_std'].values, 'R2 vs d', 'R2', use_log_scale=False)
    maybe_baseline(ax_r2, res_exact, 'R2', 'exact')
    maybe_baseline(ax_r2, res_svgp,  'R2', 'SVGP', ls=':')
    if len(ax_r2.get_legend_handles_labels()[0]) > 0: ax_r2.legend(frameon=False)

    # NLPD (log scale)
    plot_band(ax_nlpd, results_df['NLPD_mean'].values, results_df['NLPD_std'].values, 'NLPD vs d', 'NLPD', use_log_scale=True)
    maybe_baseline(ax_nlpd, res_exact, 'NLPD', 'exact')
    maybe_baseline(ax_nlpd, res_svgp,  'NLPD', 'SVGP', ls=':')
    if len(ax_nlpd.get_legend_handles_labels()[0]) > 0: ax_nlpd.legend(frameon=False)

    # time (linear scale)
    plot_band(ax_time, results_df['time_sec_mean'].values, results_df['time_sec_std'].values, 'time vs d', 'time (sec)', use_log_scale=False)
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
    rmses, cov95s, nlpds, times = [], [], [], []
    res_list = []

    for r in tqdm(range(k)):
        # use provided kernel or create default sm kernel
        if kernel_proj is None:
            k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
        else:
            # create a fresh copy of the provided kernel to avoid parameter sharing
            import copy
            k_proj = copy.deepcopy(kernel_proj)

        # change seed so projectionobjective gets different omega
        cfg_r = TrainCfg(**{**vars(cfg), 'seed': cfg.seed + r})

        res = run_proj(
            x_train, y_train, x_test, y_test,
            kernel=k_proj, d_proj=int(d),
            cfg=cfg_r, init_from_data=init_hyps_data
        )
        res_list.append(res)

        # hyperparameters (handle different kernel types)
        params = res['params']
        if len(params) == 4:  # sm kernel: [weight, scale, mean, noise_var]
            w, s, m, nv = params
            if isinstance(w, np.ndarray):   # Q>1
                w_val = float(np.mean(w)); s_val = float(np.mean(s)); m_val = float(np.mean(m))
            else:                           # Q=1
                w_val = float(w); s_val = float(s); m_val = float(m)
            weights.append(w_val); scales.append(s_val); means.append(m_val); noises.append(float(nv))
        elif len(params) == 3:  # RBF/Matern: [lengthscale, outputscale, noise_var]
            ls, os, nv = params
            weights.append(float(ls)); scales.append(float(os)); means.append(0.0); noises.append(float(nv))
        elif len(params) == 4 and kernel_proj is not None and hasattr(kernel_proj.base_kernel, 'period_length'):  # periodic: [lengthscale, period, outputscale, noise_var]
            ls, p, os, nv = params
            weights.append(float(ls)); scales.append(float(p)); means.append(float(os)); noises.append(float(nv))
        elif len(params) == 5 and kernel_proj is not None and hasattr(kernel_proj.base_kernel, 'kernels'):  # composite: [lengthscale1, period, lengthscale2, outputscale, noise_var]
            ls1, p, ls2, os, nv = params
            weights.append(float(ls1)); scales.append(float(p)); means.append(float(ls2)); noises.append(float(os))
        else:  # fallback
            weights.append(0.0); scales.append(0.0); means.append(0.0); noises.append(float(params[-1]))

        # metrics
        rmses.append(float(res['metrics']['RMSE']))
        cov95s.append(float(res['metrics']['COV95'])/100.0)
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
        'COV95': cov95s,
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

    # plot 2: metrics boxplots (RMSE, COV95, NLPD, time) + exact 'x' overlays
    fig2 = plt.figure(figsize=(11, 4))
    metrics_for_box = [df_metric['RMSE'], df_metric['COV95'], df_metric['NLPD'], df_metric['time_sec']]
    plt.boxplot(metrics_for_box, showfliers=False)
    plt.xticks([1, 2, 3, 4], ['RMSE', 'COV95', 'NLPD', 'time_sec'])
    plt.ylabel('value')
    plt.title(f'metrics over {k} runs (d={d})')
    plt.grid(alpha=0.25, axis='y')

    # overlay 'x' for exact GP (single point per metric)
    if res_exact is not None and isinstance(res_exact, dict):
        x_positions = [1, 2, 3, 4]
        ex_rmse = res_exact.get('metrics', {}).get('RMSE', None)
        ex_cov95 = res_exact.get('metrics', {}).get('COV95', None)/100.0
        ex_nlpd = exact_nlpd_val if 'exact_nlpd_val' in locals() and exact_nlpd_val is not None \
                  else res_exact.get('metrics', {}).get('NLPD', None)
        ex_time = res_exact.get('train_time', None)
        ex_vals = [ex_rmse, ex_cov95, ex_nlpd, ex_time]

        added_label = False
        for xp, val in zip(x_positions, ex_vals):
            if val is not None:
                plt.scatter([xp], [val], marker='x', s=70, linewidths=2,
                            label='exact' if not added_label else None)
                added_label = True
        if added_label:
            plt.legend(frameon=False, loc='upper left')

    plt.tight_layout()
    if savepath_nlpd:
        plt.savefig(savepath_nlpd, dpi=220, bbox_inches='tight')
    plt.show()
    return df_hyp, df_metric, res_list


def run_multiple_seeds(
    x_train, y_train, x_test, y_test, 
    k: int = 10,
    d_proj: int = 200,
    num_mixtures: int = 1,
    kernel_exact=None, kernel_proj=None, kernel_svgp=None,
    cfg: TrainCfg = TrainCfg(), svgp_cfg: SVGPcfg = SVGPcfg(),
    add_svgp: bool = False, init_hyps_data: bool = False,
    use_multiscale: bool = False, d_list=None,
    keep_exact: bool = True,
    base_seed: int = 0
):
    """run run_exact_proj_svgp for k different seeds and report RMSE, NLPD and time for all runs"""
    from .utils import set_seed
    
    runs = []
    metrics_data = []
    
    for i in tqdm(range(k), desc=f"Running {k} seeds"):
        # set seed for this run
        current_seed = base_seed + i
        cfg_i = TrainCfg(**{**vars(cfg), 'seed': current_seed})
        
        # run the experiment with plots and dataframes disabled
        res_exact, res_proj, res_svgp = run_exact_proj_svgp(
            x_train, y_train, x_test, y_test, 
            d_proj=d_proj,
            num_mixtures=num_mixtures,
            kernel_exact=kernel_exact,
            kernel_proj=kernel_proj,
            kernel_svgp=kernel_svgp,
            cfg=cfg_i,
            svgp_cfg=svgp_cfg,
            add_svgp=add_svgp,
            init_hyps_data=init_hyps_data,
            use_multiscale=use_multiscale,
            d_list=d_list,
            keep_exact=keep_exact,
            show_plots=False,
            show_dataframes=False
        )
        
        # store run results
        run_result = {
            'seed': current_seed,
            'res_exact': res_exact,
            'res_proj': res_proj,
            'res_svgp': res_svgp
        }
        runs.append(run_result)
        
        # extract metrics
        method_name = 'multiscale' if use_multiscale else 'spherical'
        
        # projection metrics
        proj_rmse = res_proj['metrics']['RMSE']
        proj_nlpd = res_proj['metrics']['NLPD']
        proj_cov95 = res_proj['metrics']['COV95']
        proj_time = res_proj['train_time']
        
        # exact GP metrics (if available)
        exact_rmse = res_exact['metrics']['RMSE'] if res_exact is not None else None
        exact_nlpd = res_exact['metrics']['NLPD'] if res_exact is not None else None
        exact_cov95 = res_exact['metrics']['COV95'] if res_exact is not None else None
        exact_time = res_exact['train_time'] if res_exact is not None else None
        
        # svgp metrics (if available)
        svgp_rmse = res_svgp['metrics']['RMSE'] if res_svgp is not None else None
        svgp_nlpd = res_svgp['metrics']['NLPD'] if res_svgp is not None else None
        svgp_cov95 = res_svgp['metrics']['COV95'] if res_svgp is not None else None
        svgp_time = res_svgp['train_time'] if res_svgp is not None else None
        
        metrics_data.append({
            'seed': current_seed,
            f'{method_name}_RMSE': proj_rmse,
            f'{method_name}_NLPD': proj_nlpd,
            f'{method_name}_COV95': proj_cov95,
            f'{method_name}_time': proj_time,
            'exact_RMSE': exact_rmse,
            'exact_NLPD': exact_nlpd,
            'exact_COV95': exact_cov95,
            'exact_time': exact_time,
            'SVGP_RMSE': svgp_rmse,
            'SVGP_NLPD': svgp_nlpd,
            'SVGP_COV95': svgp_cov95,
            'SVGP_time': svgp_time
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # calculate summary statistics
    summary_stats = {}
    for col in metrics_df.columns:
        if col != 'seed':
            values = metrics_df[col].dropna()
            if len(values) > 0:
                summary_stats[f'{col}_mean'] = float(values.mean())
                summary_stats[f'{col}_std'] = float(values.std())
                summary_stats[f'{col}_min'] = float(values.min())
                summary_stats[f'{col}_max'] = float(values.max())
                summary_stats[f'{col}_median'] = float(values.median())
    
    print(f"\n=== Summary of {k} runs ===")
    print(f"Base seed: {base_seed}")
    print(f"Projection dimension: {d_proj}")
    print(f"Number of mixtures: {num_mixtures}")
    print(f"Multi-scale: {use_multiscale}")
    print(f"Exact GP included: {keep_exact}")
    print(f"SVGP included: {add_svgp}")
    
    print(f"\n=== Metrics Summary ===")
    for key, value in summary_stats.items():
        if 'mean' in key:
            metric_name = key.replace('_mean', '')
            std_key = key.replace('_mean', '_std')
            std_val = summary_stats.get(std_key, 0)
            print(f"{metric_name}: {value:.4f} +/- {std_val:.4f}")
    
    return {
        'runs': runs,
        'metrics_df': metrics_df,
        'summary_stats': summary_stats
    }


def run_d_m_sweep_with_frontier(
    x_train, y_train, x_test, y_test,
    d_values: list = [50, 100, 200, 400, 800],
    m_values: list = [50, 100, 200, 400, 800],
    num_mixtures: int = 1,
    kernel_exact=None, kernel_proj=None, kernel_svgp=None,
    cfg: TrainCfg = TrainCfg(), svgp_cfg: SVGPcfg = SVGPcfg(),
    init_hyps_data: bool = False,
    use_multiscale: bool = False,
    keep_exact: bool = True,  # Set to True to include exact GP in frontier
    base_seed: int = 0,
    plot_frontier: bool = True,
    frontier_savepath: str = None,
    n_seeds: int = 5
):
    """
    run experiments over multiple d values (projection) and m_inducing values (SVGP)
    creates a frontier plot comparing exact GP, projection GP, and SVGP
    Uses run_multiple_seeds to calculate median time and mean RMSE across multiple seeds
    """
    from .utils import set_seed
    
    # create deterministic kernels once to ensure exact gp is truly deterministic
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))
    
    # create deterministic kernels if not provided
    if keep_exact and kernel_exact is None:
        kernel_exact = SM_kernel(num_mixtures, D, y_std=y_std_train)
    if kernel_proj is None:
        kernel_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
    if kernel_svgp is None:
        kernel_svgp = SM_kernel(num_mixtures, D, y_std=y_std_train)
    
    # storage for results
    exact_results = []
    projection_results = []
    svgp_results = []
    
    # determine the number of runs based on the shorter list
    n_runs = min(len(d_values), len(m_values))
    
    print(f"Running {n_runs} combined experiments with {n_seeds} seeds each")
    print(f"d_values: {d_values[:n_runs]}, m_values: {m_values[:n_runs]}")
    print(f"Including exact GP: {keep_exact}")
    
    # run exact gp once if requested (since it's deterministic and independent of d/m)
    exact_rmse = None
    exact_time = None
    if keep_exact:
        print(f"\nRunning exact GP once (deterministic, independent of d/m values)...")
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        
        # run exact gp once
        res_exact, _, _ = run_exact_proj_svgp(
            x_train, y_train, x_test, y_test,
            d_proj=100,  # dummy value, not used for exact gp
            num_mixtures=num_mixtures,
            kernel_exact=kernel_exact,
            cfg=cfg,
            add_svgp=False,
            keep_exact=True,
            show_plots=False,
            show_dataframes=False
        )
        
        if res_exact is not None:
            exact_rmse = res_exact['metrics']['RMSE']
            exact_time = res_exact['train_time']
            print(f"Exact GP RMSE: {exact_rmse:.6f}, Time: {exact_time:.2f}s")
        else:
            print("Warning: Exact GP failed to run")
    
    # run combined experiments
    for run_idx in tqdm(range(n_runs), desc="Combined experiments"):
        d = d_values[run_idx]
        m = m_values[run_idx]
        
        # Configure SVGP for this run
        svgp_cfg_i = SVGPcfg(**{**vars(svgp_cfg), 'm_inducing': m})
        
        # Run projection and SVGP methods (skip exact GP since we already ran it)
        combined_results = run_multiple_seeds(
            x_train, y_train, x_test, y_test,
            k=n_seeds,
            d_proj=d,
            num_mixtures=num_mixtures,
            kernel_exact=kernel_exact,
            kernel_proj=kernel_proj,
            kernel_svgp=kernel_svgp,
            cfg=cfg,
            svgp_cfg=svgp_cfg_i,
            add_svgp=True,
            init_hyps_data=init_hyps_data,
            use_multiscale=use_multiscale,
            keep_exact=False,  # skip exact gp since we already ran it
            base_seed=base_seed + run_idx * 10000
        )
        
        # extract metrics from summary statistics
        summary_stats = combined_results['summary_stats']
        method_name = 'multiscale' if use_multiscale else 'spherical'
        
        # use the single exact gp result for all runs
        if keep_exact and exact_rmse is not None and exact_time is not None:
            exact_results.append({
                'rmse': exact_rmse,
                'time': exact_time
            })
        
        # extract projection metrics
        proj_mean_rmse = summary_stats.get(f'{method_name}_RMSE_mean', 0.0)
        proj_median_time = summary_stats.get(f'{method_name}_time_median', 0.0)
        if proj_median_time == 0.0:
            proj_median_time = summary_stats.get(f'{method_name}_time_mean', 0.0)
        
        # extract svgp metrics
        svgp_mean_rmse = summary_stats.get('SVGP_RMSE_mean', 0.0)
        svgp_median_time = summary_stats.get('SVGP_time_median', 0.0)
        if svgp_median_time == 0.0:
            svgp_median_time = summary_stats.get('SVGP_time_mean', 0.0)
        
        # store results
        projection_results.append({
            'd': d,
            'rmse': proj_mean_rmse,
            'time': proj_median_time
        })
        
        svgp_results.append({
            'm': m,
            'rmse': svgp_mean_rmse,
            'time': svgp_median_time
        })
    
    # create dataframes
    proj_df = pd.DataFrame(projection_results)
    svgp_df = pd.DataFrame(svgp_results)
    
    print("\n=== Projection Results (Mean RMSE, Median Time) ===")
    print(proj_df)
    print("\n=== SVGP Results (Mean RMSE, Median Time) ===")
    print(svgp_df)
    
    if keep_exact and exact_results:
        exact_df = pd.DataFrame(exact_results)
        print("\n=== Exact GP Results (Single run, deterministic) ===")
        print(exact_df)
    else:
        exact_df = None
    
    # create frontier plot if requested
    if plot_frontier:
        plot_d_m_frontier(proj_df, svgp_df, exact_df, frontier_savepath)
    
    return {
        'exact_results': exact_df,
        'projection_results': proj_df,
        'svgp_results': svgp_df
    }


def plot_d_m_frontier(proj_df, svgp_df, exact_df=None, savepath=None, title="GP Methods Frontier"):
    """create a frontier plot comparing exact GP, projection (d values) vs SVGP (m values)"""
    # create the plot
    plt.figure(figsize=(12, 8))
    
    # plot exact gp results (if available)
    if exact_df is not None and len(exact_df) > 0:
        # use the first row for exact gp (same across all d/m values)
        exact_rmse = exact_df.iloc[0]['rmse']
        exact_time = exact_df.iloc[0]['time']
        plt.plot(exact_time, exact_rmse, 
                'D', color='green', markersize=12, linewidth=2, label='Exact GP', alpha=0.8)
        plt.annotate('Exact GP', 
                    (exact_time, exact_rmse),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, alpha=0.9, color='green', weight='bold')
    
    # plot projection results
    plt.plot(proj_df['time'], proj_df['rmse'], 
            'o-', color='blue', markersize=8, linewidth=2, label='Projection')
    
    # plot svgp results
    plt.plot(svgp_df['time'], svgp_df['rmse'], 
            's-', color='red', markersize=8, linewidth=2, label='SVGP')
    
    # add annotations for d and m values
    for _, row in proj_df.iterrows():
        plt.annotate(f'd={row["d"]}', 
                    (row['time'], row['rmse']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, alpha=0.8, color='blue')
    
    for _, row in svgp_df.iterrows():
        plt.annotate(f'm={row["m"]}', 
                    (row['time'], row['rmse']),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=9, alpha=0.8, color='red')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Frontier plot saved to {savepath}")
    
    plt.show()


def run_reliability_analysis(
    x_train, y_train, x_test, y_test,
    d_proj=200, num_mixtures=1, kernel_proj=None,
    cfg=None, n_seeds=5, confidence_levels=None,
    savepath=None, figsize=(8, 6)
):
    """run reliability analysis for projected GP"""
    from .utils.plotting import plot_reliability_projected_gp
    
    return plot_reliability_projected_gp(
        x_train, y_train, x_test, y_test,
        d_proj=d_proj, num_mixtures=num_mixtures, kernel_proj=kernel_proj,
        cfg=cfg, n_seeds=n_seeds, confidence_levels=confidence_levels,
        savepath=savepath, figsize=figsize
    )


def run_sweep_d_multiple_seeds(
    x_train, y_train, x_test, y_test,
    d_list,
    n_seeds: int = 5,
    num_mixtures: int = 1,
    kernel_proj=None,
    cfg: TrainCfg = TrainCfg(),
    init_hyps_data: bool = False,
    res_exact: dict | None = None,
    res_svgp: dict | None = None,
    savepath: str | None = None,
    xscale_log: bool = False,
    base_seed: int = 0
):
    """
    Efficiently sweep over d values using run_multiple_seeds for each d value.
    This provides better integration with the existing multi-seed infrastructure.
    """
    d_list = sorted(int(d) for d in d_list)

    # storage
    rows = []
    runs_raw = {}
    results_by_d_first = {}

    for i, d in enumerate(tqdm(d_list, desc="Sweeping d values")):
        # use run_multiple_seeds for efficient multi-seed testing
        results = run_multiple_seeds(
            x_train, y_train, x_test, y_test,
            k=n_seeds,
            d_proj=d,
            num_mixtures=num_mixtures,
            kernel_proj=kernel_proj,
            cfg=cfg,
            init_hyps_data=init_hyps_data,
            base_seed=base_seed + i * 10000,  # different base seed for each d
            add_svgp=False,
            keep_exact=False
        )
        
        # extract projection metrics from summary_stats
        method_name = 'spherical'
        summary_stats = results['summary_stats']
        
        # get mean and std for each metric
        rmse_mean = summary_stats.get(f'{method_name}_RMSE_mean', 0.0)
        rmse_std = summary_stats.get(f'{method_name}_RMSE_std', 0.0)
        cov95_mean = summary_stats.get(f'{method_name}_COV95_mean', 0.0)
        cov95_std = summary_stats.get(f'{method_name}_COV95_std', 0.0)
        nlpd_mean = summary_stats.get(f'{method_name}_NLPD_mean', 0.0)
        nlpd_std = summary_stats.get(f'{method_name}_NLPD_std', 0.0)
        time_mean = summary_stats.get(f'{method_name}_time_mean', 0.0)
        time_std = summary_stats.get(f'{method_name}_time_std', 0.0)
        
        # store results
        rows.append([
            d, rmse_mean, rmse_std,
            cov95_mean, cov95_std,
            nlpd_mean, nlpd_std,
            time_mean, time_std
        ])
        
        # store raw data for this d value
        runs_raw[d] = results
        
        # store first result for this d (for compatibility)
        if i == 0:
            # placeholder for compatibility
            results_by_d_first[d] = None

    results_df = pd.DataFrame(
        rows,
        columns=[
            'd',
            'RMSE_mean','RMSE_std',
            'COV95_mean','COV95_std',
            'NLPD_mean','NLPD_std',
            'time_sec_mean','time_sec_std'
        ]
    ).set_index('d')

    print(f"\nResults averaged over {n_seeds} seeds:")
    print(results_df)

    # plot: mean +/- std vs d with optional baselines
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_rmse, ax_cov95, ax_nlpd, ax_time = axes.ravel()

    x = results_df.index.values

    def plot_band(ax, y_mean, y_std, title, ylabel, use_log_scale=False):
        ax.plot(x, y_mean, marker='o')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        ax.set_title(title); ax.set_xlabel('d'); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
        if xscale_log: ax.set_xscale('log')
        if use_log_scale: ax.set_yscale('log')

    def maybe_baseline(ax, res, key, label, ls='--'):
        if res is not None and ('metrics' in res) and (key in res['metrics']):
            ax.axhline(res['metrics'][key], ls=ls, alpha=0.7, label=label)

    # RMSE (log scale)
    plot_band(ax_rmse, results_df['RMSE_mean'].values, results_df['RMSE_std'].values, 'RMSE vs d', 'RMSE', use_log_scale=True)
    maybe_baseline(ax_rmse, res_exact, 'RMSE', 'exact')
    maybe_baseline(ax_rmse, res_svgp,  'RMSE', 'SVGP', ls=':')
    if len(ax_rmse.get_legend_handles_labels()[0]) > 0: ax_rmse.legend(frameon=False)

    # COV95 (log scale)
    plot_band(ax_cov95, results_df['COV95_mean'].values, results_df['COV95_std'].values, 'COV95 vs d', 'COV95', use_log_scale=True)
    maybe_baseline(ax_cov95, res_exact, 'COV95', 'exact')
    maybe_baseline(ax_cov95, res_svgp,  'COV95', 'SVGP', ls=':')
    if len(ax_cov95.get_legend_handles_labels()[0]) > 0: ax_cov95.legend(frameon=False)

    # NLPD (log scale)
    plot_band(ax_nlpd, results_df['NLPD_mean'].values, results_df['NLPD_std'].values, 'NLPD vs d', 'NLPD', use_log_scale=True)
    maybe_baseline(ax_nlpd, res_exact, 'NLPD', 'exact')
    maybe_baseline(ax_nlpd, res_svgp,  'NLPD', 'SVGP', ls=':')
    if len(ax_nlpd.get_legend_handles_labels()[0]) > 0: ax_nlpd.legend(frameon=False)

    # time (linear scale)
    plot_band(ax_time, results_df['time_sec_mean'].values, results_df['time_sec_std'].values, 'time vs d', 'time (sec)', use_log_scale=False)
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
        f"COV95={results_df.loc[best_d,'COV95_mean']:.4f}+/-{results_df.loc[best_d,'COV95_std']:.4f}, "
        f"NLPD={results_df.loc[best_d,'NLPD_mean']:.4f}+/-{results_df.loc[best_d,'NLPD_std']:.4f}, "
        f"time={results_df.loc[best_d,'time_sec_mean']:.2f}+/-{results_df.loc[best_d,'time_sec_std']:.2f}s)"
    )

    return results_df, results_by_d_first, runs_raw



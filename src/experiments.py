import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from .config import TrainCfg, SVGPcfg
from .training import run_exact, run_proj, run_svgp, predict_svgp
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
    flag=True
):
    """main experiment orchestrator"""
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))

    if kernel_exact is not None:
        flag = False
    else:
        k_exact = SM_kernel(num_mixtures, D, y_std=y_std_train)
        k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)

    if add_svgp:
        k_svgp = SM_kernel(num_mixtures, D, y_std=y_std_train)
    torch.manual_seed(cfg.seed)

    res_exact = run_exact(x_train, y_train, x_test, y_test, k_exact, cfg, init_from_data=init_hyps_data, flag=flag)
    res_proj = run_proj(x_train, y_train, x_test, y_test, k_proj, d_proj, cfg, init_from_data=init_hyps_data, flag=flag)
    res_svgp  = None
    if add_svgp:
        res_svgp = run_svgp(x_train, y_train, x_test, y_test, k_svgp, svgp_cfg, device=device, flag=flag)

    # results table(s)
    df = pd.DataFrame({
        'exact'     : [res_exact['metrics']['RMSE'], res_exact['metrics']['R2'], res_exact['metrics']['NLPD'], res_exact['metrics']['COV95'], res_exact['metrics']['PIW95'], res_exact['train_time']],
        'spherical' : [res_proj ['metrics']['RMSE'], res_proj ['metrics']['R2'], res_proj ['metrics']['NLPD'], res_proj ['metrics']['COV95'], res_proj ['metrics']['PIW95'], res_proj ['train_time']],
    }, index=['RMSE_test','R2_test','NLPD_test', 'COV95_test', 'PIW95_test', 'time_sec'])

    if add_svgp and res_svgp is not None:
        df['SVGP'] = [res_svgp['metrics']['RMSE'], res_svgp['metrics']['R2'], res_svgp['metrics']['NLPD'], res_svgp['metrics']['COV95'], res_svgp['metrics']['PIW95'], res_svgp['train_time']]

    if flag:
        df_hyp = pd.DataFrame({
            'GPyTorch (exact)': res_exact['params'],
            'GPyTorch (proj)' : res_proj['params'],
        }, index=['weight', 'scale', 'mean', 'noise_var'])
        if add_svgp:
            df_hyp['GPyTorch (SVGP)'] = res_svgp['params']

    if flag:
        print(df_hyp)
    print(df)
    
    # plotting
    _ = plot_posteriors_auto(
    D=x_train.shape[1],
    res_exact=res_exact, res_proj=res_proj, res_svgp=res_svgp,
    x_test=x_test, y_test=y_test, x_plot=x_plot,
    x_train=x_train, x_grid=x_grid, grid_shape=grid_shape, extent=extent,
    d_proj=d_proj,
    predict_svgp_fn=predict_svgp,
    plot1d_fn=None,
    add_svgp_heatmap=add_svgp_heatmap,
    savepath=savepath 
    )

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
            # fresh kernel each repeat to avoid parameter carry-over
            k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)

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

    # build tidy DF
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
    plot_band(ax_time, results_df['time_sec_mean'].values, results_df['time_sec_std'].values, 'train time vs d', 'seconds')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=220, bbox_inches='tight')
    plt.show()

    # quick summary
    best_d = int(results_df['RMSE_mean'].idxmin())
    print(
        f"[sweep-d agg] best d by RMSE_mean: {best_d} "
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
        # fresh kernel each repeat (avoid parameter carry-over)
        k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)

        # change seed so ProjectionObjective gets different Omega
        cfg_r = TrainCfg(**{**vars(cfg), 'seed': cfg.seed + r})

        res = run_proj(
            x_train, y_train, x_test, y_test,
            kernel=k_proj, d_proj=int(d),
            cfg=cfg_r, init_from_data=init_hyps_data
        )
        res_list.append(res)

        # hyperparameters (handle Q=1 vs Q>1 by simple averaging for plotting)
        w, s, m, nv = res['params']
        if isinstance(w, np.ndarray):   # Q>1
            w_val = float(np.mean(w)); s_val = float(np.mean(s)); m_val = float(np.mean(m))
        else:                           # Q=1
            w_val = float(w); s_val = float(s); m_val = float(m)
        weights.append(w_val); scales.append(s_val); means.append(m_val); noises.append(float(nv))

        # metrics
        rmses.append(float(res['metrics']['RMSE']))
        r2s.append(float(res['metrics']['R2']))
        nlpds.append(float(res['metrics']['NLPD']))
        times.append(float(res['train_time']))

    # dataframes
    df_hyp = pd.DataFrame({
        'weight': weights,
        'scale': scales,
        'mean': means,
        'noise_var': noises
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
            exact_hyp_vals = _scalarize_params(res_exact['params'])
        except Exception:
            exact_hyp_vals = None
        if 'metrics' in res_exact and 'NLPD' in res_exact['metrics']:
            exact_nlpd_val = float(res_exact['metrics']['NLPD'])

    # plot 1: hyperparameter boxplots (+ exact cross)
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(9, 4))
    plt.boxplot([df_hyp['weight'], df_hyp['scale'], df_hyp['mean'], df_hyp['noise_var']],
                showfliers=False)
    plt.xticks([1, 2, 3, 4], ['weight', 'scale', 'mean', 'noise_var'])
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

def _scalarize_params(params):
    """params is [w, s, m, noise_var]; each of w/s/m may be scalar or np.ndarray (Q>1)"""
    w, s, m, nv = params
    to_scalar = lambda x: float(np.mean(x)) if isinstance(x, np.ndarray) else float(x)
    return to_scalar(w), to_scalar(s), to_scalar(m), float(nv)

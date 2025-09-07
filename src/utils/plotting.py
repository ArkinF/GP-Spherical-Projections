import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from scipy.stats import norm

def plot_posterior(x_train, y_train, x_test, y_mean, y_std, title='posterior gp'):
    """posterior plot"""
    plt.figure(figsize=(10, 4))
    plt.plot(x_train, y_train, 'r.', label='train data')
    plt.plot(x_test, y_mean, 'b', label='posterior mean')
    plt.fill_between(x_test.ravel(), y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.3, color='blue', label='95% CI')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_posteriors_1d(
    res_exact, res_proj, res_svgp,
    x_plot, x_test, y_test,
    d_proj=None,
    predict_svgp_fn=None,
    plot_fn=None,
    savepath=None
):
    """plot 1d posteriors"""
    # device from model params - use proj if exact is None
    device = next(res_proj['model'].parameters()).device
    x_plot = np.asarray(x_plot).reshape(-1)
    x_plot_tensor = torch.as_tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # exact (optional)
        mu_e = sd_e = None
        if res_exact is not None:
            pe = res_exact['likelihood'](res_exact['model'](x_plot_tensor))
            mu_e = pe.mean.detach().cpu().numpy()
            sd_e = pe.variance.detach().cpu().numpy() ** 0.5

        # spherical
        pp = res_proj['likelihood'](res_proj['model'](x_plot_tensor))
        mu_p = pp.mean.detach().cpu().numpy()
        sd_p = pp.variance.detach().cpu().numpy() ** 0.5

        # svgp (optional)
        mu_s = sd_s = None
        if (res_svgp is not None) and (predict_svgp_fn is not None):
            mu_s_flat, var_s_flat = predict_svgp_fn(
                res_svgp['model'], res_svgp['likelihood'], x_plot_tensor, device=device
            )
            mu_s = mu_s_flat.detach().cpu().numpy()
            sd_s = (var_s_flat.detach().cpu().numpy()) ** 0.5

    # sort for fill_between safety
    order = np.argsort(x_plot)
    xs = x_plot[order]

    def _fallback_plot(x_te, y_te, xx, mu, sd, title):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x_te, y_te, s=10, alpha=0.3, label="test")
        ax.plot(xx, mu, lw=2, label="mean")
        ax.fill_between(xx, mu - 2*sd, mu + 2*sd, alpha=0.2, label="+/-2sigma")
        ax.set_title(title); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        ax.legend()
        if savepath: plt.savefig(savepath, dpi=220)
        plt.show()

    # choose plot function
    _plot = plot_fn if plot_fn is not None else _fallback_plot

    x_te_np = x_test.detach().cpu().numpy().reshape(-1)
    y_te_np = y_test.detach().cpu().numpy().reshape(-1)

    if mu_e is not None:
        _plot(x_te_np, y_te_np, xs, mu_e[order], sd_e[order], title="posterior (exact GP)")
    title_proj = f"posterior (projection GP{'' if d_proj is None else f', d={d_proj}'})"
    _plot(x_te_np, y_te_np, xs, mu_p[order], sd_p[order], title=title_proj)

    if mu_s is not None:
        _plot(x_te_np, y_te_np, xs, mu_s[order], sd_s[order], title="posterior (SVGP)")

    return dict(x_plot=xs, 
                mu_exact=(mu_e[order] if mu_e is not None else None), 
                sd_exact=(sd_e[order] if sd_e is not None else None),
                mu_proj=mu_p[order], sd_proj=sd_p[order],
                mu_svgp=(mu_s[order] if mu_s is not None else None),
                sd_svgp=(sd_s[order] if sd_s is not None else None))

def plot_posteriors_1d_stacked(
    res_exact, res_proj, res_svgp,
    x_plot, x_test, y_test,
    d_proj=None,
    predict_svgp_fn=None,
    plot_fn=None,
    savepath=None
):
    """stacked 1d posterior plots"""
    # prepare line inputs - use proj device if exact is None
    device = next(res_proj['model'].parameters()).device
    x_plot = np.asarray(x_plot).reshape(-1)
    order  = np.argsort(x_plot)
    xs     = x_plot[order]
    x_line = torch.as_tensor(xs, dtype=torch.float32, device=device).unsqueeze(-1)

    # predict on x_line
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mu_e = sd_e = None
        if res_exact is not None:
            pe = res_exact['likelihood'](res_exact['model'](x_line))
            mu_e = pe.mean.detach().cpu().numpy()
            sd_e = pe.variance.detach().cpu().numpy()**0.5

        pp = res_proj['likelihood'](res_proj['model'](x_line))
        mu_p = pp.mean.detach().cpu().numpy()
        sd_p = pp.variance.detach().cpu().numpy()**0.5

        mu_s = sd_s = None
        have_svgp = (res_svgp is not None) and (predict_svgp_fn is not None)
        if have_svgp:
            mu_s_flat, var_s_flat = predict_svgp_fn(
                res_svgp['model'], res_svgp['likelihood'], x_line, device=device
            )
            mu_s = mu_s_flat.detach().cpu().numpy()
            sd_s = (var_s_flat.detach().cpu().numpy())**0.5

    # observations for scatter (use test for consistency)
    x_te = x_test.detach().cpu().numpy().reshape(-1)
    y_te = y_test.detach().cpu().numpy().reshape(-1)

    # plot stacked - adjust nrows based on what we have
    have_exact = mu_e is not None
    nrows = (1 if have_exact else 0) + 1 + (1 if mu_s is not None else 0)  # exact + proj + svgp
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.1*nrows), sharex=True)
    
    if nrows == 1:
        axes = [axes]

    def _one(ax, title, mu, sd):
        ax.scatter(x_te, y_te, color='r', marker='.', label="test")
        ax.plot(xs, mu, 'b', label="mean")
        ax.fill_between(xs, mu - 2*sd, mu + 2*sd, alpha=0.3, color='blue', label="+/-2sigma")
        ax.set_title(title)
        ax.set_ylabel("$y$")
        ax.grid(alpha=0.2)

    ax_idx = 0
    first_ax = None
    
    if have_exact:
        _one(axes[ax_idx], "exact GP", mu_e, sd_e)
        first_ax = axes[ax_idx]
        ax_idx += 1
    
    _one(axes[ax_idx], f"spherical GP{'' if d_proj is None else f' (d={d_proj})'}", mu_p, sd_p)
    if first_ax is None:
        first_ax = axes[ax_idx]
    ax_idx += 1
    
    if mu_s is not None:
        _one(axes[ax_idx], "SVGP", mu_s, sd_s)

    axes[-1].set_xlabel("$x$")
    # legend once (top)
    handles, labels = first_ax.get_legend_handles_labels()
    first_ax.legend(handles, labels)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=220)
    plt.show()

    # return the arrays in case you want to reuse them
    return dict(x_plot=xs, 
                mu_exact=mu_e, sd_exact=sd_e,
                mu_proj=mu_p,  sd_proj=sd_p,
                mu_svgp=mu_s,  sd_svgp=sd_s)

def plot_posteriors_2d(
    res_exact, res_proj, res_svgp,
    x_train, x_grid, grid_shape, extent,
    d_proj=None,
    predict_svgp_fn=None,
    add_svgp_heatmap=True,
    diff_against="proj",
    savepath=None
):
    """plot 2d posteriors"""
    # use proj device if exact is None
    device = next(res_proj['model'].parameters()).device
    xg = x_grid.to(device=device, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # exact and spherical means on grid
        mu_e = None
        if res_exact is not None:
            pe = res_exact['likelihood'](res_exact['model'](xg))
            mu_e = pe.mean.view(*grid_shape).detach().cpu().numpy()
            
        pp = res_proj ['likelihood'](res_proj ['model'](xg))
        mu_p = pp.mean.view(*grid_shape).detach().cpu().numpy()

        mu_s = None
        if add_svgp_heatmap and (res_svgp is not None) and (predict_svgp_fn is not None):
            mu_s_flat, _ = predict_svgp_fn(res_svgp['model'], res_svgp['likelihood'], xg, device=device)
            mu_s = mu_s_flat.view(*grid_shape).detach().cpu().numpy()

    xmin, xmax, ymin, ymax = extent
    
    # determine number of columns based on what we have
    have_exact = mu_e is not None
    have_svgp = add_svgp_heatmap and (mu_s is not None)
    
    ncols = (1 if have_exact else 0) + 1 + (1 if have_svgp else 0) + 1  # exact + proj + svgp + diff
    figwidth = 4.8 * ncols + 1  # adjust width based on number of columns
    fig, axs = plt.subplots(1, ncols, figsize=(figwidth, 4.8), constrained_layout=True)
    
    if ncols == 1:
        axs = [axs]

    col_idx = 0
    
    if have_exact:
        im0 = axs[col_idx].imshow(mu_e, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
        axs[col_idx].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
        axs[col_idx].set_title("exact: mean"); axs[col_idx].set_xlabel("$x_1$"); axs[col_idx].set_ylabel("$x_2$")
        fig.colorbar(im0, ax=axs[col_idx], fraction=0.046)
        col_idx += 1

    im1 = axs[col_idx].imshow(mu_p, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
    axs[col_idx].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
    axs[col_idx].set_title(f"spherical (d={d_proj})" if d_proj is not None else "spherical"); axs[col_idx].set_xlabel("$x_1$"); axs[col_idx].set_ylabel("$x_2$")
    fig.colorbar(im1, ax=axs[col_idx], fraction=0.046)
    col_idx += 1
    
    if have_svgp:
        im2 = axs[col_idx].imshow(mu_s, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
        axs[col_idx].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
        axs[col_idx].set_title("SVGP: mean"); axs[col_idx].set_xlabel("$x_1$"); axs[col_idx].set_ylabel("$x_2$")
        fig.colorbar(im2, ax=axs[col_idx], fraction=0.046)
        col_idx += 1

    # mean difference vs exact (only if we have exact)
    if have_exact:
        if diff_against == "proj":
            diff = np.abs(mu_p - mu_e)
            diff_title = "|mean diff| (proj - exact)"
        else:
            if mu_s is None:
                diff = np.abs(mu_p - mu_e)
                diff_title = "|mean diff| (proj - exact)"
            else:
                diff = np.abs(mu_s - mu_e)
                diff_title = "|mean diff| (SVGP - exact)"

        imd = axs[col_idx].imshow(diff, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
        axs[col_idx].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
        axs[col_idx].set_title(diff_title); axs[col_idx].set_xlabel("$x_1$"); axs[col_idx].set_ylabel("$x_2$")
        fig.colorbar(imd, ax=axs[col_idx], fraction=0.046)
    else:
        # no exact gp, so no meaningful difference to plot
        diff = None

    if savepath:
        plt.savefig(savepath, dpi=220)
    plt.show()

    return dict(mu_exact=mu_e, mu_proj=mu_p, mu_svgp=mu_s, diff=diff)

def plot_posteriors_auto(
    D,
    res_exact, res_proj, res_svgp,
    x_test=None, y_test=None, x_plot=None,
    x_train=None, x_grid=None, grid_shape=None, extent=None,
    d_proj=None,
    predict_svgp_fn=None, plot1d_fn=None,
    add_svgp_heatmap=True,
    savepath=None
):
    """auto-detect 1d vs 2d plotting"""
    if D == 1 and x_plot is not None and x_test is not None and y_test is not None:
        return plot_posteriors_1d_stacked(
            res_exact, res_proj, res_svgp,
            x_plot=x_plot, x_test=x_test, y_test=y_test,
            d_proj=d_proj, predict_svgp_fn=predict_svgp_fn,
            plot_fn=plot1d_fn, savepath=savepath
        )
    elif D == 2 and (x_train is not None) and (x_grid is not None) and (grid_shape is not None) and (extent is not None):
        return plot_posteriors_2d(
            res_exact, res_proj, res_svgp,
            x_train=x_train, x_grid=x_grid, grid_shape=grid_shape, extent=extent,
            d_proj=d_proj, predict_svgp_fn=predict_svgp_fn,
            add_svgp_heatmap=add_svgp_heatmap, savepath=savepath
        )
    else:
        print("no plotting")
        return None

def plot_reliability_projected_gp(
    x_train, y_train, x_test, y_test,
    d_proj=200, num_mixtures=2, kernel_proj=None,
    cfg=None, n_seeds=5, confidence_levels=None,
    savepath=None, figsize=(8, 6)
):
    """reliability plot for projected gp"""
    from ..config import TrainCfg
    from ..kernels import SM_kernel
    from ..experiments import run_exact_proj_svgp
    from . import set_seed
    import copy
    
    if cfg is None:
        cfg = TrainCfg()
    
    if confidence_levels is None:
        confidence_levels = np.linspace(0.05, 0.95, 19)  # 0.05, 0.1, ..., 0.95
    
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))
    
    # storage for empirical coverages across seeds
    empirical_coverages = []
    
    print(f"Running reliability analysis with {n_seeds} seeds")
    
    for seed in range(n_seeds):

        current_seed = cfg.seed + seed * 1000  # use large gaps between seeds
        
        # set seed for reproducibility
        set_seed(current_seed)
        
        # create kernel
        if kernel_proj is None:
            k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
        else:
            k_proj = copy.deepcopy(kernel_proj)
        
        # create a new config with the updated seed to ensure projectionobjective gets different seed
        cfg_seed = TrainCfg(**{**vars(cfg), 'seed': current_seed})
        
        # run projected gp using run_exact_proj_svgp with keep_exact=false and add_svgp=false
        _, res_proj, _ = run_exact_proj_svgp(
            x_train, y_train, x_test, y_test,
            d_proj=d_proj,
            num_mixtures=num_mixtures,
            kernel_proj=k_proj,
            cfg=cfg_seed,
            keep_exact=False,
            add_svgp=False,
            show_plots=False,
            show_dataframes=False
        )
        
        # get predictions on test set
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = res_proj['likelihood'](res_proj['model'](x_test))
            mu = pred_dist.mean
            var = pred_dist.variance
        
        # convert to numpy
        mu_np = mu.detach().cpu().numpy().ravel()
        var_np = var.detach().cpu().numpy().ravel()
        y_test_np = y_test.detach().cpu().numpy().ravel()
        
        # compute empirical coverage for each confidence level
        seed_coverages = []
        for coverage_level in confidence_levels:
            # get z-score for this coverage level (coverage_level is the desired coverage, e.g., 0.95 for 95%)
            # for a two-sided interval with coverage_level coverage, we need z-score for (1 + coverage_level)/2
            z_alpha = norm.ppf((1 + coverage_level) / 2)
            
            # compute prediction intervals
            lower = mu_np - z_alpha * np.sqrt(var_np)
            upper = mu_np + z_alpha * np.sqrt(var_np)
            
            # compute empirical coverage
            coverage = np.mean((y_test_np >= lower) & (y_test_np <= upper))
            seed_coverages.append(coverage)
        
        empirical_coverages.append(seed_coverages)
    
    # average across seeds
    empirical_coverages = np.array(empirical_coverages)
    mean_coverages = np.mean(empirical_coverages, axis=0)
    std_coverages = np.std(empirical_coverages, axis=0)
    
    # create the reliability plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot diagonal line (perfect calibration)
    nominal_levels = confidence_levels  # confidence_levels are already coverage levels
    ax.plot(nominal_levels, nominal_levels, 'k--', alpha=0.7, label='Perfect calibration')
    
    # plot empirical coverage with error bars
    ax.errorbar(nominal_levels, mean_coverages, yerr=std_coverages, 
                fmt='o-', capsize=3, capthick=1, elinewidth=1,
                label=f'Projected GP (d={d_proj})', markersize=6)
    
    # add confidence band (shaded area)
    ax.fill_between(nominal_levels, 
                    mean_coverages - std_coverages,
                    mean_coverages + std_coverages,
                    alpha=0.2)
    
    # customise plot
    ax.set_xlabel('Nominal coverage level')
    ax.set_ylabel('Empirical coverage level')
    ax.set_title(f'Reliability Plot - Projected GP (d={d_proj}, {n_seeds} seeds)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # add text with summary statistics
    # calculate calibration error (mean absolute difference from diagonal)
    calibration_error = np.mean(np.abs(mean_coverages - nominal_levels))
    ax.text(0.05, 0.95, f'Calibration Error: {calibration_error:.3f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=220, bbox_inches='tight')
    
    plt.show()
    
    # return the data for further analysis
    return {
        'nominal_levels': nominal_levels,
        'mean_coverages': mean_coverages,
        'std_coverages': std_coverages,
        'calibration_error': calibration_error,
        'empirical_coverages': empirical_coverages
    }


def plot_resampling_policy_comparison(
    x_train, y_train, x_test, y_test,
    d_values=[50, 300],
    resample_policies=[None, 5, 20],
    n_seeds=5,
    num_mixtures=1,
    kernel_proj=None,
    cfg=None,
    savepath=None,
    figsize=(10, 6)
):
    """
    Create a grouped bar chart comparing RMSE across different resampling policies
    for different projection dimensions (d values).
    
    Parameters:
    -----------
    x_train, y_train, x_test, y_test : torch.Tensor
        Training and test data
    d_values : list
        List of projection dimensions to test (e.g., [50, 300])
    resample_policies : list
        List of resample_every values to test (e.g., [None, 5, 20])
        None means no resampling (resample_every = max_iter + 1)
    n_seeds : int
        Number of random seeds to run for error bars
    num_mixtures : int
        Number of spectral mixture components
    kernel_proj : kernel object, optional
        Custom kernel to use (if None, uses default SM kernel)
    cfg : TrainCfg, optional
        Training configuration (if None, uses default)
    savepath : str, optional
        Path to save the plot
    figsize : tuple
        Figure size for the plot
    
    Returns:
    --------
    dict : Results dictionary containing:
        - 'results_df': DataFrame with mean and std for each condition
        - 'raw_data': Raw results for each seed
        - 'plot_data': Data used for plotting
    """
    from ..config import TrainCfg
    from ..kernels import SM_kernel
    from ..training import run_proj
    from ..utils import set_seed
    import copy
    import pandas as pd
    
    if cfg is None:
        cfg = TrainCfg()
    
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))
    
    # storage for results
    results_data = []
    raw_data = {}
    
    print(f"Running resampling policy comparison:")
    print(f"  d_values: {d_values}")
    print(f"  resample_policies: {resample_policies}")
    print(f"  n_seeds: {n_seeds}")
    
    # run experiments for each combination
    for d in d_values:
        for resample_every in resample_policies:
            policy_name = "no resampling" if resample_every is None else f"every-{resample_every}"
            print(f"\nTesting d={d}, {policy_name}")
            
            # storage for this condition
            rmse_values = []
            nlpd_values = []
            condition_key = f"d{d}_{policy_name.replace(' ', '_')}"
            raw_data[condition_key] = []
            
            for seed_idx in range(n_seeds):
                # set seed for reproducibility
                current_seed = cfg.seed + seed_idx * 1000 + hash(f"{d}_{resample_every}") % 10000
                set_seed(current_seed)
                
                # create kernel
                if kernel_proj is None:
                    k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
                else:
                    k_proj = copy.deepcopy(kernel_proj)
                
                # create config with specific resample_every
                cfg_seed = TrainCfg(**{**vars(cfg), 'seed': current_seed})
                if resample_every is None:
                    # no resampling: set resample_every to be larger than max_iter
                    cfg_seed.resample_every = cfg_seed.max_iter + 1
                else:
                    cfg_seed.resample_every = resample_every
                
                # train projected gp
                try:
                    res_proj = run_proj(
                        x_train, y_train, x_test, y_test,
                        kernel=k_proj, d_proj=d,
                        cfg=cfg_seed, init_from_data=False
                    )
                    
                    # extract metrics
                    rmse = res_proj['metrics']['RMSE']
                    nlpd = res_proj['metrics']['NLPD']
                    
                    rmse_values.append(rmse)
                    nlpd_values.append(nlpd)
                    raw_data[condition_key].append({
                        'seed': current_seed,
                        'rmse': rmse,
                        'nlpd': nlpd,
                        'train_time': res_proj['train_time']
                    })
                    
                except Exception as e:
                    print(f"  Warning: Failed for seed {current_seed}: {e}")
                    continue
            
            # calculate statistics
            if rmse_values:
                rmse_mean = np.mean(rmse_values)
                rmse_std = np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0
                nlpd_mean = np.mean(nlpd_values)
                nlpd_std = np.std(nlpd_values, ddof=1) if len(nlpd_values) > 1 else 0.0
                
                results_data.append({
                    'd': d,
                    'resample_policy': policy_name,
                    'resample_every': resample_every,
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std,
                    'nlpd_mean': nlpd_mean,
                    'nlpd_std': nlpd_std,
                    'n_successful': len(rmse_values)
                })
                
                print(f"  RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")
                print(f"  NLPD: {nlpd_mean:.4f} +/- {nlpd_std:.4f}")
    
    # create dataframe
    results_df = pd.DataFrame(results_data)
    
    if results_df.empty:
        print("No successful experiments completed!")
        return {'results_df': results_df, 'raw_data': raw_data, 'plot_data': None}
    
    # create the grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # prepare data for plotting
    policy_names = results_df['resample_policy'].unique()
    d_values_plot = results_df['d'].unique()
    
    # set up bar positions
    x = np.arange(len(policy_names))
    width = 0.35  # width of bars
    
    # colours for different d values
    colors = ['#1f77b4', '#ff7f0e']  # blue and orange
    
    # plot bars for each d value
    for i, d in enumerate(d_values_plot):
        d_data = results_df[results_df['d'] == d]
        
        # get values in the same order as policy_names
        rmse_means = []
        rmse_stds = []
        for policy in policy_names:
            policy_data = d_data[d_data['resample_policy'] == policy]
            if not policy_data.empty:
                rmse_means.append(policy_data['rmse_mean'].iloc[0])
                rmse_stds.append(policy_data['rmse_std'].iloc[0])
            else:
                rmse_means.append(0)
                rmse_stds.append(0)
        
        # plot bars with error bars
        bars = ax.bar(x + i * width, rmse_means, width, 
                     label=f'd={d}', color=colors[i % len(colors)],
                     alpha=0.8, capsize=5)
        
        # add error bars
        ax.errorbar(x + i * width, rmse_means, yerr=rmse_stds,
                   fmt='none', color='black', capsize=3, capthick=1)
    
    # customise the plot
    ax.set_xlabel('Resampling Policy')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('RMSE Comparison Across Resampling Policies\n(Error bars = +/-1 std over seeds)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(policy_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # add value labels on bars
    for i, d in enumerate(d_values_plot):
        d_data = results_df[results_df['d'] == d]
        for j, policy in enumerate(policy_names):
            policy_data = d_data[d_data['resample_policy'] == policy]
            if not policy_data.empty:
                value = policy_data['rmse_mean'].iloc[0]
                ax.text(x[j] + i * width, value + policy_data['rmse_std'].iloc[0] + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=220, bbox_inches='tight')
        print(f"Plot saved to {savepath}")
    
    plt.show()
    
    # print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df[['d', 'resample_policy', 'rmse_mean', 'rmse_std', 'nlpd_mean', 'nlpd_std']].to_string(index=False))
    
    return {
        'results_df': results_df,
        'raw_data': raw_data,
        'plot_data': {
            'policy_names': policy_names,
            'd_values': d_values_plot,
            'x_positions': x,
            'width': width,
            'colors': colors
        }
    }

def plot_multiscale_comparison(
    x_train, y_train, x_test, y_test,
    single_scale_d=200,
    multiscale_configs=[
        [50, 100],  # two-scale: 50 + 100
        [50, 100, 200]  # three-scale: 50 + 100 + 200
    ],
    n_seeds=5,
    num_mixtures=1,
    kernel_proj=None,
    cfg=None,
    savepath=None,
    figsize=(10, 6)
):
    """
    Create a grouped bar chart comparing single-scale vs multi-scale projection GP methods.
    
    Parameters:
    -----------
    x_train, y_train, x_test, y_test : torch.Tensor
        Training and test data
    single_scale_d : int
        Single-scale projection dimension (default: 200)
    multiscale_configs : list
        List of d_list configurations for multi-scale methods
        Each element is a list of dimensions for different scales
    n_seeds : int
        Number of random seeds to run for error bars
    num_mixtures : int
        Number of spectral mixture components
    kernel_proj : kernel object, optional
        Custom kernel to use (if None, uses default SM kernel)
    cfg : TrainCfg, optional
        Training configuration (if None, uses default)
    savepath : str, optional
        Path to save the plot
    figsize : tuple
        Figure size for the plot
    
    Returns:
    --------
    dict : Results dictionary containing:
        - 'results_df': DataFrame with mean and std for each condition
        - 'raw_data': Raw results for each seed
        - 'plot_data': Data used for plotting
    """
    from ..config import TrainCfg
    from ..kernels import SM_kernel
    from ..training import run_proj, run_proj_multiscale
    from ..utils import set_seed
    import copy
    import pandas as pd
    
    if cfg is None:
        cfg = TrainCfg()
    
    device = y_train.device
    D = x_train.shape[1]
    y_std_train = float(torch.std(y_train))
    
    # storage for results
    results_data = []
    raw_data = {}
    
    print(f"Running multi-scale comparison:")
    print(f"  Single-scale d: {single_scale_d}")
    print(f"  Multi-scale configs: {multiscale_configs}")
    print(f"  n_seeds: {n_seeds}")
    
    # test single-scale method
    print(f"\nTesting single-scale d={single_scale_d}")
    rmse_values = []
    nlpd_values = []
    condition_key = f"single_scale_d{single_scale_d}"
    raw_data[condition_key] = []
    
    for seed_idx in range(n_seeds):
        # set seed for reproducibility
        current_seed = cfg.seed + seed_idx * 1000 + hash(f"single_{single_scale_d}") % 10000
        set_seed(current_seed)
        
        # create kernel
        if kernel_proj is None:
            k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
        else:
            k_proj = copy.deepcopy(kernel_proj)
        
        # create config
        cfg_seed = TrainCfg(**{**vars(cfg), 'seed': current_seed})
        
        # train single-scale projected gp
        try:
            res_proj = run_proj(
                x_train, y_train, x_test, y_test,
                kernel=k_proj, d_proj=single_scale_d,
                cfg=cfg_seed, init_from_data=False
            )
            
            # extract metrics
            rmse = res_proj['metrics']['RMSE']
            nlpd = res_proj['metrics']['NLPD']
            
            rmse_values.append(rmse)
            nlpd_values.append(nlpd)
            raw_data[condition_key].append({
                'seed': current_seed,
                'rmse': rmse,
                'nlpd': nlpd,
                'train_time': res_proj['train_time']
            })
            
        except Exception as e:
            print(f"  Warning: Failed for seed {current_seed}: {e}")
            continue
    
    # calculate statistics for single-scale
    if rmse_values:
        rmse_mean = np.mean(rmse_values)
        rmse_std = np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0
        nlpd_mean = np.mean(nlpd_values)
        nlpd_std = np.std(nlpd_values, ddof=1) if len(nlpd_values) > 1 else 0.0
        
        results_data.append({
            'method': 'Single-scale',
            'config': f'd={single_scale_d}',
            'total_d': single_scale_d,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'nlpd_mean': nlpd_mean,
            'nlpd_std': nlpd_std,
            'n_successful': len(rmse_values)
        })
        
        print(f"  RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")
        print(f"  NLPD: {nlpd_mean:.4f} +/- {nlpd_std:.4f}")
    
    # test multi-scale methods
    for i, d_list in enumerate(multiscale_configs):
        total_d = sum(d_list)
        config_name = f"{len(d_list)}-scale"
        config_desc = " + ".join(map(str, d_list))
        
        print(f"\nTesting {config_name} ({config_desc}, total d={total_d})")
        rmse_values = []
        nlpd_values = []
        condition_key = f"multiscale_{config_name}_{total_d}"
        raw_data[condition_key] = []
        
        for seed_idx in range(n_seeds):
            # set seed for reproducibility
            current_seed = cfg.seed + seed_idx * 1000 + hash(f"multi_{i}") % 10000
            set_seed(current_seed)
            
            # create kernel
            if kernel_proj is None:
                k_proj = SM_kernel(num_mixtures, D, y_std=y_std_train)
            else:
                k_proj = copy.deepcopy(kernel_proj)
            
            # create config
            cfg_seed = TrainCfg(**{**vars(cfg), 'seed': current_seed})
            
            # train multi-scale projected gp
            try:
                res_proj = run_proj_multiscale(
                    x_train, y_train, x_test, y_test,
                    kernel=k_proj, d_list=d_list,
                    cfg=cfg_seed, init_from_data=False
                )
                
                # extract metrics
                rmse = res_proj['metrics']['RMSE']
                nlpd = res_proj['metrics']['NLPD']
                
                rmse_values.append(rmse)
                nlpd_values.append(nlpd)
                raw_data[condition_key].append({
                    'seed': current_seed,
                    'rmse': rmse,
                    'nlpd': nlpd,
                    'train_time': res_proj['train_time']
                })
                
            except Exception as e:
                print(f"  Warning: Failed for seed {current_seed}: {e}")
                continue
        
        # calculate statistics for multi-scale
        if rmse_values:
            rmse_mean = np.mean(rmse_values)
            rmse_std = np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0
            nlpd_mean = np.mean(nlpd_values)
            nlpd_std = np.std(nlpd_values, ddof=1) if len(nlpd_values) > 1 else 0.0
            
            results_data.append({
                'method': config_name,
                'config': config_desc,
                'total_d': total_d,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std,
                'nlpd_mean': nlpd_mean,
                'nlpd_std': nlpd_std,
                'n_successful': len(rmse_values)
            })
            
            print(f"  RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")
            print(f"  NLPD: {nlpd_mean:.4f} +/- {nlpd_std:.4f}")
    
    # create dataframe
    results_df = pd.DataFrame(results_data)
    
    if results_df.empty:
        print("No successful experiments completed!")
        return {'results_df': results_df, 'raw_data': raw_data, 'plot_data': None}
    
    # create the grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # prepare data for plotting
    methods = results_df['method'].unique()
    configs = results_df['config'].values
    
    # set up bar positions
    x = np.arange(len(configs))
    width = 0.6  # width of bars
    
    # colors for different methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    # plot bars
    rmse_means = results_df['rmse_mean'].values
    rmse_stds = results_df['rmse_std'].values
    
    bars = ax.bar(x, rmse_means, width, 
                 color=[colors[i % len(colors)] for i in range(len(rmse_means))],
                 alpha=0.8, capsize=5)
    
    # add error bars
    ax.errorbar(x, rmse_means, yerr=rmse_stds,
               fmt='none', color='black', capsize=3, capthick=1)
    
    # customise the plot
    ax.set_xlabel('Projection Configuration')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('RMSE Comparison: Single-scale vs Multi-scale Projection GP\n(Error bars = +/-1 std over seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # add value labels on bars
    for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
        ax.text(x[i], mean + std + 0.01,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # add legend for methods
    method_colors = {}
    for i, method in enumerate(methods):
        method_colors[method] = colors[i % len(colors)]
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=method) 
                      for method, color in method_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=220, bbox_inches='tight')
        print(f"Plot saved to {savepath}")
    
    plt.show()
    
    # print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df[['method', 'config', 'total_d', 'rmse_mean', 'rmse_std', 'nlpd_mean', 'nlpd_std']].to_string(index=False))
    
    return {
        'results_df': results_df,
        'raw_data': raw_data,
        'plot_data': {
            'methods': methods,
            'configs': configs,
            'x_positions': x,
            'width': width,
            'colors': colors
        }
    }
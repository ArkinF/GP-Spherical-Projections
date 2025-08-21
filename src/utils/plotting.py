import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

def plot_posterior(x_train, y_train, x_test, y_mean, y_std, title='Posterior GP'):
    """simple posterior plot"""
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
    """plot 1D posteriors for all methods"""
    # device from model params
    device = next(res_exact['model'].parameters()).device
    x_plot = np.asarray(x_plot).reshape(-1)
    x_plot_tensor = torch.as_tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # exact
        pe = res_exact['likelihood'](res_exact['model'](x_plot_tensor))
        mu_e = pe.mean.detach().cpu().numpy()
        sd_e = pe.variance.detach().cpu().numpy() ** 0.5

        # spherical
        pp = res_proj['likelihood'](res_proj['model'](x_plot_tensor))
        mu_p = pp.mean.detach().cpu().numpy()
        sd_p = pp.variance.detach().cpu().numpy() ** 0.5

        # SVGP (optional)
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

    _plot(x_te_np, y_te_np, xs, mu_e[order], sd_e[order], title="posterior (exact GP)")
    title_proj = f"posterior (projection GP{'' if d_proj is None else f', d={d_proj}'})"
    _plot(x_te_np, y_te_np, xs, mu_p[order], sd_p[order], title=title_proj)

    if mu_s is not None:
        _plot(x_te_np, y_te_np, xs, mu_s[order], sd_s[order], title="posterior (SVGP)")

    return dict(x_plot=xs, mu_exact=mu_e[order], sd_exact=sd_e[order],
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
    """produce 2 or 3 vertically stacked subplots"""
    # prepare line inputs
    device = next(res_exact['model'].parameters()).device
    x_plot = np.asarray(x_plot).reshape(-1)
    order  = np.argsort(x_plot)
    xs     = x_plot[order]
    x_line = torch.as_tensor(xs, dtype=torch.float32, device=device).unsqueeze(-1)

    # predict on x_line
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
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

    # plot stacked
    nrows = 3 if (mu_s is not None) else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.1*nrows), sharex=True)

    if nrows == 2:
        ax_exact, ax_proj = axes
    else:
        ax_exact, ax_proj, ax_svgp = axes

    def _one(ax, title, mu, sd):
        ax.scatter(x_te, y_te, color='r', marker='.', label="test")
        ax.plot(xs, mu, 'b', label="mean")
        ax.fill_between(xs, mu - 2*sd, mu + 2*sd, alpha=0.3, color='blue', label="+/-2sigma")
        ax.set_title(title)
        ax.set_ylabel("$y$")
        ax.grid(alpha=0.2)

    _one(ax_exact, "exact GP", mu_e, sd_e)
    _one(ax_proj,  f"spherical GP{'' if d_proj is None else f' (d={d_proj})'}", mu_p, sd_p)
    if nrows == 3:
        _one(ax_svgp, "SVGP", mu_s, sd_s)

    axes[-1].set_xlabel("$x$")
    # legend once (top)
    handles, labels = ax_exact.get_legend_handles_labels()
    ax_exact.legend(handles, labels)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=220)
    plt.show()

    # return the arrays in case you want to reuse them
    return dict(x_plot=xs, mu_exact=mu_e, sd_exact=sd_e,
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
    """plot 2D posteriors"""
    device = next(res_exact['model'].parameters()).device
    xg = x_grid.to(device=device, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # exact and spherical means on grid
        pe = res_exact['likelihood'](res_exact['model'](xg))
        pp = res_proj ['likelihood'](res_proj ['model'](xg))
        mu_e = pe.mean.view(*grid_shape).detach().cpu().numpy()
        mu_p = pp.mean.view(*grid_shape).detach().cpu().numpy()

        mu_s = None
        if add_svgp_heatmap and (res_svgp is not None) and (predict_svgp_fn is not None):
            mu_s_flat, _ = predict_svgp_fn(res_svgp['model'], res_svgp['likelihood'], xg, device=device)
            mu_s = mu_s_flat.view(*grid_shape).detach().cpu().numpy()

    xmin, xmax, ymin, ymax = extent
    ncols = 4 if (add_svgp_heatmap and (mu_s is not None)) else 3
    fig, axs = plt.subplots(1, ncols, figsize=(19 if ncols==4 else 15, 4.8), constrained_layout=True)

    im0 = axs[0].imshow(mu_e, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
    axs[0].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
    axs[0].set_title("exact: mean"); axs[0].set_xlabel("$x_1$"); axs[0].set_ylabel("$x_2$")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(mu_p, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
    axs[1].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
    axs[1].set_title(f"spherical (d={d_proj})" if d_proj is not None else "spherical"); axs[1].set_xlabel("$x_1$"); axs[1].set_ylabel("$x_2$")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    col = 2
    if ncols == 4:
        im2 = axs[2].imshow(mu_s, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
        axs[2].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
        axs[2].set_title("SVGP: mean"); axs[2].set_xlabel("$x_1$"); axs[2].set_ylabel("$x_2$")
        fig.colorbar(im2, ax=axs[2], fraction=0.046)
        col = 3

    # |mean difference| vs exact
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

    imd = axs[col].imshow(diff, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto")
    axs[col].scatter(x_train[:,0].cpu(), x_train[:,1].cpu(), s=6, c="k", alpha=0.25)
    axs[col].set_title(diff_title); axs[col].set_xlabel("$x_1$"); axs[col].set_ylabel("$x_2$")
    fig.colorbar(imd, ax=axs[col], fraction=0.046)

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
    """auto-detects 1D vs 2D based on D and provided plotting inputs"""
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
        print("no plotting: provide x_plot for 1D OR (x_train,x_grid,grid_shape,extent) for 2D.")
        return None

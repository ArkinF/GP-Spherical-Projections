import torch
import gpytorch
import time
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset

from ..config import TrainCfg, SVGPcfg
from ..utils import compute_metrics, extract_hyp, create_mean_function 
from ..models import (
    ExactGPModel, 
    SVGPModel,
    ProjectionObjective,
    MultiScaleProjectionObjective,
)

def train_exact(model, likelihood, x_train, y_train, cfg: TrainCfg):
    """train exact gp"""
    model.train(); likelihood.train()

    # optimiser
    if cfg.use_lbfgs:
        opt = torch.optim.LBFGS(model.parameters(),
                               lr=cfg.lr, max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if cfg.verbose:
        print(f"\nrunning exact GP training | iters={cfg.max_iter} lr={cfg.lr} "
              f"lbfgs={cfg.use_lbfgs} jitter={cfg.jitter_train:g}")

    t0 = time.time(); wait=0; best=float("inf"); prev=float("inf")

    for it in range(cfg.max_iter):
        if cfg.use_lbfgs:
            def closure():
                opt.zero_grad()
                try:
                    output = model(x_train)
                    loss = -mll(output, y_train)
                except gpytorch.utils.errors.NotPSDError:
                    with gpytorch.settings.cholesky_jitter(cfg.jitter_train):
                        output = model(x_train)
                        loss = -mll(output, y_train)
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            try:
                output = model(x_train)
                loss = -mll(output, y_train)
            except gpytorch.utils.errors.NotPSDError:
                with gpytorch.settings.cholesky_jitter(cfg.jitter_train):
                    output = model(x_train)
                    loss = -mll(output, y_train)
            loss.backward()
            opt.step()

        dl = abs(prev - loss.item())
        grad_norm = sum((p.grad.norm().item() for p in model.parameters() if p.grad is not None), 0.0)
        
        if cfg.verbose and (it % cfg.print_every == 0):
            rn = likelihood.noise_covar.raw_noise
            rn_grad = None if rn.grad is None else rn.grad.detach().abs().max().item()
            print(f"[exact] iter={it:04d} loss={float(loss):.6f} Delta={dl:.2e} ||grad||={grad_norm:.2e}")
            print(f"noise={likelihood.noise.item():.3g} |raw_noise|grad={rn_grad}")
            
        if loss.item() < best - cfg.loss_tol:
            best = loss.item(); wait=0
        else:
            wait += 1
            if wait >= cfg.patience:
                if cfg.verbose:
                    print(f"[exact] early stop @ {it+1}")
                break
        if (dl < cfg.loss_tol) or (grad_norm < cfg.grad_tol):
            if cfg.verbose:
                print(f"[exact] converged @ {it+1}")
            break
        prev = loss.item()

    t = time.time() - t0

    if cfg.verbose:
        print(f"[exact] done | time={t:.2f}s best_loss={best:.6f}")
    
    return dict(loss=best, train_time=t)


def train_proj(proj_obj, model, likelihood, cfg: TrainCfg):
    """train projection gp"""
    model.train(); likelihood.train()
    
    if cfg.use_lbfgs:
        opt = torch.optim.LBFGS(
            model.covar_module.parameters(),
            lr=cfg.lr, max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-9)
    else:
        # use smaller lr for likelihood noise to avoid noise inflation
        opt = torch.optim.Adam([
            { 'params': model.covar_module.parameters(), 'lr': cfg.lr},
            { 'params': likelihood.noise_covar.raw_noise, 'lr': 0.1 * cfg.lr}
        ])

    if cfg.verbose:
        print(f"\nrunning spherical GP training | iters={cfg.max_iter} lr={cfg.lr} d={proj_obj.d} jitter={cfg.jitter_train:g}")

    t0 = time.time(); wait=0; best=float("inf"); prev=float("inf")

    for it in range(cfg.max_iter):
        if cfg.use_lbfgs:
            def closure_proj():
                opt.zero_grad()
                loss = proj_obj()
                loss.backward()
                return loss
            loss = opt.step(closure_proj)
        else:
            opt.zero_grad()
            loss = proj_obj()
            loss.backward()
            opt.step()

            # refresh projection directions to reduce MC bias
            if (it > 0) and (it % cfg.resample_every == 0):
                with torch.no_grad():
                    proj_obj.resample_omegas(seed=cfg.seed + it)

        dl = abs(prev - loss.item())
        grad_norm = sum((p.grad.norm().item() for p in model.parameters() if p.grad is not None), 0.0)

        if cfg.verbose and (it % cfg.print_every == 0):
            rn = likelihood.noise_covar.raw_noise
            rn_grad = None if rn.grad is None else rn.grad.detach().abs().max().item()
            print(f"[proj ] iter={it:04d} loss={float(loss):.6f} Delta={dl:.2e} ||grad||={grad_norm:.2e}")
            print(f"noise={likelihood.noise.item():.3g} |raw_noise|grad={rn_grad}")
        
        if loss.item() < best - cfg.loss_tol:
            best = loss.item(); wait=0
        else:
            wait += 1
            if wait >= cfg.patience:
                if cfg.verbose:
                    print(f"[proj] early stop @ {it+1}")
                break
        if (dl < cfg.loss_tol) or (grad_norm < cfg.grad_tol):
            if cfg.verbose:
                print(f"[proj] converged @ {it+1}")
            break
        prev = loss.item()

    t = time.time() - t0

    if cfg.verbose:
        print(f"[proj ] done | time={t:.2f}s best_loss={best:.6f}")
    
    return dict(loss=best, train_time=t)

def train_svgp(model, likelihood, x_train, y_train, cfg: SVGPcfg, device=None):
    """train svgp"""
    if device is None:
        device = y_train.device

    # elbo (beta=1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x_train.shape[0])

    # optimisers
    ngd  = gpytorch.optim.NGD(model.variational_parameters(), num_data=x_train.shape[0], lr=cfg.lr_ngd)
    adam = torch.optim.Adam(list(model.hyperparameters()) + list(likelihood.parameters()), lr=cfg.lr_hyp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, mode='max', factor=0.5, patience=5)

    # dataloader
    seed_gen = torch.Generator(device=device)
    seed_gen.manual_seed(cfg.seed)
    dl = DataLoader(TensorDataset(x_train, y_train.view(-1)),
                            batch_size=min(cfg.batch_size, x_train.shape[0]), shuffle=True,
                            generator=seed_gen)

    # training
    model.train(); likelihood.train()
    best_elbo = -float('inf'); bad = 0
    prev_loss = float('inf')

    if cfg.verbose:
        print(f"\nrunning SVGP training | steps={cfg.steps} inducing points={cfg.m_inducing} lr={cfg.lr_hyp}")

    t0 = time.time()

    with gpytorch.settings.cholesky_jitter(1e-3):
        for step in range(1, cfg.steps+1):
            epoch_loss = 0
            num_batches = 0
            for xb, yb in dl:
                ngd.zero_grad(); adam.zero_grad()
                
                try:
                    out = model(xb)
                    loss = -mll(out, yb)
                except gpytorch.utils.errors.NotPSDError:
                    # increase jitter and retry
                    with gpytorch.settings.cholesky_jitter(1e-4):
                        output = model(x_train)
                        loss = -mll(output, y_train)

                loss.backward()
                ngd.step(); adam.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            loss_change = abs(prev_loss - avg_epoch_loss) if prev_loss != float('inf') else float('inf')

            # progress reporting
            if step % cfg.print_every == 0 or step == cfg.steps:
                grad_norm = 0.0
                num_params = 0
                for param in model.hyperparameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                        num_params += 1
                for param in likelihood.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                        num_params += 1
                grad_norm = (grad_norm / max(num_params, 1)) ** 0.5
                if cfg.verbose:
                    print(f"[SVGP] epoch {step:4d}/{cfg.steps}: loss={avg_epoch_loss:.6f}, "
                        f"Delta_loss={loss_change:.2e}, gradnorm={grad_norm:.2e}")     
            
            prev_loss = avg_epoch_loss

            # evaluation for scheduler and early stopping
            if step % cfg.eval_every == 0 or step == cfg.steps:
                with torch.no_grad():
                    elbo_now = sum([mll(model(xb), yb).item() for xb, yb in dl]) / len(dl)
                scheduler.step(elbo_now)
                if elbo_now > best_elbo:
                    best_elbo, bad = elbo_now, 0
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        break

    t_svgp = time.time() - t0

    return dict(train_time=t_svgp, best_elbo=best_elbo)

def predict_test(model, likelihood, X):
    """predict on test data"""
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
        mu = pred.mean.detach().cpu().numpy()
        var = pred.variance.detach().cpu().numpy()
    return mu, var

@torch.no_grad()
def predict_svgp(model, likelihood, X, batch_size=4096, device=None):
    """predict with svgp"""
    model.eval(); likelihood.eval()
    means, vars_ = [], []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size]
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(xb))
        means.append(pred.mean.detach().cpu())
        vars_.append(pred.variance.detach().cpu())
    return torch.cat(means), torch.cat(vars_)

def run_exact(x_train, y_train, x_test, y_test, kernel, cfg: TrainCfg, init_from_data: bool=False, flag=True):
    """run exact gp experiment"""
    
    y_std_train = float(torch.std(y_train))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    mean_module = create_mean_function(cfg.mean_cfg, input_dim=x_train.shape[1]) # mean function
    
    model = ExactGPModel(x_train, y_train, likelihood, kernel, mean_module=mean_module)
    if init_from_data:
        model.covar_module.initialize_from_data(x_train, y_train)
    with torch.no_grad():
        likelihood.noise = cfg.noise_init * y_std_train
    out = train_exact(model, likelihood, x_train, y_train, cfg)
    mu, var = predict_test(model, likelihood, x_test)
    rmse, r2, nlpd, cov95, piw95 = compute_metrics(y_test.cpu().numpy(), mu, var)
    if flag:
        params = extract_hyp(model, likelihood)
    else:
        params = likelihood.noise.detach().cpu().item()
    return dict(model=model, likelihood=likelihood,
                mu_test=mu, var_test=var, metrics=dict(RMSE=rmse, R2=r2, NLPD=nlpd, COV95=cov95, PIW95=piw95),
                train_time=out['train_time'], loss=out['loss'], params=params)

def run_proj(x_train, y_train, x_test, y_test, kernel, d_proj: int, cfg: TrainCfg, init_from_data: bool=False, flag=True):
    """run projection gp experiment"""

    y_std_train = float(torch.std(y_train))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    mean_module = create_mean_function(cfg.mean_cfg, input_dim=x_train.shape[1]) # mean function
    # noise_prior = gpytorch.priors.GammaPrior(100, 0.001)
    # likelihood.noise_covar.register_prior('noise_prior', noise_prior, 'raw_noise')

    model = ExactGPModel(x_train, y_train, likelihood, kernel, mean_module=mean_module)
    if init_from_data:
        model.covar_module.initialize_from_data(x_train, y_train)
    with torch.no_grad():
        likelihood.noise = cfg.noise_init * y_std_train
    proj_obj = ProjectionObjective(model, likelihood, y_train, d=d_proj, seed=cfg.seed, jitter=1e-4)

    # sanity: omegas unit norm
    with torch.no_grad():
        norms = torch.norm(proj_obj.Omegas, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    out = train_proj(proj_obj, model, likelihood, cfg)
    mu, var = predict_test(model, likelihood, x_test)
    rmse, r2, nlpd, cov95, piw95 = compute_metrics(y_test.cpu().numpy(), mu, var)
    if flag:
        params = extract_hyp(model, likelihood)
    else:
        params = likelihood.noise.detach().cpu().item()
    return dict(model=model, likelihood=likelihood,
                mu_test=mu, var_test=var, metrics=dict(RMSE=rmse, R2=r2, NLPD=nlpd, COV95=cov95, PIW95=piw95),
                train_time=out['train_time'], loss=out['loss'], params=params)

def run_svgp(x_train, y_train, x_test, y_test, kernel_svgp, cfg: SVGPcfg, num_mixtures=1, device=None, flag=True):
    """run svgp experiment"""
    if device is None:
        device = y_train.device

    y_std_train = float(torch.std(y_train))

    # default: SM kernel if none is passed
    if kernel_svgp is None:
        from ..kernels import SM_kernel
        kernel_svgp = SM_kernel(
            num_mixtures=num_mixtures,
            num_dims=x_train.shape[-1],
            y_std=y_std_train
        )

    # inducing points
    N = x_train.shape[0]
    m = min(cfg.m_inducing, N)
    idx = torch.linspace(0, N-1, m, dtype=torch.long, device=x_train.device)
    inducing = x_train[idx]

    mean_module = create_mean_function(cfg.mean_cfg, input_dim=x_train.shape[1]) # mean function
    
    # model & likelihood
    model = SVGPModel(inducing_points=inducing, kernel=kernel_svgp, mean_module=mean_module).to(device)
    if cfg.init_hyps_data:
        model.covar_module.initialize_from_data(x_train, y_train)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    with torch.no_grad():
        likelihood.noise = 0.01 * y_std_train

    out = train_svgp(model, likelihood, x_train, y_train, cfg, device=device)

    # predict on test
    with torch.no_grad():
        mu_te, var_te = predict_svgp(model, likelihood, x_test, device=device)
        mu_te, var_te = mu_te.numpy(), var_te.numpy()

    # metrics
    rmse_te, r2_te, nlpd_te, cov95, piw95 = compute_metrics(y_test.numpy(), mu_te, var_te)

    # extract learned hypers generically
    learned = {}
    for name, param in model.covar_module.named_parameters():
        learned[name] = param.detach().cpu().numpy()
    learned['noise'] = float(likelihood.noise.detach().cpu().item())

    if flag:
        params = extract_hyp(model, likelihood) 
    else:
        params = likelihood.noise.detach().cpu().item()

    return dict(
        model=model, likelihood=likelihood,
        mu_test=mu_te, var_test=var_te,
        metrics=dict(RMSE=rmse_te, R2=r2_te, NLPD=nlpd_te, COV95=cov95, PIW95=piw95),
        train_time=out['train_time'], best_elbo=out['best_elbo'],
        params=params, learned=learned
    )

def run_proj_multiscale(x_train, y_train, x_test, y_test, kernel, cfg: TrainCfg, d_list=[50, 100, 200], 
                       init_from_data: bool=False, flag=True):
    """run multi-scale projection gp experiment"""
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_module = create_mean_function(cfg.mean_cfg, input_dim=x_train.shape[1]) # mean function
    model = ExactGPModel(x_train, y_train, likelihood, kernel, mean_module=mean_module)
    if init_from_data:
        model.covar_module.initialize_from_data(x_train, y_train)
    with torch.no_grad():
        likelihood.noise = cfg.noise_init
    
    proj_obj = MultiScaleProjectionObjective(
        model, likelihood, y_train, d_list=d_list, seed=cfg.seed, jitter=1e-4
    )

    # check omegas norm
    with torch.no_grad():
        if hasattr(proj_obj, 'Omegas'):
            norms = torch.norm(proj_obj.Omegas, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        else:
            for proj in proj_obj.proj_objects:
                norms = torch.norm(proj.Omegas, dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    out = train_proj(proj_obj, model, likelihood, cfg)
    mu, var = predict_test(model, likelihood, x_test)
    rmse, r2, nlpd, cov95, piw95 = compute_metrics(y_test.cpu().numpy(), mu, var)
    
    if flag:
        params = extract_hyp(model, likelihood)
    else:
        params = likelihood.noise.detach().cpu().item()
    
    # add multi-scale specific information
    result = dict(
        model=model, likelihood=likelihood,
        mu_test=mu, var_test=var, 
        metrics=dict(RMSE=rmse, R2=r2, NLPD=nlpd, COV95=cov95, PIW95=piw95),
        train_time=out['train_time'], loss=out['loss'], params=params,
        d_list=d_list, total_d=proj_obj.d
    )

    return result


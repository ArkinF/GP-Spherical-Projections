import numpy as np
from scipy.stats import norm

Z95 = norm.ppf(0.975)

def compute_cov95_and_piwidth(y_true, mu, var, center_offset=None):
    """compute coverage and prediction interval width"""
    # to numpy 1D
    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    y = to_np(y_true).ravel()
    m = to_np(mu).ravel()
    s = np.sqrt(np.clip(to_np(var).ravel(), 1e-12, np.inf))

    if center_offset is not None:
        m = m + float(center_offset)

    lower = m - Z95 * s
    upper = m + Z95 * s
    coverage = 100.0 * np.mean((y >= lower) & (y <= upper))
    width = np.mean(upper - lower)
    return float(coverage), float(width)

def extract_hyp(model, likelihood=None):
    """extract hyperparameters from model, handling different kernel types"""
    kernel = model.covar_module
    
    # noise variance
    if likelihood is None:
        noise_var = model.likelihood.noise.detach().cpu().item()
    else:
        noise_var = likelihood.noise.detach().cpu().item()
    
    # extract based kernel
    if hasattr(kernel, 'mixture_weights'):  # Spectral Mixture kernel
        w = kernel.mixture_weights.detach().cpu().numpy().flatten()
        m = kernel.mixture_means.detach().cpu().numpy().flatten()
        s = kernel.mixture_scales.detach().cpu().numpy().flatten()
        
        if w.size == 1:
            return [float(w.item()), float(s.item()), float(m.item()), noise_var]
        else:
            return [w, s, m, noise_var]
    
    elif hasattr(kernel, 'base_kernel'): # other kernels with ScaleKernel wrapper (added for stability)
        base_kernel = kernel.base_kernel
        outputscale = kernel.outputscale.detach().cpu().item()
        
        if hasattr(base_kernel, 'kernels'):  # composite kernels (like periodic * rbf)
            params = []
            for k in base_kernel.kernels:
                if hasattr(k, 'lengthscale'):
                    params.append(k.lengthscale.detach().cpu().numpy())
                if hasattr(k, 'period_length'):
                    params.append(k.period_length.detach().cpu().item())
            params.append(outputscale)
            params.append(noise_var)
            return params
        
        elif hasattr(base_kernel, 'lengthscale'):  # RBF, Matern, etc.
            lengthscale = base_kernel.lengthscale.detach().cpu().numpy()
            if hasattr(base_kernel, 'period_length'):  # Periodic kernel
                period = base_kernel.period_length.detach().cpu().item()
                return [lengthscale, period, outputscale, noise_var]
            else:
                return [lengthscale, outputscale, noise_var]
    
    return "unknown kernel"

def compute_metrics(y_true, mu, var, proj_obj=None, center_offset=None):
    """compute all evaluation metrics"""
    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    y = to_np(y_true).ravel()
    m = to_np(mu).ravel()
    v = np.clip(to_np(var).ravel(), 1e-12, np.inf)
    if center_offset is None and proj_obj is not None and hasattr(proj_obj, "y_mean"):
        center_offset = float(proj_obj.y_mean.detach().cpu())
    if center_offset is not None:
        m = m + center_offset

    rmse = float(np.sqrt(np.mean((y - m)**2)))
    r2   = float(1.0 - np.var(y - m) / (np.var(y) + 1e-12))
    nlpd = float(0.5 * np.mean(np.log(2*np.pi*v) + (y - m)**2 / v))

    sd = np.sqrt(v)
    cov95 = 100.0 * np.mean((y >= m - Z95*sd) & (y <= m + Z95*sd))
    piw95 = np.mean(2.0 * Z95 * sd)

    return [rmse, r2, nlpd, cov95, piw95]

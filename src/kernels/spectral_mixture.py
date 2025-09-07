import torch
import gpytorch

def SM_kernel(num_mixtures=1, num_dims=1, initial_hyps=None, y_std=None):
    """sm kernel init for q=1 or q=2 mixtures"""
    kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=num_dims)

    if y_std is None:
        y_std = 1.0

    if num_mixtures == 1:
        if initial_hyps is None:
            weights = torch.tensor([[0.5 * y_std]])  # [Q=1, 1]
            scales  = torch.tensor([[0.1] * num_dims])  # [Q=1, D] 
            means   = torch.tensor([[0.05] * num_dims]) # [Q=1, D]
        else:
            weights, scales, means = initial_hyps
            
    elif num_mixtures == 2:
        # trend + annual
        if initial_hyps is None:
            weights = torch.tensor([[0.8 * y_std], [0.4 * y_std]])  # [Q=2, 1]
            scales  = torch.tensor([[0.05] * num_dims, [0.5] * num_dims])  # [Q=2, D]
            means   = torch.tensor([[0.0] * num_dims, [1.0] * num_dims])   # [Q=2, D]
        else:
            weights, scales, means = initial_hyps

    else:
        # for num_mixtures > 2, return uninitialized kernel
        return kernel

    with torch.no_grad():
        kernel.mixture_weights = weights
        kernel.mixture_scales = scales  
        kernel.mixture_means = means

    kernel.register_constraint("raw_mixture_weights", gpytorch.constraints.GreaterThan(1e-6))
    kernel.register_constraint("raw_mixture_scales", gpytorch.constraints.GreaterThan(1e-6))

    return kernel

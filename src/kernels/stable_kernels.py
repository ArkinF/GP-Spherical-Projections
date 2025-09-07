import gpytorch

def long_periodic_plus_short_rbf(init_period=1.0, init_rbf_lengthscale=0.1):
    periodic = gpytorch.kernels.PeriodicKernel()
    periodic.period_length = init_period
    periodic.lengthscale = 1.0
    
    periodic.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-4))
    
    rbf_short = gpytorch.kernels.RBFKernel()
    rbf_short.lengthscale = init_rbf_lengthscale
    rbf_short.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-4))
    
    combined = gpytorch.kernels.ScaleKernel(periodic * rbf_short)
    combined.register_constraint("raw_outputscale", gpytorch.constraints.GreaterThan(1e-4))
    
    return combined

def stable_periodic_kernel(init_period=1.0, init_lengthscale=1.0):
    """periodic kernel with fixed period and lengthscale"""
    periodic = gpytorch.kernels.PeriodicKernel()
    periodic.period_length = init_period
    periodic.lengthscale = init_lengthscale
    return gpytorch.kernels.ScaleKernel(periodic)

def stable_rbf_kernel(init_lengthscale=1.0):
    rbf = gpytorch.kernels.RBFKernel()
    rbf.lengthscale = init_lengthscale
    return gpytorch.kernels.ScaleKernel(rbf)

def stable_matern_kernel(init_lengthscale=1.0, nu=2.5):
    matern = gpytorch.kernels.MaternKernel(nu=nu)
    matern.lengthscale = init_lengthscale
    return gpytorch.kernels.ScaleKernel(matern)

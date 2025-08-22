import gpytorch


def create_mean_function(mean_type: str, input_dim: int = None):
    """used for GP mean"""
    if mean_type == "zero":
        return gpytorch.means.ZeroMean()
    
    elif mean_type == "constant":
        return gpytorch.means.ConstantMean()
    
    elif mean_type == "linear":
        if input_dim is None:
            raise ValueError("input_dim must be provided for linear mean")
        return gpytorch.means.LinearMean(input_dim, bias=True)
    
    else:
        raise ValueError(f"Unknown mean function type: {mean_type}")

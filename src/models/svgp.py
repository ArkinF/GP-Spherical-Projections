import torch
import gpytorch
from gpytorch.variational import (
    NaturalVariationalDistribution,
    VariationalStrategy,
)

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel, mean_module=None):
        M, _ = inducing_points.shape
        q_dist = NaturalVariationalDistribution(M)

        strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=q_dist,
            learn_inducing_locations=True,
        )

        super().__init__(strategy)

        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # add diagonal jitter for numerical stability
        if self.training:
            covar_x = covar_x.add_jitter(1e-4)
            
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

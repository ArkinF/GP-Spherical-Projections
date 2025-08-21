import torch

class ProjectionObjective(torch.nn.Module):
    def __init__(self, model, likelihood, train_y, d=100, seed=0, device=None, dtype=None, jitter: float = 1e-4):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.jitter = float(jitter)

        self.y = train_y.view(-1)
        self.d = d
        
        n = self.y.numel()

        # device/dtype
        if device is None:
            device = self.y.device
        if dtype is None:
            dtype = self.y.dtype

        # fixed omegas for the whole optimization (monte-carlo set)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        Omegas = torch.randn(d, n, generator=g, device=device, dtype=dtype)
        Omegas = Omegas / (Omegas.norm(dim=1, keepdim=True) + 1e-12)
        self.register_buffer("Omegas", Omegas)               # [d, n]
        self.register_buffer("Omega_sq_sum", (Omegas**2).sum(dim=1))  # [d]
        self.register_buffer("proj_y", Omegas @ self.y)      # [d]

        self.eps = 1e-10

    @torch.no_grad()
    def resample_omegas(self, seed=None):
        d, n = self.Omegas.shape
        if seed is not None:
            g = torch.Generator(device=self.y.device); g.manual_seed(seed)
            Omegas = torch.randn(d, n, generator=g, device=self.y.device, dtype=self.y.dtype)
        else:
            Omegas = torch.randn(d, n, device=self.y.device, dtype=self.y.dtype)
        Omegas = Omegas / (Omegas.norm(dim=1, keepdim=True) + 1e-12)
        self.Omegas.copy_(Omegas)
        self.Omega_sq_sum.copy_((Omegas**2).sum(dim=1))
        self.proj_y.copy_(Omegas @ self.y)

    def forward(self):
        X = self.model.train_inputs[0]
        K = self.model.covar_module(X).add_jitter(self.jitter)

        # compute v = K * Omegas^T - [n, d]
        v = K.matmul(self.Omegas.T)
        
        # quadratic forms omega^T K omega = sum_i omega_i * (v_i)
        quad = (self.Omegas.T * v).sum(dim=0)

        # add learned noise sigma_n^2
        sigma_n2 = self.likelihood.noise

        var = quad + sigma_n2 * self.Omega_sq_sum
        var = var.clamp_min(self.eps)
        proj = self.proj_y

        # base projected gaussian NLL (averaged over d projections)
        loss = 0.5 * (torch.log(2 * torch.pi * var) + (proj**2) / var).mean()

        return loss

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

        # inital omega sample (monte-carlo set)
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

        # compute loss
        loss = 0.5 * (torch.log(2 * torch.pi * var) + (proj**2) / var).mean()
        return loss


class MultiScaleProjectionObjective(torch.nn.Module):
    """uses multiple direction samples"""
    def __init__(self, model, likelihood, train_y, d_list=[50, 100, 200], seed=0, 
                 device=None, dtype=None, jitter: float = 1e-4):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.jitter = float(jitter)
        self.d_list = d_list
        self.num_scales = len(d_list)

        self.y = train_y.view(-1)
        n = self.y.numel()

        # device/dtype
        if device is None:
            device = self.y.device
        if dtype is None:
            dtype = self.y.dtype

        # create concatenated omegas for all scales
        total_d = sum(d_list)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        Omegas = torch.randn(total_d, n, generator=g, device=device, dtype=dtype)
        Omegas = Omegas / (Omegas.norm(dim=1, keepdim=True) + 1e-12)
        self.register_buffer("Omegas", Omegas)               # [total_d, n]
        self.register_buffer("Omega_sq_sum", (Omegas**2).sum(dim=1))  # [total_d]
        self.register_buffer("proj_y", Omegas @ self.y)      # [total_d]

        # create scale indices for slicing
        self.scale_indices = []
        start_idx = 0
        for d in d_list:
            end_idx = start_idx + d
            self.scale_indices.append((start_idx, end_idx))
            start_idx = end_idx

        self.eps = 1e-10

    @property 
    def d(self):
        """total dimension across all scales"""
        return sum(self.d_list)

    @torch.no_grad()
    def resample_omegas(self, seed=None):
        """resample omegas"""
        total_d, n = self.Omegas.shape
        if seed is not None:
            g = torch.Generator(device=self.y.device); g.manual_seed(seed)
            Omegas = torch.randn(total_d, n, generator=g, device=self.y.device, dtype=self.y.dtype)
        else:
            Omegas = torch.randn(total_d, n, device=self.y.device, dtype=self.y.dtype)
        Omegas = Omegas / (Omegas.norm(dim=1, keepdim=True) + 1e-12)
        self.Omegas.copy_(Omegas)
        self.Omega_sq_sum.copy_((Omegas**2).sum(dim=1))
        self.proj_y.copy_(Omegas @ self.y)

    def forward(self):
        """compute loss as average across scales"""
        X = self.model.train_inputs[0]
        K = self.model.covar_module(X).add_jitter(self.jitter)

        # compute v = K * Omegas^T - [n, total_d] (shared computation)
        v = K.matmul(self.Omegas.T)
        
        # quadratic forms omega^T K omega = sum_i omega_i * (v_i)
        quad = (self.Omegas.T * v).sum(dim=0)

        # add learned noise sigma_n^2
        sigma_n2 = self.likelihood.noise

        var = quad + sigma_n2 * self.Omega_sq_sum
        var = var.clamp_min(self.eps)
        proj = self.proj_y

        # compute loss for each scale separately
        scale_losses = []
        for start_idx, end_idx in self.scale_indices:
            scale_var = var[start_idx:end_idx]
            scale_proj = proj[start_idx:end_idx]
            scale_loss = 0.5 * (torch.log(2 * torch.pi * scale_var) + (scale_proj**2) / scale_var).mean()
            scale_losses.append(scale_loss)

        # average across scales
        total_loss = torch.stack(scale_losses).mean()
        return total_loss

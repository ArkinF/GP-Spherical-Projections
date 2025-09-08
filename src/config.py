from dataclasses import dataclass

@dataclass
class TrainCfg:
    max_iter: int = 500
    loss_tol: float = 1e-6
    grad_tol: float = 1e-4
    patience: int = 40
    lr: float = 0.02
    use_lbfgs: bool = False
    jitter_train: float = 1e-4
    seed: int = 0
    verbose: bool = True
    print_every: int = 100
    upper_noise_constraint: int = 0.2
    noise_init: float = 0.01
    mean_cfg: str = "zero"
    resample_every: int = 25
    use_separate_noise_lr: bool = True
    exact_gp_restarts: int = 3

@dataclass
class SVGPcfg:
    m_inducing: int = 256
    batch_size: int = 1024
    steps: int = 200
    lr_hyp: float = 3e-3
    lr_ngd: float = 0.01
    seed: int = 0
    eval_every: int = 100
    patience: int = 25
    verbose: bool = True
    print_every: int = 100
    init_hyps_data: bool = False
    mean_cfg: str = "zero"

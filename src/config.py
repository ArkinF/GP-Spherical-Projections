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
    seed: int = 20250814
    verbose: bool = True
    print_every: int = 100
    upper_noise_constraint: int = 0.2
    noise_init: float = 0.1
    mean_cfg: str = "zero"

@dataclass
class SVGPcfg:
    m_inducing: int = 128
    batch_size: int = 1024
    steps: int = 1500
    lr_hyp: float = 3e-3
    lr_ngd: float = 0.01
    seed: int = 0
    eval_every: int = 200
    patience: int = 25
    verbose: bool = True
    print_every: int = 200
    init_hyps_data: bool = False
    mean_cfg: str = "zero"

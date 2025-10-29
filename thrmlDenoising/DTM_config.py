from dataclasses import dataclass
from typing import Literal, Tuple, Union

from thrmlDenoising.base_graphs.abstract_base_graph_manager import AbstractBaseGraphManager

# ---------- Data ----------
@dataclass(frozen=True)
class DataConfig:
    dataset_name: Literal["mnist", "fashion_mnist", "smoke_testing"] = "mnist"
    target_classes: Tuple[int, ...] = tuple(range(4))
    pixel_threshold_for_single_trials: float = 0.1

# ---------- Graph ----------
@dataclass(frozen=True)
class GraphConfig:
    graph_preset_architecture: int = 601
    num_label_spots: int = 5
    grayscale_levels: int = 1
    torus: bool = True
    base_graph_manager: Union[str, AbstractBaseGraphManager] = "poisson_binomial_ising_graph_manager"

# ---------- Sampling (Gibbs/CD) schedule ----------
@dataclass(frozen=True)
class SamplingScheduleConfig:
    batch_size: int = 400
    n_samples: int = 50
    steps_per_sample: int = 8
    steps_warmup: int = 400
    training_beta: float = 1.0

# --- Image Generation ---
@dataclass(frozen=True)
class GenerateImagesConfig:
    generation_beta_start: float = 0.8
    generation_beta_end: float = 1.2
    fid_images_per_digit: int = 512
    steps_warmup: int = 600 # the steps before taking the sample for generation

# ---------- Diffusion schedule (time grid) ----------
@dataclass(frozen=True)
class DiffusionScheduleConfig:
    num_diffusion_steps: int = 1
    kind: Literal["linear", "log"] = "log"
    diffusion_offset: float = 0.1

# ---------- Diffusion rates (forward/noising) ----------
@dataclass(frozen=True)
class DiffusionRatesConfig:
    image_rate: float = .8
    label_rate: float = .2

# ---------- Optim / LR decay ----------
@dataclass(frozen=True)
class OptimConfig:
    momentum: float = 0.9
    b2_adam: float = 0.999
    step_learning_rates: Tuple[float, ...] = (0.05,) # per-step learning rates
    alpha_cosine_decay: float = .2
    n_epochs_for_lrd: int = 50 # span (in epochs) used to compute cosine LR decay length

# ---------- Regularization: Correlation Penalty ----------
@dataclass(frozen=True)
class CorrelationPenaltyConfig:
    correlation_penalty: Tuple[float, ...] = (0.0,)
    adaptive_cp: bool = False
    cp_min: float = 0.001
    adaptive_threshold: float = 0.016  # autocorr threshold, also used for adaptive weight decay

# ---------- Regularization: Weight Decay ----------
@dataclass(frozen=True)
class WeightDecayConfig:
    weight_decay: Tuple[float, ...] = (0.0,)
    adaptive_wd: bool = False
    wd_min: float = 0.001

# ---------- Meta / run ----------
@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    graph_seeds: Tuple[int, ...] = ()
    descriptor: str = ""
    n_cores: int = 1
    compute_autocorr: bool = True # in order to not compute autocorrelation, adaptive_wd, adaptive_wd and compute autocorr must all be false
    generate_gif: bool = True
    drawn_images_per_digit: int = 4
    animated_images_per_digit: int = 2
    steps_per_sample_in_gif: int = 10

# ---------- Top-level ----------
@dataclass(frozen=True)
class DTMConfig:
    exp: ExperimentConfig = ExperimentConfig()
    data: DataConfig = DataConfig()
    graph: GraphConfig = GraphConfig()
    sampling: SamplingScheduleConfig = SamplingScheduleConfig()
    generation: GenerateImagesConfig = GenerateImagesConfig()
    diffusion_schedule: DiffusionScheduleConfig = DiffusionScheduleConfig()
    diffusion_rates: DiffusionRatesConfig = DiffusionRatesConfig()
    optim: OptimConfig = OptimConfig()
    cp: CorrelationPenaltyConfig = CorrelationPenaltyConfig()
    wd: WeightDecayConfig = WeightDecayConfig()

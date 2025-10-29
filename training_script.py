"""
A vanilla training script for a DTM. All hyperparameters are prepopulated with their defaults from DTM_config.py, but can be easily changed here.
Each section of hyperparameters is grouped into a dict corresponding to the dataclass in DTM_config.py.
After setting the dicts, the script creates a DTMConfig using make_cfg with these dicts, initializes the DTM, and trains it.
"""

from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg

# ---------- Data parameters ----------
"""
dataset_name: Name of the dataset to train and evaluate on. Supported values include "mnist", "fashion_mnist", or "smoke_testing_<n_image_pixels>_<n_grayscale_levels>" for toy datasets.
target_classes: Specifies which classes from the dataset to include for training and generation.
pixel_threshold_for_single_trials: Threshold for binarizing images when grayscale_levels=1 (values above this become 1, else 0).
"""
data_params = {
    "dataset_name": "mnist",  # Literal["mnist", "fashion_mnist", "smoke_testing"]
    "target_classes": tuple(range(4)),  # Tuple[int, ...]
    "pixel_threshold_for_single_trials": 0.1,  # float
}

# ---------- Graph parameters ----------
"""
graph_preset_architecture: Preset architecture for the base graph, defined as an integer combining grid side length and degree (e.g., 6012 for side=60, degree=12).
num_label_spots: Number of times to repeat the one-hot label encoding in the Ising graph.
grayscale_levels: Number of grayscale levels above 0 for image encoding.
torus: If True, wraps edges around the square grid for toroidal topology.
base_graph_manager: Specifies the graph manager class from base_graphs/ for constructing hidden and output nodes (e.g., "poisson_binomial_ising_graph_manager").
"""
graph_params = {
    "graph_preset_architecture": 60_12,  # int
    "num_label_spots": 5,  # int
    "grayscale_levels": 1,  # int
    "torus": True,  # bool
    "base_graph_manager": "poisson_binomial_ising_graph_manager",  # str
}

# ---------- Sampling (Gibbs/CD) schedule parameters ----------
"""
batch_size: Batch size used during training.
n_samples: Number of samples to collect in the contrastive divergence negative phase.
steps_per_sample: Number of Gibbs steps between each collected sample in training.
steps_warmup: Number of initial warmup Gibbs steps before sampling in training.
training_beta: Inverse temperature (beta) used during training; typically set to 1.0 for training.
"""
sampling_params = {
    "batch_size": 400,  # int
    "n_samples": 50,  # int
    "steps_per_sample": 8,  # int
    "steps_warmup": 400,  # int
    "training_beta": 1.0,  # float
}

# ---------- Image Generation parameters ----------
"""
generation_beta_start: Starting value for the linearly increasing beta schedule in image generation annealing.
generation_beta_end: Ending value for the linearly increasing beta schedule in image generation annealing.
fid_images_per_digit: Number of images to generate per digit/class for FID computation; also used as batch size per digit for smoke-testing accuracy calculations.
steps_warmup: Number of Gibbs steps before taking the final sample during generation.
"""
generation_params = {
    "generation_beta_start": 0.8,  # float
    "generation_beta_end": 1.2,  # float
    "fid_images_per_digit": 512,  # int
    "steps_warmup": 600,  # int
}

# ---------- Diffusion schedule (time grid) parameters ----------
"""
num_diffusion_steps: Number of diffusion steps (and thus independent models) in the DTM; if 1 with a log schedule, effectively reduces to a single bipartite Boltzmann machine as the infinite time step sends couplings to zero.
kind: Type of noising schedule ("linear" or "log").
diffusion_offset: Offset applied to diffusion times for adjusting perturbation probabilities.
"""
diffusion_schedule_params = {
    "num_diffusion_steps": 1,  # int
    "kind": "log",  # Literal["linear", "log"]
    "diffusion_offset": 0.1,  # float
}

# ---------- Diffusion rates (forward/noising) parameters ----------
"""
image_rate: Diffusion rate applied to noising image nodes.
label_rate: Diffusion rate applied to noising label nodes.
"""
diffusion_rates_params = {
    "image_rate": 0.8,  # float
    "label_rate": 0.2,  # float
}

# ---------- Optim / LR decay parameters ----------
"""
momentum: Beta1 parameter for Adam optimizer.
b2_adam: Beta2 parameter for Adam optimizer.
step_learning_rates: Learning rates for each diffusion step; if fewer than num_diffusion_steps, the last value is repeated to extend the tuple.
alpha_cosine_decay: Alpha parameter for cosine learning rate decay schedule.
n_epochs_for_lrd: Number of epochs over which to apply cosine learning rate decay.
"""
optim_params = {
    "momentum": 0.9,  # float
    "b2_adam": 0.999,  # float
    "step_learning_rates": (0.05,),  # Tuple[float, ...]  # per-step learning rates
    "alpha_cosine_decay": 0.2,  # float
    "n_epochs_for_lrd": 50,  # int  # span (in epochs) used to compute cosine LR decay length
}

# ---------- Regularization: Correlation Penalty parameters ----------
"""
correlation_penalty: Per-step initial correlation penalty coefficients; extended to match num_diffusion_steps if shorter.
adaptive_cp: If True, enables adaptive correlation penalty based on measured autocorrelation.
cp_min: Minimum correlation penalty value when adaptive_cp is True.
adaptive_threshold: Autocorrelation threshold for triggering adaptive increases; also used in adaptive weight decay.
"""
cp_params = {
    "correlation_penalty": (0.0,),  # Tuple[float, ...]
    "adaptive_cp": False,  # bool
    "cp_min": 0.001,  # float
    "adaptive_threshold": 0.016,  # float  # autocorr threshold, also used for adaptive weight decay
}

# ---------- Regularization: Weight Decay parameters ----------
"""
weight_decay: Per-step weight decay coefficients; extended to match num_diffusion_steps if shorter (not used in original experiments but included for future use).
adaptive_wd: If True, enables adaptive weight decay based on measured autocorrelation.
wd_min: Minimum weight decay value when adaptive_wd is True.
"""
wd_params = {
    "weight_decay": (0.0,),  # Tuple[float, ...]
    "adaptive_wd": False,  # bool
    "wd_min": 0.001,  # float
}

# ---------- Meta / run parameters ----------
"""
seed: Main seed for initializing most JAX random keys in the model.
graph_seeds: Per-step seeds for deterministic graph construction; if empty tuple, defaults to (seed + i for i in range(num_diffusion_steps)). Ensures consistent node placement for weight/bias alignment during model loading.
descriptor: Descriptive prefix for the training run, used in logging and saving paths.
n_cores: Number of GPUs to use for parallel autocorrelation computation.
compute_autocorr: If True, computes autocorrelation during training; must be False to disable if adaptive_cp or adaptive_wd is False.
generate_gif: If True, generates animated GIFs of the sampling process during evaluation epochs.
drawn_images_per_digit: Number of images per digit/class to include in saved PNG grids during evaluation.
animated_images_per_digit: Number of images per digit/class to animate in GIFs if generate_gif is True.
steps_per_sample_in_gif: Number of Gibbs steps between frames in generated GIFs.
"""
exp_params = {
    "seed": 42,  # int
    "graph_seeds": (),  # Tuple[int, ...]
    "descriptor": "",  # str
    "n_cores": 1,  # int
    "compute_autocorr": True,  # bool
    "generate_gif": True,  # bool
    "drawn_images_per_digit": 4,  # int
    "animated_images_per_digit": 2,  # int
    "steps_per_sample_in_gif": 10,  # int
}

# ---------- Training parameters (not part of DTMConfig) ----------
n_epochs = 50  # int  # Number of epochs to train for
evaluate_every = 1  # int  # Evaluate and save every N epochs (set to 0 to disable)

# Build the config using the parameter dicts above
cfg = make_cfg(
    exp=exp_params,
    data=data_params,
    graph=graph_params,
    sampling=sampling_params,
    generation=generation_params,
    diffusion_schedule=diffusion_schedule_params,
    diffusion_rates=diffusion_rates_params,
    optim=optim_params,
    cp=cp_params,
    wd=wd_params,
)

# Create and train the DTM
dtm = DTM(cfg)
dtm.train(n_epochs, evaluate_every)
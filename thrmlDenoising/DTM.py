from __future__ import annotations

from dataclasses import asdict, replace
from typing import Optional

import concurrent.futures
import datetime
import os
import pprint
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml

from thrml.block_sampling import SamplingSchedule
from thrmlDenoising.DTM_config import DTMConfig
from thrmlDenoising.base_graphs.abstract_base_graph_manager import (
    AbstractBaseGraphManager,
)
from thrmlDenoising.fid.fid import bootstrap_fid_fn
from thrmlDenoising.sampling_specs import get_new_per_block_interactions
from thrmlDenoising.smoke_testing import (
    compute_conditional_accuracy,
    compute_free_accuracy,
)
from thrmlDenoising.step import DiffusionStep
from thrmlDenoising.utils import (
    adapt_param,
    config_to_yaml_dict,
    denoise_arrays_to_gif,
    draw_image_batch,
    extend_params_or_zeros,
    load_dataset,
    load_yaml_config_from_dict,
    write,
)

class DTM:
    """Denoising Thermodynamic Model (DTM) for generative tasks using Ising-based EBMs.

    Orchestrates multiple diffusion steps to form a reverse diffusion chain for generation.
    Handles training across steps in parallel, evaluation with FID or accuracy metrics,
    autocorrelation monitoring for adaptive regularization, model saving/loading, and
    visualization utilities like image grids and GIFs.

    **Attributes:**
    - `cfg`: Configuration dataclass guiding model setup and hyperparameters.
    - `steps`: List of DiffusionStep instances, one per diffusion interval.
    - `n_image_pixels`: Number of pixels per image in the dataset.
    - `n_label_nodes`: Number of nodes used for labels.
    - `n_grayscale_levels`: Number of grayscale levels in images.
    - `train_dataset`: Dictionary with training images and labels.
    - `test_dataset`: Dictionary with test images and labels.
    - `one_hot_target_labels`: One-hot encoded target classes for conditional generation.
    - `logging_and_saving_dir`: Directory for logs, saves, and outputs.
    - `log_file`: Path to training log file.
    - `model_saving_path`: Path for saving model checkpoints.
    - `fids_dict`: Dictionary tracking FID scores for free/clamped generation.
    - `eval_epochs`: List of epochs where evaluation was performed.
    - `base_graph_manager`: Manager for graph construction and data conversions.
    - `is_smoke_test`: Flag indicating if using smoke test dataset.
    """
    cfg: DTMConfig
    steps: list[DiffusionStep]

    n_image_pixels: int
    n_label_nodes: int
    n_grayscale_levels: int

    train_dataset: dict
    test_dataset: dict

    one_hot_target_labels: Array

    logging_and_saving_dir: str = ''
    log_file: str = ''
    model_saving_path: str = ''
    fids_dict: dict
    eval_epochs: list[int]

    base_graph_manager: AbstractBaseGraphManager

    is_smoke_test: bool

    def __init__(self, cfg: Optional[DTMConfig] = None):
        """Initializes the DTM with configuration, dataset loading, and step setup.

        Loads dataset, selects base graph manager based on config, computes diffusion
        schedule times, and initializes diffusion steps. Seeds are auto-generated if
        not provided in config.

        **Arguments:**
        - `cfg`: Optional DTMConfig instance; defaults to values in DTM_config if None.
        """
        if cfg is None:
            cfg = DTMConfig()

        self.key = jr.PRNGKey(cfg.exp.seed)
        self.fids_dict = {"free": [], "clamped": []}
        self.eval_epochs = []

        self.is_smoke_test = "smoke_testing" in cfg.data.dataset_name

        self.train_dataset, self.test_dataset, self.one_hot_target_labels = load_dataset(
            cfg.data.dataset_name,
            cfg.graph.grayscale_levels,
            cfg.data.target_classes,
            cfg.graph.num_label_spots,
            cfg.data.pixel_threshold_for_single_trials,
        )
        
        self.n_image_pixels = self.train_dataset["image"].shape[1]
        self.n_label_nodes = self.train_dataset["label"].shape[1]

        self.n_grayscale_levels = cfg.graph.grayscale_levels
        # For smoke testing make sure model has sufficient grayscale levels to express smoke test 'images'
        #   and print a warning if n_grayscale is larger than the dataset requires
        if self.is_smoke_test:
            max_pixel_value = jnp.max(self.train_dataset["image"])
            assert max_pixel_value <= self.n_grayscale_levels, (
                f"Model grayscale levels ({self.n_grayscale_levels}) insufficient for data "
                f"(max pixel value: {max_pixel_value})"
            )
            if self.n_grayscale_levels > max_pixel_value:
                write(f"Warning: Model is more expressive than data requires. "
                      f"Model has {self.n_grayscale_levels} grayscale levels but data only uses "
                      f"up to {max_pixel_value}. This will hurt performance.")

            # For smoke testing data always use full dataset length as batch size
            smoke_testing_batch_size = self.train_dataset["image"].shape[0]
            cfg = replace(cfg, sampling=replace(cfg.sampling, batch_size=smoke_testing_batch_size))

        if cfg.graph.base_graph_manager == "poisson_binomial_ising_graph_manager":
            from thrmlDenoising.base_graphs.poisson_binomial_ising_graph_manager import PoissonBinomialIsingGraphManager
            self.base_graph_manager = PoissonBinomialIsingGraphManager(self.n_image_pixels, self.n_label_nodes, self.n_grayscale_levels)
        elif cfg.graph.base_graph_manager == "binary_representation_ising_graph_manager":
            from thrmlDenoising.base_graphs.binary_representation_ising_graph_manager import BinaryRepresentationIsingGraphManager
            self.base_graph_manager = BinaryRepresentationIsingGraphManager(self.n_image_pixels, self.n_label_nodes, self.n_grayscale_levels)
        elif cfg.graph.base_graph_manager == "convolved_poisson_binomial_ising_graph_manager":
            from thrmlDenoising.base_graphs.convolved_poisson_binomial_ising_graph_manager import ConvolvedPoissonBinomialIsingGraphManager
            self.base_graph_manager = ConvolvedPoissonBinomialIsingGraphManager(self.n_image_pixels, self.n_label_nodes, self.n_grayscale_levels)
        elif issubclass(cfg.graph.base_graph_manager, AbstractBaseGraphManager):
            self.base_graph_manager = cfg.graph.base_graph_manager(self.n_image_pixels, self.n_label_nodes, self.n_grayscale_levels)
            cfg = replace(cfg, graph=replace(cfg.graph, base_graph_manager="custom_base_graph_manager"))
        else:
            print(cfg.graph.base_graph_manager)
            raise ValueError("Invalid cfg.graph.base_graph_manager")

        dataset_length = self.train_dataset["image"].shape[0]
        n_batches_per_epoch = dataset_length // cfg.sampling.batch_size

        if cfg.diffusion_schedule.kind == "linear":
            schedule_fn = lambda x: x
        elif cfg.diffusion_schedule.kind == "log":
            schedule_fn = lambda x: -jnp.log(1 - x)
        else:
            raise ValueError(f"Invalid diffusion schedule: {cfg.diffusion_schedule.kind}")

        step_times = schedule_fn(
            jnp.linspace(0.0, 1.0, cfg.diffusion_schedule.num_diffusion_steps + 1, endpoint=True)
        )

        if cfg.exp.graph_seeds:
            assert len(cfg.exp.graph_seeds) == cfg.diffusion_schedule.num_diffusion_steps
        else:
            # If the config does not come with graph seeds, place the seeds to be used for graph in the config for future loading
            auto_seeds = tuple(cfg.exp.seed + i for i in range(cfg.diffusion_schedule.num_diffusion_steps))
            cfg = replace(cfg, exp=replace(cfg.exp, graph_seeds=auto_seeds))

        self.cfg = cfg

        self._init_steps(step_times, n_batches_per_epoch)

    def _init_steps(self,step_times, n_batches_per_epoch):
        """Initializes the list of DiffusionStep instances for the model.

        Sets up training and generation schedules, extends learning rates if needed,
        creates optimizers with cosine decay for each step, and constructs each
        DiffusionStep with its time interval and parameters.

        **Arguments:**
        - `step_times`: Array of time points defining diffusion intervals.
        - `n_batches_per_epoch`: Number of batches per epoch for decay scheduling.
        """

        training_sampling_schedule = SamplingSchedule(
            self.cfg.sampling.steps_warmup,
            self.cfg.sampling.n_samples,
            self.cfg.sampling.steps_per_sample,
        )

        total_decay_steps = self.cfg.optim.n_epochs_for_lrd * n_batches_per_epoch

        learning_rate_per_step = extend_params_or_zeros(
            self.cfg.optim.step_learning_rates, self.cfg.diffusion_schedule.num_diffusion_steps
        )

        # Here we set the same generating betas and schedule across all steps
        #   The generating betas are set to linearly increase from generation_beta_start to generation_beta_end across the genertaion steps
        #   The generation schedule is the dummy schedule used for generation, when only one sample is taken after generation.steps_warmup steps
        generating_betas = jnp.linspace(self.cfg.generation.generation_beta_start, self.cfg.generation.generation_beta_end, self.cfg.generation.steps_warmup)
        generation_schedule = SamplingSchedule(self.cfg.generation.steps_warmup,1,-1)

        steps = []
        for i in range(self.cfg.diffusion_schedule.num_diffusion_steps):
            graph_key = jr.PRNGKey(self.cfg.exp.graph_seeds[i])

            lrd_schedule = optax.cosine_decay_schedule(
                init_value=jnp.float32(learning_rate_per_step[i]),
                decay_steps=total_decay_steps,
                alpha=self.cfg.optim.alpha_cosine_decay,
            )

            optim = optax.adam(learning_rate=lrd_schedule, b1=self.cfg.optim.momentum, b2=self.cfg.optim.b2_adam)

            step = DiffusionStep(
                step_times[i],
                step_times[i+1],
                self.n_grayscale_levels,
                self.n_image_pixels,
                self.n_label_nodes,
                self.cfg.diffusion_rates.image_rate,
                self.cfg.diffusion_rates.label_rate,
                self.cfg.diffusion_schedule.diffusion_offset,
                training_sampling_schedule,
                self.cfg.sampling.training_beta,
                generation_schedule,
                generating_betas,
                self.cfg.graph.graph_preset_architecture,
                self.cfg.graph.torus,
                optim,
                graph_key,
                self.base_graph_manager,
            )
            steps.append(step)
        self.steps = steps
    
    def train(self, n_epochs, evaluate_every):
        """Trains the DTM over multiple epochs with parallel step updates.

        Handles logging setup, autocorrelation computation for adaptive regularization,
        parameter updates using concurrent execution across devices, and periodic
        evaluation. Supports smoke testing mode with accuracy metrics.

        **Arguments:**
        - `n_epochs`: Total number of training epochs.
        - `evaluate_every`: Interval (in epochs) for evaluation and saving.
        """
        start_training_time = time.time()

        cp_coeffs = extend_params_or_zeros(self.cfg.cp.correlation_penalty, len(self.steps))
        wd_coeffs = extend_params_or_zeros(self.cfg.wd.weight_decay, len(self.steps))

        if evaluate_every:
            descriptor = f"{self.cfg.exp.descriptor}_{self.cfg.diffusion_schedule.num_diffusion_steps}_step_{self.cfg.graph.graph_preset_architecture}_grid"
            if self.cfg.cp.adaptive_cp:
                descriptor += "_with_adaptive_cp"
            if self.is_smoke_test:
                descriptor = "smoke_test_" + descriptor

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.logging_and_saving_dir = os.path.join(
                "model_logging_and_saving/", f"{descriptor}_{timestamp}"
            )
            os.makedirs(self.logging_and_saving_dir, exist_ok=False)
            self.log_file = os.path.join(self.logging_and_saving_dir, f"training_log_{descriptor}.txt")
            # write all of cfg to logging file
            config_str = pprint.pformat(asdict(self.cfg), indent=2)
            write("Configuration:\n" + config_str + "\n", self.log_file)

        devices = jax.devices("gpu")
        n_devices = len(devices)

        def train_one_step(step, key, image_data, label_data, cp_coeff, wd_coeff):
            return step.train_step_model(
                key, self.cfg.sampling.batch_size, image_data, label_data, cp_coeff, wd_coeff, None
            )

        # these jitted training functions should only trace once per step
        train_fns = [
            eqx.filter_jit(fun=train_one_step, device=devices[i % n_devices])
            for i in range(len(self.steps))
        ]

        def thread_worker(i, step, key, image_data, label_data, cp_coeff, wd_coeff):
            return train_fns[i](step, key, image_data, label_data, cp_coeff, wd_coeff)

        # compute autocorr at epoch 0 for autocorr comparison at epoch 1
        if self.cfg.cp.adaptive_cp or self.cfg.wd.adaptive_wd or self.cfg.exp.compute_autocorr:
            autocorr_comp_start_time = time.time()
            self.compute_autocorr(epoch=0)
            acs_str = f"autocorrelations at epoch {0}: "
            for i, step in enumerate(self.steps): 
                acs_str += f"step {i}: {step.autocorrelations[0]} "
            write(acs_str + "\n", self.log_file)
            write(f"Time to compute autocorrelations: {time.time() - autocorr_comp_start_time:.1f}s\n")

        # evaluate model at epoch 0 for reference
        if evaluate_every:
            eval_start_time = time.time()
            self.eval_epoch(0)
            if not self.is_smoke_test:
                write(f"Time to evaluate epoch {0}: {(time.time() - eval_start_time):.1f}\n")

        for epoch in range(1, n_epochs + 1):
            keys = jr.split(self._get_new_key(), len(self.steps))
            epoch_training_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        thread_worker, i, self.steps[i], keys[i], self.train_dataset["image"], self.train_dataset["label"], cp_coeffs[i], wd_coeffs[i]
                    )
                    for i in range(len(self.steps))
                ]
                results = [f.result() for f in futures]

            for i, result in enumerate(results):
                new_step_weights, new_step_biases, new_opt_state = result

                old = self.steps[i]

                new_pos_per_block_interact_train = get_new_per_block_interactions(old.training_spec.program_positive, new_step_weights, new_step_biases)
                new_neg_per_block_interact_train = get_new_per_block_interactions(old.training_spec.program_negative, new_step_weights, new_step_biases)

                new_free_per_block_interact_gen = get_new_per_block_interactions(old.generation_spec.program_free, new_step_weights, new_step_biases)
                new_cond_per_block_interact_gen = get_new_per_block_interactions(old.generation_spec.program_conditioned, new_step_weights, new_step_biases)

                # Wow this is ugly, but it lets us jit _run_denoising- I think what matters is that we eqx.tree_at arrays from only the top python object, the DiffusionSteps here
                self.steps[i] = eqx.tree_at(
                    lambda s: (s.model.weights, s.model.biases, s.opt_state, s.training_spec.program_positive.per_block_interactions, s.training_spec.program_negative.per_block_interactions, s.generation_spec.program_free.per_block_interactions, s.generation_spec.program_conditioned.per_block_interactions),
                    old, (new_step_weights, new_step_biases, new_opt_state, new_pos_per_block_interact_train, new_neg_per_block_interact_train, new_free_per_block_interact_gen, new_cond_per_block_interact_gen)
                )

            if not self.is_smoke_test:
                write(f"Time to train epoch {epoch}: {time.time() - epoch_training_start_time:.1f}s\n")

            if self.cfg.cp.adaptive_cp or self.cfg.wd.adaptive_wd or self.cfg.exp.compute_autocorr:
                autocorr_comp_start_time = time.time()
                self.compute_autocorr(epoch=epoch)
                acs_str = f"autocorrelations at epoch {epoch}: "
                for i, step in enumerate(self.steps): 
                    acs_str += f"step {i}: {step.autocorrelations[epoch]} "
                write(acs_str + "\n", self.log_file)
                write(f"Time to compute autocorrelations: {time.time() - autocorr_comp_start_time:.1f}s\n")
            
            if self.cfg.cp.adaptive_cp:
                cp_str = "correlation penalties: "
                for i, step in enumerate(self.steps):
                    assert (epoch - 1) in step.autocorrelations and epoch in step.autocorrelations
                    step_cp = jnp.maximum(cp_coeffs[i], self.cfg.cp.cp_min)
                    step_cp = adapt_param(step.autocorrelations[epoch], step.autocorrelations[epoch - 1], step_cp, self.cfg.cp.adaptive_threshold)
                    if step_cp < self.cfg.cp.cp_min:
                        step_cp = 0.0
                    cp_coeffs = cp_coeffs.at[i].set(step_cp)
                    cp_str += f"step {i}: {cp_coeffs[i]} "
                write(cp_str + "\n", self.log_file)

            if self.cfg.wd.adaptive_wd:
                wd_str = " weight decays: "
                for i, step in enumerate(self.steps):
                    assert (epoch - 1) in step.autocorrelations and epoch in step.autocorrelations
                    step_wd = jnp.maximum(wd_coeffs[i], self.cfg.wd.wd_min)
                    step_wd = adapt_param(step.autocorrelations[epoch], step.autocorrelations[epoch - 1], step_wd, self.cfg.cp.adaptive_threshold)
                    if step_wd < self.cfg.cp.cp_min:
                        step_wd = 0.0
                    wd_coeffs = wd_coeffs.at[i].set(step_wd)
                    wd_str += f"step {i}: {wd_coeffs[i]} "
                write(wd_str + "\n", self.log_file)

            if evaluate_every and epoch % evaluate_every == 0:
                eval_start_time = time.time()
                self.eval_epoch(epoch)
                if not self.is_smoke_test:
                    write(f"Time to evaluate epoch {epoch}: {(time.time() - eval_start_time):.1f}\n")

        write(f"total training time: {(time.time() - start_training_time):.0f}")

    def _run_denoising(self, key, free: bool, batch_size: int, schedule: SamplingSchedule):
        """Runs the full reverse diffusion chain to generate samples.

        Initializes inputs with noise, then iteratively denoises over reversed steps,
        vmapping over conditions (labels) for batched generation. Supports free or
        conditional (clamped labels) modes.

        **Arguments:**
        - `key`: PRNG key for randomness in initialization and sampling.
        - `free`: If True, free generation (no label clamping); else conditional.
        - `batch_size`: Number of samples per condition.
        - `schedule`: Sampling schedule for each denoise call.

        **Returns:**
        - Tuple of image readout list and label readout list, each a list over steps
          with arrays of shape `(n_labels, batch_size, n_samples, size)`.
        """
        
        def get_in_data(key_in_data, block):
            return jr.bernoulli(key_in_data, 0.5, (batch_size, len(block)))

        input_key, key = jr.split(key, 2)
        input_data = get_in_data(input_key, self.steps[0].generation_spec.input_block)

        n_labels = self.one_hot_target_labels.shape[0]
        if free:
            label_out = None  # stays None for free gen
        else:
            assert self.one_hot_target_labels.shape == (n_labels, self.base_graph_manager.n_label_nodes)
            label_out = jnp.broadcast_to(self.one_hot_target_labels[:, None, :], (n_labels, batch_size, self.base_graph_manager.n_label_nodes))
        input_in = jnp.broadcast_to(input_data[None, ...], (n_labels,) + input_data.shape)

        image_readout_list, label_readout_list = [], []
        step_keys = jr.split(key, len(self.steps))
        for i, step in enumerate(reversed(self.steps)):
            step_device = step.model.weights.device

            # Move clamps to the step's device
            input_in = jax.device_put(input_in, step_device)
            if label_out is not None:
                label_out = jax.device_put(label_out, step_device)

            # vmap over condition axis; each slice runs a single-batch denoise with (batch_size, block_len) arrays
            image_outs, label_outs = single_step_denoise(step, step_keys[i], free, n_labels, input_in, label_out, schedule)

            # Assert all image and label output block shapes match expected shapes
            assert all(arr.shape == (n_labels, batch_size, schedule.n_samples, self.base_graph_manager.image_output_block_lengths[i]) for i, arr in enumerate(image_outs))
            assert all(arr.shape == (n_labels, batch_size, schedule.n_samples, self.base_graph_manager.label_output_block_lengths[i]) for i, arr in enumerate(label_outs))

            # Append converted from output blocks to readout lists
            converted_image_outs = self.base_graph_manager.convert_output_blocks_to_pixels(image_outs)
            converted_label_outs = self.base_graph_manager.convert_label_out_blocks_to_label(label_outs)

            assert converted_image_outs.shape == (n_labels, batch_size, schedule.n_samples, self.base_graph_manager.n_image_pixels)
            assert converted_label_outs.shape == (n_labels, batch_size, schedule.n_samples, self.base_graph_manager.n_label_nodes)

            image_readout_list.append(converted_image_outs)
            label_readout_list.append(converted_label_outs)

            # Only pass the final sample to next diffusion step's input block
            final_sampled_image = converted_image_outs[:, :, -1, :]
            final_sampled_label = converted_label_outs[:, :, -1, :]
            input_in = self.base_graph_manager.convert_pixels_and_labels_to_input_block(final_sampled_image, final_sampled_label)

        return image_readout_list, label_readout_list

    def gen_images(
        self,
        key,
        batch_size,
        free,
        drawn_images_per_digit,
    ):
        """Generates images using the reverse diffusion process.

        Runs denoising chain, extracts final samples across steps, normalizes images,
        and prepares batches for drawing or FID computation. Supports free or conditional modes.

        **Arguments:**
        - `key`: PRNG key for generation randomness.
        - `batch_size`: Samples per digit/condition.
        - `free`: If True, free generation; else conditional on target labels.
        - `drawn_images_per_digit`: Number of images per digit for drawing.

        **Returns:**
        - Tuple of image grid array, label grid array, and final images for FID.
        """
        n_digits = self.one_hot_target_labels.shape[0]

        all_images, all_labels = self._run_denoising(
            key,
            free,
            batch_size,
            self.steps[0].generation_spec.schedule,
        )

        def take_last_and_normalise(x):
            assert x.shape == (
                n_digits,
                batch_size,
                self.steps[0].generation_spec.schedule.n_samples,
                self.n_image_pixels,
            )
            x = x[:, :, -1, :]
            return jnp.array(x, dtype=jnp.float32) / self.n_grayscale_levels
        all_images = jtu.tree_map(take_last_and_normalise, all_images)

        def take_last(x):
            assert x.shape == (
                n_digits,
                batch_size,
                self.steps[0].generation_spec.schedule.n_samples,
                self.n_label_nodes,
            )
            x = x[:, :, -1, :]
            return x
        all_labels = jtu.tree_map(take_last, all_labels)
        
        common_device = jax.devices()[0]
        all_images = [jax.device_put(img, common_device) for img in all_images]
        all_labels = [jax.device_put(lbl, common_device) for lbl in all_labels]

        # For FID, extract images from the final step in the list of steps
        images_for_fid = all_images[-1]
        expected_image_shape = (n_digits, batch_size, self.n_image_pixels)
        expected_label_shape = (n_digits, batch_size, self.n_label_nodes)
        for i, (images, labels) in enumerate(zip(all_images, all_labels)):
            assert images.shape == expected_image_shape, f"{images.shape} != {expected_image_shape}"
            assert labels.shape == expected_label_shape, f"{labels.shape} != {expected_label_shape}"
            images = images[:, :drawn_images_per_digit, :]
            labels = labels[:, :drawn_images_per_digit, :]
            all_images[i] = images.reshape(
                n_digits, 1, drawn_images_per_digit, self.n_image_pixels
            )
            all_labels[i] = labels.reshape(
                n_digits, 1, drawn_images_per_digit, self.n_label_nodes
            )

        to_draw = jnp.concatenate(all_images, axis=1)
        to_draw_labels = jnp.concatenate(all_labels, axis=1)
        to_draw = jnp.reshape(
            to_draw, (n_digits * len(self.steps), drawn_images_per_digit, self.n_image_pixels)
        )
        to_draw_labels = jnp.reshape(
            to_draw_labels, (n_digits * len(self.steps), drawn_images_per_digit, self.n_label_nodes)
        )
        return to_draw, to_draw_labels, images_for_fid
    
    def do_draw_and_fid(
            self, 
            key,
            filename, 
            free,
    ):
        """Generates, draws, and evaluates images with FID for free or clamped mode.

        Calls `gen_images` to produce samples, draws a grid of selected images,
        saves the figure, and computes FID using precomputed dataset statistics.

        **Arguments:**
        - `key`: PRNG key for generation.
        - `filename`: Base filename for saved image grid.
        - `free`: If True, free generation; else conditional.

        **Returns:**
        - Computed FID score for the generated images.
        """
        drawn_images_per_digit = self.cfg.exp.drawn_images_per_digit
        image_size = self.train_dataset["image"].shape[1]
        to_draw, _, images_for_fid = self.gen_images(key, self.cfg.generation.fid_images_per_digit, free, drawn_images_per_digit)
        to_draw = jnp.reshape(to_draw, (-1, image_size))
        assert (
            to_draw.shape[0]
            == self.one_hot_target_labels.shape[0] * len(self.steps) * drawn_images_per_digit
        )
        to_draw = jnp.clip(to_draw, 0, 1)

        fig = draw_image_batch(
            to_draw,
            self.one_hot_target_labels.shape[0] * len(self.steps),
            drawn_images_per_digit,
            super_columns=2 if self.one_hot_target_labels.shape[0] > 4 else 1,
        )
        desc_str = "free" if free else "clamped"
        fig.savefig(f"{filename}_{desc_str}.png")

        # the npz file with precomputed stats for fid eval assumed to be in 
        #   thrmlDenoising/fid/precomputed_stats/{'gs' grayscale or 'bw' black and white}_train.npz
        image_mode = "gs" if self.n_grayscale_levels > 1 else "bw"
        data_name = f"{image_mode}_{self.cfg.data.dataset_name}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fid_precomputed_stats_path = os.path.join(script_dir, 'fid', 'precomputed_stats', f"{data_name}_train.npz")
        fid, _, _ = bootstrap_fid_fn(images_for_fid, fid_precomputed_stats_path)
        return fid
    
    def generate_gif(
        self,
        epoch: Optional[int] = None,
        runs_per_label: int = 2,
        steps_per_sample: int = 10,
        frame_stride: int = 2,
        fps: int = 800,
    ):
        """Generates and saves GIFs showing the denoising process for free and clamped modes.

        Runs denoising with sample retention, creates GIFs visualizing evolution across
        steps and within-step sampling.

        **Arguments:**
        - `epoch`: Optional epoch number for naming; if None, uses generic name.
        - `steps_per_sample`: Gibbs steps between saved samples in schedule.
        - `runs_per_label`: Number of runs (columns) per label in GIF grid.
        - `frame_stride`: Stride to thin frames for shorter GIFs.
        - `fps`: Frames per second for GIF playback.
        """
        start_gif_time = time.time()

        key = self._get_new_key()

        if self.is_smoke_test:
            raise ValueError("Gif not supported for smoke tests.")
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.logging_and_saving_dir:
            descriptor = f"{self.cfg.exp.descriptor}_{self.cfg.diffusion_schedule.num_diffusion_steps}_step_{self.cfg.graph.graph_preset_architecture}_grid"
            if self.cfg.cp.adaptive_cp:
                descriptor += "_with_adaptive_cp"
            self.logging_and_saving_dir = os.path.join(
                "model_logging_and_saving/", f"{descriptor}_{timestamp}"
            )
            os.makedirs(self.logging_and_saving_dir, exist_ok=False)
        gif_folder = os.path.join(self.logging_and_saving_dir, "gifs")
        if epoch is not None:
            file_name_without_freeness = (
                f"epoch{epoch}_gifs"
            )
        else:
            file_name_without_freeness = (
                f"gifs"
            )

        n_samples = self.steps[0].generation_spec.schedule.n_warmup // steps_per_sample #run for a total time of warmup in generation spec
        schedule = SamplingSchedule(0, n_samples, steps_per_sample)

        side = int(round(np.sqrt(self.n_image_pixels)))
        assert side * side == self.n_image_pixels, "self.n_image_pixels must be a perfect square for gif generation."

        # Run denoising with per-sample retention for GIF frames
        # image_readout_list: list over steps, each of shape (n_labels, batch_size, num_samples, image_block_len)
        for free in [True, False]:
            gif_file = os.path.join(gif_folder, f"{file_name_without_freeness}_{'free' if free else 'clamped'}.gif")

            image_readout_list, label_readout_list = self._run_denoising(
                key,
                free,
                runs_per_label,
                schedule,
            )

            denoise_arrays_to_gif(
                image_readout_list,
                gif_file,
                n_grayscale_levels=self.n_grayscale_levels,
                runs_per_label=runs_per_label,
                frame_stride=frame_stride,
                fps=fps,
                image_side_len=side,
                pad=1,
                label_readout_list=label_readout_list,
                enable_label_bars=True,
                steps_per_sample=steps_per_sample,
            )

        write(f"Time to generate gifs: {time.time() - start_gif_time:.1f}s\n")

    
    def compute_autocorr(
        self,
        epoch = None,
        n_samples: Optional[int] = None,
    ):
        """Computes autocorrelation for all steps as a mixing time proxy.

        Uses test dataset for perturbation and sampling, encodes samples to lower
        dimension, computes autocorrelation vector, and stores mean tail value.

        **Arguments:**
        - `epoch`: Epoch number to associate with results (optional).
        - `n_samples`: Number of samples per chain; defaults to 800 if None.
        """
        assert self.test_dataset, "Test dataset required for autocorrelation computation."
        assert self.test_dataset["image"].shape == (1000, self.n_image_pixels), (
            f"Expected test_images shape (1000, {self.n_image_pixels}), got {self.test_dataset['image'].shape}"
        )
        assert self.test_dataset["label"].shape == (1000, self.n_label_nodes), (
            f"Expected test_labels shape (1000, {self.n_label_nodes}), got {self.test_dataset['label'].shape}"
        )

        # Some hardcoded values for autocorrelation that have been found to work with mnist and fashion_mnist   
        #   We use a total number of 64 different images from testing. 
        #   n_reps is computed dynamically to pmap over n_cores many cores for an approximate total of 64
        #   Each image is run on 16 independent chains, sampled schedule.n_sample's many times

        #   We sample every 4 steps, and take approx a fourth len of warmup many samples
        #   Projecting the 28x28 image nodes down to 32 latent vars with a random matrix has been 
        #   found to be sufficient to retain autocorrelation information +/- .1

        n_reps = 64 // self.cfg.exp.n_cores
        n_chains = 16

        steps_per_sample_ac = 4
        if not n_samples:
            n_samples = self.cfg.generation.steps_warmup
        n_warmup = self.cfg.sampling.steps_warmup // 2
        schedule = SamplingSchedule(n_warmup, n_samples, steps_per_sample_ac)

        encoded_dim = 32

        random_matrix = jr.uniform(
            self._get_new_key(), (self.n_image_pixels, encoded_dim), jnp.float32, -1.0, 1.0
        ) / jnp.sqrt(self.n_image_pixels)

        for i, step in enumerate(self.steps):
            step.compute_autocorr(
                self._get_new_key(),
                self.cfg.exp.n_cores,
                n_reps,
                n_chains,
                self.test_dataset["image"],
                self.test_dataset["label"],
                schedule,
                random_matrix,
                encoded_dim,
                epoch,
            )


    def save_autocorr_plot(self, epoch):
        """Saves plots of autocorrelation vs epoch (all steps) and vs lag (current epoch).
        autocorrelation vs epoch plot overwrite earlier epochs. Assumes complete autocorrelation data across epochs
        for vs epoch plot. Saves to logging directory.

        **Arguments:**
        - `epoch`: Current epoch for lag plot.
        """

        epochs = sorted(self.steps[0].autocorrelations.keys())
        xs = np.asarray(epochs, dtype=int)

        plt.figure()
        for i, step in enumerate(self.steps):
            # Fast path: build a dense array since we assume complete data.
            assert len(step.autocorrelations) == len(epochs), f"step {i} does not have all autocorrelations in its dict"
            ys = np.asarray([step.autocorrelations[e] for e in epochs], dtype=float)
            plt.plot(xs, ys, marker="o", label=f"step {i}")

        plt.xlabel("Epoch")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation vs Epoch (all steps)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        autocorr_vs_epoch_filename = os.path.join(self.logging_and_saving_dir, "autocorr_vs_epoch.png")
        plt.savefig(autocorr_vs_epoch_filename, dpi=150)
        plt.close()

        # Now create the autocorrelation vs lag plot for this specific epoch
        lags_dir = os.path.join(self.logging_and_saving_dir, "autocorrelation_vs_lags")
        os.makedirs(lags_dir, exist_ok=True)
        autocorr_vs_lags_filename = os.path.join(lags_dir, f"auto_vs_lag_epoch_{epoch}.png")
        plt.figure()
        for i, step in enumerate(self.steps):
            if epoch in step.autocorrelation_vectors:
                vec = step.autocorrelation_vectors[epoch]
                lags = np.arange(len(vec)) * 4
                plt.plot(lags, vec, marker=".", label=f"step {i}")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title(f"Autocorrelation vs Lag at Epoch {epoch}")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(autocorr_vs_lags_filename, dpi=150)
        plt.close()


    def save_fid_plot(self):
        """Overwrite a single PNG of 1/FID vs epoch."""
        xs = self.eval_epochs
        ys_free = np.array(self.fids_dict.get("free", []), dtype=float)
        ys_clamp = np.array(self.fids_dict.get("clamped", []), dtype=float)

        def inv(arr):
            return np.where(arr > 0.0, 1.0 / arr, np.nan)

        plt.figure()
        if ys_free.size:
            plt.plot(xs, inv(ys_free), marker="o", label="free")
        if ys_clamp.size:
            plt.plot(xs, inv(ys_clamp), marker="o", label="clamped")
        plt.xlabel("Epoch")
        plt.ylabel("1 / FID")
        plt.title("1/FID vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.logging_and_saving_dir, "fid_vs_epoch.png"), dpi=150)
        plt.close()


    def eval_epoch(
        self,
        epoch,
    ):
        self.save_epoch(epoch)

        if self.is_smoke_test:
            self._smoke_eval_epoch(epoch)
            return

        images_folder = fig_filename = os.path.join(self.logging_and_saving_dir, "images")
        os.makedirs(images_folder, exist_ok=True)
        fig_file_str = (
            f"epoch{epoch}"
        )
        fig_filename = os.path.join(images_folder, f"{fig_file_str}_images")
        fid_free = self.do_draw_and_fid(
            self._get_new_key(), fig_filename, free=True
        )
        self.fids_dict["free"].append(fid_free)
        best_free_fid = min(self.fids_dict["free"])
        fid_str = f"Epoch {epoch} free FID: {fid_free:.2f} (best: {best_free_fid:.2f}) "

        if self.cfg.exp.generate_gif:
            self.generate_gif(epoch, self.cfg.exp.animated_images_per_digit, self.cfg.exp.steps_per_sample_in_gif)

        fid_clamped = self.do_draw_and_fid(
            self._get_new_key(), fig_filename, free=False
        )
        self.fids_dict["clamped"].append(fid_clamped)
        best_clamped_fid = min(self.fids_dict["clamped"])
        fid_str += f"clamped FID: {fid_clamped:.2f} (best: {best_clamped_fid:.2f})" + "\n"
        write(fid_str, self.log_file)

        self.eval_epochs.append(epoch)
        self.save_fid_plot()

        if self.cfg.cp.adaptive_cp or self.cfg.wd.adaptive_wd or self.cfg.exp.compute_autocorr:
            self.save_autocorr_plot(epoch)


    def _smoke_eval_epoch(self, epoch):
        key = self._get_new_key()
        batch_size = self.cfg.generation.fid_images_per_digit  # Used as batch size for evals in smoke testing
        cond_acc = compute_conditional_accuracy(self, key, batch_size, None)
        self.fids_dict["clamped"].append(cond_acc)
        free_acc = compute_free_accuracy(self, key, batch_size, None)
        self.fids_dict["free"].append(free_acc)
        acc_str = f"Epoch {epoch} conditional acc: {cond_acc:.2f}% free acc: {free_acc:.2f}%\n"
        write(acc_str)
        self.eval_epochs.append(epoch)


    def save_epoch(self, epoch: int):
        """Saves model configuration and parameters for a specific epoch.

        Saves top-level config once and per-step weights/biases/optimizer states
        in epoch-specific directory.

        **Arguments:**
        - `epoch`: Epoch number for directory naming.
        """
        self.model_saving_path = os.path.join(self.logging_and_saving_dir, "model_saving")

        os.makedirs(self.model_saving_path, exist_ok=True)

        # 1) Save config (once)
        config_path = os.path.join(self.model_saving_path, "config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                yaml.safe_dump(config_to_yaml_dict(self.cfg), f, sort_keys=True)

        # 2) Save this epoch directory
        epoch_dir = os.path.join(self.model_saving_path, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # 3) Save each step's arrays + metadata (graph seed)
        for i, step in enumerate(self.steps):
            save_mask = jax.tree_util.tree_map(lambda _: False, step)
            save_mask = eqx.tree_at(
                lambda s: (s.model.weights, s.model.biases, s.opt_state),
                save_mask,
                (True, True, True),
            )

            params, _ = eqx.partition(step, save_mask)

            # Save
            step_eqx = os.path.join(epoch_dir, f"step_{i:02d}.eqx")
            eqx.tree_serialise_leaves(step_eqx, params)


    @staticmethod
    def _configs_compatible(cfg1: dict, cfg2: dict) -> bool:
        """Checks compatibility between two configurations for loading.

        Verifies critical sections match and seeds are identical to ensure
        consistent graph structure and parameter mapping.
        """
        critical_sections = ["graph", "diffusion_schedule", "diffusion_rates", "sampling", "generation", "data"]
        for section in critical_sections:
            if cfg1.get(section) != cfg2.get(section):
                return False
        return True


    @staticmethod
    def _rebuild_step_interactions(loaded_step):
        """Recompute all per-block interactions from freshly loaded weights/biases."""

        # Training spec
        new_pos = get_new_per_block_interactions(
            loaded_step.training_spec.program_positive,
            loaded_step.model.weights,
            loaded_step.model.biases,
        )
        new_neg = get_new_per_block_interactions(
            loaded_step.training_spec.program_negative,
            loaded_step.model.weights,
            loaded_step.model.biases,
        )
        training_spec2 = eqx.tree_at(
            lambda ts: (ts.program_positive.per_block_interactions, ts.program_negative.per_block_interactions),
            loaded_step.training_spec,
            (new_pos, new_neg),
        )
        loaded_step = eqx.tree_at(lambda s: s.training_spec, loaded_step, training_spec2)

        # Generation spec
        new_free = get_new_per_block_interactions(
            loaded_step.generation_spec.program_free,
            loaded_step.model.weights,
            loaded_step.model.biases,
        )
        new_cond = get_new_per_block_interactions(
            loaded_step.generation_spec.program_conditioned,
            loaded_step.model.weights,
            loaded_step.model.biases,
        )
        generation_spec2 = eqx.tree_at(
            lambda gs: (gs.program_free.per_block_interactions, gs.program_conditioned.per_block_interactions),
            loaded_step.generation_spec,
            (new_free, new_cond),
        )
        loaded_step = eqx.tree_at(lambda s: s.generation_spec, loaded_step, generation_spec2)
        return loaded_step


    @classmethod
    def load(
        cls,
        base_path: str,
        epoch: int | None = None,
        step_sources: Optional[list[str]] = None,
    ):
        """Loads a trained DTM from saved files.

        Reads base config, optionally overrides per-step sources, verifies compatibility,
        deserializes parameters, rebuilds interactions, and returns a ready DTM instance.

        **Arguments:**
        - `base_path`: Directory with config.yaml and epoch directories.
        - `epoch`: Epoch to load if no step_sources; loads from base_path/epoch_XXX.
        - `step_sources`: Optional list of per-step file paths for mixed loading.

        **Returns:**
        - Loaded DTM instance with restored steps and parameters.
        """
        # Read base config and build a DTM to supply structure for deserialization
        base_cfg_path = os.path.join(base_path, "config.yaml")
        if not os.path.exists(base_cfg_path):
            raise ValueError(f"Config not found at {base_cfg_path}")
        with open(base_cfg_path, "r") as f:
            base_cfg_data = yaml.safe_load(f)
        cfg = DTMConfig()
        cfg = load_yaml_config_from_dict(cfg, base_cfg_data)

        num_steps = cfg.diffusion_schedule.num_diffusion_steps

        if step_sources:
            assert len(step_sources) == num_steps # if step_sources is passed, assumed to contain paths to all steps, num_steps being specified by base config    
            config_paths = [os.path.join(os.path.dirname(os.path.dirname(p)), "config.yaml") for p in step_sources]
            graph_seeds = []
            for i, config_path in enumerate(config_paths):
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config not found at {config_path} (for step index {i})")

                with open(config_path, "r") as f:
                    src_cfg_data = yaml.safe_load(f)

                assert DTM._configs_compatible(base_cfg_data, src_cfg_data), (
                    f"Config mismatch between base run and source '{config_path}' (step {i}). "
                    f"This would change graph structure and corrupt parameters."
                )
                graph_seeds.append(src_cfg_data["exp"]["graph_seeds"][i])
            cfg = replace(cfg, exp=replace(cfg.exp, graph_seeds=tuple(graph_seeds)))
            dtm = cls(cfg)
        else:
            step_sources = [os.path.join(base_path, f"epoch_{epoch:03d}", f"step_{i:02d}.eqx") for i in range(num_steps)]
            dtm = cls(cfg)

        # Actually load each step
        new_steps = []
        for i, step_file in enumerate(step_sources):
            if not os.path.exists(step_file):
                raise ValueError(f"Step file not found: {step_file}")
            
            like_step = dtm.steps[i]
            
            # Rebuild the same mask/partition on the fresh object
            save_mask = jax.tree_util.tree_map(lambda _: False, like_step)
            save_mask = eqx.tree_at(
                lambda s: (s.model.weights, s.model.biases, s.opt_state),
                save_mask,
                (True, True, True),
            )
            params_like, static_like = eqx.partition(like_step, save_mask)

            loaded_params = eqx.tree_deserialise_leaves(step_file, params_like)

            # Write them into the live object
            step = eqx.combine(loaded_params, static_like)

            # Rebuild per-block interactions to re-wire views into newly restored params
            loaded_step = cls._rebuild_step_interactions(step)

            new_steps.append(loaded_step)

        # Install loaded steps (DTM isn't an eqx.Module; assignment is fine)
        dtm.steps = new_steps
        return dtm

    def _get_new_key(self):
        new_key, self.key = jr.split(self.key, 2)
        return new_key
    

@eqx.filter_jit
def single_step_denoise(step: DiffusionStep, key: Array, free: bool, n_labels: int, input_in: Array, label_out: Optional[Array], schedule: SamplingSchedule):
    """Performs a single denoise step vmapped over conditions.

    Splits keys for vmap, calls denoise on each slice, and returns batched outputs.

    **Arguments:**
    - `step`: DiffusionStep instance to run.
    - `key`: PRNG key for vmap randomness.
    - `free`: If True, free generation; else conditional.
    - `n_labels`: Number of conditions (labels) to vmap over.
    - `input_in`: Input data, shape `(n_labels, batch_size, input_length)`.
    - `label_out`: Optional label outputs for clamping.
    - `schedule`: Sampling schedule.

    **Returns:**
    - Tuple of image and label output lists from denoise calls.
    """
    
    keys = jr.split(key, n_labels)

    def _denoise_with_step(step, k, input_i, label_o) -> tuple[list[Array], list[Array]]:
        return step.denoise(k, input_i, label_o, schedule)

    in_axes = (None, 0, 0, None if free else 0)
    vmapped = jax.vmap(_denoise_with_step, in_axes=in_axes)
    image_out, label_sampled_out = vmapped(step, keys, input_in, label_out)

    return image_out, label_sampled_out
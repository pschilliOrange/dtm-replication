from __future__ import annotations

from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array, ArrayLike, PRNGKeyArray

import equinox as eqx
import optax

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.observers import StateObserver

from thrmlDenoising.autocorr_fun import autocorr_fn
from thrmlDenoising.base_graphs.abstract_base_graph_manager import AbstractBaseGraphManager
from thrmlDenoising.annealing_graph_ising import IsingNode, hinton_init_from_graph
from thrmlDenoising.ising_training import FloatScalarLike, do_epoch
from thrmlDenoising.sampling_specs import BinomialIsingGenerationSpec, BinomialIsingTrainingSpec
from thrmlDenoising.step_ebm import DiffusionStepEBM
from thrmlDenoising.utils import batch_sample

class DiffusionStep(eqx.Module):
    """Represents a single step in the diffusion process for an Ising-based energy model.

    Manages the EBM model, training/generation specifications, diffusion parameters,
    optimizer state, and autocorrelation tracking for a specific time interval in the
    diffusion schedule. Constructs the underlying graph and sets up sampling programs
    for training and generation.

    **Attributes:**
    - `model`: The diffusion step EBM defining energies and factors.
    - `training_spec`: Specification for training with positive/negative phases.
    - `generation_spec`: Specification for free/conditioned generation sampling.
    - `start_time`: Start time of this diffusion step.
    - `end_time`: End time of this diffusion step.
    - `image_diffusion_rate`: Diffusion rate for image nodes.
    - `label_diffusion_rate`: Diffusion rate for label nodes.
    - `n_grayscale_levels`: Number of grayscale levels (above 0).
    - `n_image_pixels`: Number of pixels in each image.
    - `n_label_nodes`: Number of label nodes.
    - `diffusion_offset`: Offset applied to diffusion times.
    - `optim`: Optimizer for parameter updates.
    - `opt_state`: Current optimizer state.
    - `autocorrelations`: Dictionary mapping epochs to autocorrelation scalars.
    - `autocorrelation_vectors`: Dictionary mapping epochs to autocorrelation vectors over lags.
    - `base_graph_manager`: Manager for base graph construction and conversions.
    """
    model: DiffusionStepEBM
    training_spec: BinomialIsingTrainingSpec
    generation_spec: BinomialIsingGenerationSpec

    start_time: FloatScalarLike
    end_time: FloatScalarLike
    image_diffusion_rate: FloatScalarLike
    label_diffusion_rate: FloatScalarLike
    n_grayscale_levels: int # n_grayscale_levels above 0 that is
    n_image_pixels: int
    n_label_nodes: int
    diffusion_offset: FloatScalarLike

    optim: optax.GradientTransformation
    opt_state: optax.OptState

    autocorrelations: dict
    autocorrelation_vectors: dict

    base_graph_manager: AbstractBaseGraphManager

    def __init__(
        self,
        start_time,
        end_time,
        grayscale,
        n_image_pixels,
        n_label_nodes,
        image_diffusion_rate,
        label_diffusion_rate,
        diffusion_offset,
        training_schedule: SamplingSchedule,
        training_beta,
        generating_schedule: SamplingSchedule,
        generating_betas: Array,
        graph_preset_architecture,
        torus,
        optim,
        key,
        base_graph_manager,
    ):
        """Initializes the diffusion step with graph construction and sampling setups.

        Builds the base graph using the provided manager, initializes the EBM,
        sets up training and generation specifications, computes coupling weights,
        and initializes optimizer and autocorrelation tracking.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.n_grayscale_levels = grayscale
        self.n_image_pixels = n_image_pixels
        self.n_label_nodes = n_label_nodes
        self.image_diffusion_rate = image_diffusion_rate
        self.label_diffusion_rate = label_diffusion_rate
        self.diffusion_offset = diffusion_offset
        self.optim = optim
        self.autocorrelations = {}
        self.autocorrelation_vectors = {}
        self.base_graph_manager = base_graph_manager

        base_graph_image_nodes, base_graph_label_nodes, base_graph_hidden_nodes, base_graph_edges, image_output_blocks, label_output_blocks, hidden_blocks = self.base_graph_manager.make_base_graph(
                            key, 
                            graph_preset_architecture,
                            n_image_pixels,  
                            n_label_nodes, 
                            torus
                        )

        self.model = DiffusionStepEBM(
            base_graph_image_nodes, 
            base_graph_label_nodes, 
            base_graph_hidden_nodes, 
            base_graph_edges,
        )

        # data_blocks must be in the same order as the data (made in _make_training_data) for proper training. We default to image blocks before label blocks
        data_blocks = image_output_blocks + label_output_blocks
        # the conditioning block for the input nodes is always a single block of image the label nodes.
        conditioning_block = Block(self.model.graph.image_input_nodes + self.model.graph.label_input_nodes)

        self.training_spec = BinomialIsingTrainingSpec(self.model, data_blocks, conditioning_block, hidden_blocks, training_schedule, training_schedule, training_beta)
        self.generation_spec = BinomialIsingGenerationSpec(self.model, conditioning_block, image_output_blocks, label_output_blocks, hidden_blocks, generating_schedule, generating_betas)

        self._set_coupling_weights()

        self.opt_state = self.optim.init((self.model.weights, self.model.biases))

    def _set_coupling_weights(self):
        r"""Sets the fixed coupling weights for input-output connections.

        Computes and assigns coupling weights for image and label edges based on
        the diffusion rates and time difference for this step. These weights are
        fixed and implement the forward conditional probability in the model.

        The coupling weight \( w \) for each connection is calculated as:
        \[ w = -\frac{1}{2} \ln\left( \tanh\left( \frac{\lambda \Delta t}{2} \right) \right) \]
        where \( \lambda \) is the diffusion rate (image or label specific) and
        \( \Delta t \) is the time difference (`end_time - start_time`).

        Updates the model's weights array with these values at the corresponding
        edge indices.
        """
        t_diff = self.end_time - self.start_time
        io_image_coupling_weight = -(1/2)*jnp.log(jnp.tanh((self.image_diffusion_rate / 2) * t_diff))
        io_label_coupling_weight = -(1/2)*jnp.log(jnp.tanh((self.label_diffusion_rate / 2) * t_diff))

        edge_map = self.model.graph.edge_mapping
        io_image_edge_indices = jnp.array([edge_map[edge] for edge in self.model.graph.image_coupling_edges])
        io_label_edge_indices = jnp.array([edge_map[edge] for edge in self.model.graph.label_coupling_edges])

        weights = jnp.zeros((len(self.model.graph.edges)), dtype=jnp.float32)
        weights = weights.at[io_image_edge_indices].set(io_image_coupling_weight)
        weights = weights.at[io_label_edge_indices].set(io_label_coupling_weight)

        self.model = self.model.set_coupling_weights(weights)


    def _make_training_data(self, key, image_data: Array, label_data: Array):
        """Prepares positive and negative phase data for training.

        Perturbs image and label data according to the diffusion rates and times
        for this step, converting them into block formats for clamped and free blocks.
        Positive data includes output (start) and input (end) blocks; negative data
        includes only input blocks.

        **Arguments:**
        - `key`: PRNG key for perturbation randomness.
        - `image_data`: Array of image pixel values, shape `(dataset_len, n_image_pixels)`.
        - `label_data`: Array of label values, shape `(dataset_len, n_label_nodes)`.

        **Returns:**
        - Tuple of positive phase data (output + input blocks) and negative phase data (input blocks only).
        """
        keys = jr.split(key, 4)
        start_time = self.start_time * (1 - self.diffusion_offset)
        end_time = self.end_time * (1 + self.diffusion_offset)

        dataset_len = image_data.shape[0]

        assert image_data.shape == (dataset_len, self.n_image_pixels,), f"Expected shape {(dataset_len, self.n_image_pixels)}, got {image_data.shape}"

        assert label_data.shape == (dataset_len, self.n_label_nodes,), f"Expected shape {(dataset_len, self.n_label_nodes)}, got {label_data.shape}"
        label_start = get_perturbed_data(keys[2], label_data, start_time, self.label_diffusion_rate, 1)
        label_end = get_perturbed_data(keys[3], label_start, end_time - start_time, self.label_diffusion_rate, 1)
        converted_label_start = self.base_graph_manager.convert_label_to_label_out_blocks(label_start)

        if self.base_graph_manager.noise_image_data_in_pixel_space:
            # empirically it has been found to be marginally better to perturb integer pixel values, and then convert to front filled ising
            #   this prevents the BMs from ever seeing pixel data with a 1 after a 0, ie [..., 0, ... 1, ...]
            image_pixel_start = get_perturbed_data(keys[0], image_data, start_time, self.image_diffusion_rate, self.n_grayscale_levels)
            image_pixel_end = get_perturbed_data(keys[1], image_pixel_start, end_time - start_time, self.image_diffusion_rate, self.n_grayscale_levels)
            image_ising_start_blocks = self.base_graph_manager.convert_pixels_to_output_blocks(image_pixel_start)
            ising_input = self.base_graph_manager.convert_pixels_and_labels_to_input_block(image_pixel_end, label_end)

        else: 
            raise ValueError("Noising ising image data would require an additional method in the base graph manager " \
            "to go from ising image end data and label end data to input block, which is not currently supported.")

        assert ising_input.shape == (dataset_len, len(self.generation_spec.input_block)) # there is only every one block for the end data, the input to each step
        in_data = (ising_input,)

        assert all(arr.shape == (dataset_len, self.base_graph_manager.image_output_block_lengths[i]) for i, arr in enumerate(image_ising_start_blocks))  
        out_data = tuple(image_ising_start_blocks + converted_label_start)
        assert len(out_data) == len(self.base_graph_manager.image_output_block_lengths) + len(self.base_graph_manager.label_output_block_lengths), (f"{len(out_data)} != {len(self.base_graph_manager.image_output_block_lengths)} + {len(self.base_graph_manager.label_output_block_lengths)}")

        data_pos = out_data + in_data
        data_neg = in_data

        return data_pos, data_neg
        
    def train_step_model(
        self,
        key,
        batch_size,
        image_data: Array,
        label_data: Array,
        correlation_penalty: Optional[FloatScalarLike],
        weight_decay: Optional[FloatScalarLike],
        bias_decay: Optional[FloatScalarLike] = None,
    ):
        """Trains the diffusion step on a batch of data.

        Prepares training data, computes gradients via symmetric KL, and applies
        optimizer updates to weights and biases. Returns updated parameters and
        optimizer state for JIT compatibility.

        **Arguments:**
        - `key`: PRNG key for randomness in initialization and sampling.
        - `batch_size`: Number of examples per batch.
        - `image_data`: Full dataset of image pixels.
        - `label_data`: Full dataset of labels.
        - `correlation_penalty`: Optional coefficient for correlation regularization.
        - `weight_decay`: Optional coefficient for weight decay regularization.
        - `bias_decay`: Optional coefficient for bias decay (defaults to None).

        **Returns:**
        - Tuple of updated weights, biases, and optimizer state.
        """
        key_init, key_train = jr.split(key, 2)
        dataset_len = image_data.shape[0]
        assert image_data.shape == (dataset_len, self.n_image_pixels), f"Expected shape {(dataset_len, self.n_image_pixels)}, got {image_data.shape}"
        assert label_data.shape == (dataset_len, self.n_label_nodes), f"Expected shape {(dataset_len, self.n_label_nodes)}, got {label_data.shape}"

        data_pos, data_neg = self._make_training_data(key_init, image_data, label_data)

        new_weights, new_biases, new_opt_state = do_epoch(
            key_train,
            self.model,
            self.training_spec,
            self.model.graph.output_nodes + self.model.graph.hidden_nodes, # I don't think the nodes order matters here because we use global in symetric kl grad
            self.model.graph.base_graph_edges,
            batch_size,
            data_pos,
            data_neg,
            self.training_spec.beta,
            self.optim,
            self.opt_state,
            weight_decay,
            bias_decay,
            correlation_penalty,
        )

        return new_weights, new_biases, new_opt_state
    
    def denoise(
        self,
        key: PRNGKeyArray,
        input_data: Array,  # Array of shape (batch_size, len(self.generation_spec.input_block))
        label_output_data: Optional[Array],  # None for free, (batch_size, self.base_graph_manager.n_label_nodes) for clamped
        schedule: SamplingSchedule,
    ) -> tuple[list[Array], list[Array]]:
        """Runs the denoising process for this diffusion step.

        Initializes states, sets up clamped/free sampling based on whether labels
        are provided, and observes image/label output blocks over the schedule.
        Returns lists of image and label output blocks across samples.

        **Arguments:**
        - `key`: PRNG key for sampling randomness.
        - `input_data`: Input data array, shape `(batch_size, input_block_length)`.
        - `label_output_data`: Optional label data for conditional generation;
          if None, performs free generation.
        - `schedule`: Sampling schedule defining warmup and sampling steps.

        **Returns:**
        - Tuple of image output blocks and label output blocks, each a list of
          arrays with shape `(batch_size, n_samples, block_length)`.
        """
        key, k_init, k_sample = jr.split(key, 3)
        batch_size = input_data.shape[0]
        
        # Assert input shapes match expected block shapes
        assert input_data.shape == (batch_size, len(self.generation_spec.input_block))
        if label_output_data is not None:
            assert label_output_data.shape == (batch_size, self.base_graph_manager.n_label_nodes)

        if label_output_data is None:
            # free generation
            init = hinton_init_from_graph(
                k_init,
                self.model,
                self.generation_spec.program_free.gibbs_spec.free_blocks,
                batch_size,
                self.generation_spec.beta_schedule[-1],
            )
            clamps = [input_data]
            observer = StateObserver(self.generation_spec.image_output_blocks + self.generation_spec.label_output_blocks)
            samples = batch_sample(k_sample, init, clamps, self.generation_spec.program_free, schedule, observer)

        else:
            #conditional generation
            init = hinton_init_from_graph(
                k_init,
                self.model,
                self.generation_spec.program_conditioned.gibbs_spec.free_blocks,
                batch_size,
                self.generation_spec.beta_schedule[-1],
            )
            clamps = self.base_graph_manager.convert_label_to_label_out_blocks(label_output_data) + [input_data]

            observer = StateObserver(self.generation_spec.image_output_blocks + self.generation_spec.label_output_blocks) #Here we observe the values we are clamping just for consistency
            samples = batch_sample(k_sample, init, clamps, self.generation_spec.program_conditioned, schedule, observer)
        
        image_out_blocks = samples[:len(self.generation_spec.image_output_blocks)]
        label_out_blocks = samples[len(self.generation_spec.image_output_blocks):]

        # Assert shapes for each block
        for i, block_array in enumerate(image_out_blocks):
            expected_shape = (batch_size, schedule.n_samples, len(self.generation_spec.image_output_blocks[i]))
            assert block_array.shape == expected_shape, \
                f"Image block {i} shape mismatch: got {block_array.shape}, expected {expected_shape}"
        
        for i, block_array in enumerate(label_out_blocks):
            expected_shape = (batch_size, schedule.n_samples, len(self.generation_spec.label_output_blocks[i]))
            assert block_array.shape == expected_shape, \
                f"Label block {i} shape mismatch: got {block_array.shape}, expected {expected_shape}"

        return image_out_blocks, label_out_blocks

    def compute_autocorr(
        self,
        key, 
        n_cores,
        n_reps,
        n_chains,
        test_images,
        test_labels,
        schedule,
        random_matrix,
        encoded_dim,
        epoch: Optional[int] = None,
    ):
        """Computes autocorrelation as a proxy for mixing time using test data.

        Perturbs test images and labels, runs the denoise process to generate samples,
        encodes samples to a lower dimension, and computes autocorrelation over lags.
        Stores scalar mean of tail autocorrelation and full vector if epoch provided.

        **Arguments:**
        - `key`: PRNG key for randomness in selection and perturbation.
        - `n_cores`: Number of cores for parallel computation.
        - `n_reps`: Repetitions per core for sampling.
        - `n_chains`: Number of independent chains per repetition.
        - `test_images`: Test image dataset for perturbation and sampling.
        - `test_labels`: Test labels for perturbation.
        - `schedule`: Sampling schedule for denoise runs.
        - `random_matrix`: Matrix for encoding samples to lower dimension.
        - `encoded_dim`: Dimensionality of encoded space.
        - `epoch`: Optional epoch number to store results under.

        **Returns:**
        - Scalar autocorrelation value (mean of tail segment).
        """
        @jax.jit
        def inner_fn(inner_key):

            inner_key, select_key, perturb_key = jr.split(inner_key, 3)

            # Randomly select a test image.
            rand_idx = jr.randint(select_key, (1,), 0, 1000)[0]

            img = test_images[rand_idx]
            label = test_labels[rand_idx] #[None]

            # Perturb the image based on the diffusion time.
            time_in = self.end_time * (1 + self.diffusion_offset)
            perturbed_image = get_perturbed_data(
                perturb_key, img, time_in, self.image_diffusion_rate, self.n_grayscale_levels
            )
            perturbed_image = jnp.tile(perturbed_image, (n_chains, 1))
            assert perturbed_image.shape == (n_chains, self.n_image_pixels)

            perturbed_label = get_perturbed_data(
                perturb_key, label, self.start_time, self.label_diffusion_rate, 1 #fine to hardcode bin_trails as 1 because labels are always just one hot, ie bin_trails = 1 binomial nodes
            )
            perturbed_label = jnp.tile(perturbed_label, (n_chains, 1))
            assert perturbed_label.shape == (n_chains, self.n_label_nodes)   
            input_data = self.base_graph_manager.convert_pixels_and_labels_to_input_block(perturbed_image, perturbed_label)

            # Run one diffusion step to get long-run samples.
            image_long_run_samples_ising, _ = self.denoise( # only image nodes are used for autocorr calc
                inner_key,
                input_data,
                None, # we always compute autocorrelations where all output and hidden nodes are free
                schedule,
            )
            image_long_run_samples = self.base_graph_manager.convert_output_blocks_to_pixels(image_long_run_samples_ising)
            assert image_long_run_samples.shape == (n_chains, schedule.n_samples, self.n_image_pixels)
            image_long_run_samples = jnp.astype(image_long_run_samples, jnp.float32) / self.n_grayscale_levels

            encoded_samples = jnp.dot(image_long_run_samples, random_matrix)
            assert encoded_samples.shape == (n_chains, schedule.n_samples, encoded_dim)
            return encoded_samples
        
        inner_keys = jr.split(key, n_cores*n_reps).reshape(n_cores, n_reps, 2)

        map_inner_fn = lambda _keys: jax.lax.map(inner_fn, _keys)
        encoded_samples = jax.pmap(
            map_inner_fn, in_axes=0, out_axes=0, backend="cuda"
        )(inner_keys)
        assert encoded_samples.shape == (     
            n_cores,
            n_reps,
            n_chains,
            schedule.n_samples,
            encoded_dim,
        )

        autocorr = autocorr_fn(encoded_samples, "cuda") # hardcode backend cuda

        # autocorr is a vector here of autocorrelation over all lags computed for
        autocorr_scalar = max(0.0, float(np.mean(autocorr[-(len(autocorr) // 16) :])))

        if epoch is not None:
            self.autocorrelations[epoch] = autocorr_scalar
            self.autocorrelation_vectors[epoch] = autocorr

        return autocorr_scalar


def get_perturbed_data(
    key, data: Array, dt: FloatScalarLike, rates: FloatScalarLike, bin_trials: int
) -> Array:
    """Applies perturbation noise to data over a time interval.

    Independently flips each trial in the data with probability based on the rate
    and time `dt`, accumulating results to produce perturbed integer values.

    **Arguments:**
    - `key`: PRNG key for generating random flips.
    - `data`: Input data array to perturb.
    - `dt`: Time interval for perturbation.
    - `rates`: Flip rate (scalar).
    - `bin_trials`: Number of binomial trials per data element.

    **Returns:**
    - Perturbed data array with same shape as input.
    """
    rates = jnp.array(rates)
    assert rates.ndim == 0

    # Compute the flip probability
    flip_prob = 1 - jnp.exp(-rates * dt)

    # Pre-split the key into 2 keys per trial (one for the flip mask, one for the flip value)
    all_keys = jr.split(key, bin_trials * 2)
    keys_mask = all_keys[:bin_trials]
    keys_val = all_keys[bin_trials:]

    # The scan body iterates over each trial index along with its corresponding pre-split keys.
    def scan_body(acc, xs):
        i, key_mask, key_val = xs
        # For this trial, compare data with the trial index i.
        trial_data = data > i
        # Generate random booleans for the flip decision using flip_prob.
        flip_mask = jr.bernoulli(key_mask, p=flip_prob, shape=data.shape)
        # Generate random booleans (with probability 0.5) for the flipped value.
        flip_value = jr.bernoulli(key_val, p=0.5, shape=data.shape)
        # If flip_mask is True, use flip_value; otherwise use trial_data.
        trial_result = jnp.where(flip_mask, flip_value, trial_data)
        # Accumulate the result (casting booleans to data.dtype).
        acc = acc + trial_result.astype(data.dtype)
        return acc, None

    # Prepare the scan inputs: trial indices and pre-split keys.
    xs = (jnp.arange(bin_trials), keys_mask, keys_val)
    init_acc = jnp.zeros(data.shape, dtype=data.dtype)
    final_acc, _ = lax.scan(scan_body, init_acc, xs)
    return final_acc
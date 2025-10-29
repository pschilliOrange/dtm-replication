import unittest
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Key
import numpy as np
import optax

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, BlockGibbsSpec, sample_states
from thrml.factor import FactorSamplingProgram
from thrmlDenoising.annealing_graph_ising import AbstractAnnealingIsingSampler
from thrmlDenoising.step import DiffusionStep, get_perturbed_data
from thrmlDenoising.base_graphs.poisson_binomial_ising_graph_manager import PoissonBinomialIsingGraphManager

def generate_perturbed_samples(
    key: Key, 
    data: Array, 
    dt: float, 
    rate: float, 
    bin_trials: int, 
    n_samples: int
) -> Array:
    keys = random.split(key, n_samples)
    
    def perturb_single(k):
        return get_perturbed_data(k, data, dt, rate, bin_trials)
    
    return jax.vmap(perturb_single)(keys)

class TestDiffusionStep(unittest.TestCase):
    """Tests coupling weights in DiffusionStep.
    
    Verifies that sampling from input blocks (with clamped outputs) matches expected
    perturbations from get_perturbed_data and manual computation. Hidden nodes do not
    connect to inputs, so their clamping is irrelevant here. Normally, forward perturbation
    is computed analytically via get_perturbed_data for efficiency.
    """
    key = random.PRNGKey(1)
    n_samples = 100000

    def test_coupling_weights(self, n_trials=1):
        start_time = jnp.asarray(0)
        end_time = jnp.asarray(1)
        image_size = 100
        image_diffusion_rate = 1
        label_size = 20
        label_diffusion_rate = 0.5
        graph_preset = 44_8
        torus = False

        graph_manager = PoissonBinomialIsingGraphManager(image_size, label_size, n_trials)

        step = DiffusionStep(
            start_time,
            end_time,
            n_trials,
            image_size,
            label_size,
            image_diffusion_rate,
            label_diffusion_rate,
            0.0,
            SamplingSchedule(0,0,0),
            1.0,
            SamplingSchedule(0,0,0),
            jnp.array(1.0),
            graph_preset,
            torus,
            optax.adam(1.0), 
            self.key,
            graph_manager,
        )
    
        # a spec that clamps all nodes except the input nodes
        test_diffusion_rate_spec = BlockGibbsSpec(
            [Block(step.model.graph.image_input_nodes), Block(step.model.graph.label_input_nodes)],
            [Block(step.model.graph.hidden_nodes), 
             Block(step.model.graph.image_output_nodes),
             Block(step.model.graph.label_output_nodes)],
             step.model.node_shape_dtypes)

        factors =  step.model.factors
        class BinomialSampler(AbstractAnnealingIsingSampler):
            betas = jnp.array(1.0)
            schedule = SamplingSchedule(100, self.n_samples, 5)
        samplers = [BinomialSampler() for _ in range(6)] # 6 being hardcoded to how many factors there are

        prog = FactorSamplingProgram(
            gibbs_spec=test_diffusion_rate_spec,
            samplers=samplers,
            factors=factors,
            other_interaction_groups=[]
        )
        init_free = [jnp.zeros((len(b.nodes),), step.model.node_shape_dtypes[type(b.nodes[0])]) for b in prog.gibbs_spec.free_blocks]

        hidden_zeros = [jnp.zeros(len(step.model.graph.hidden_nodes), dtype=jnp.bool)]
        
        dt = float(end_time - start_time)
        flip_prob_image = 0.5*(1 - np.exp(-image_diffusion_rate * dt))
        flip_prob_label = 0.5*(1 - np.exp(-label_diffusion_rate * dt))

        # case 1, all-zero data
        data_image = jnp.zeros(image_size, dtype=jnp.bool_)
        data_label = jnp.zeros(label_size, dtype=jnp.bool_)
        data = [data_image, data_label]

        samples = sample_states(
            key=self.key,
            program=prog,
            schedule=samplers[0].schedule,
            init_state_free=init_free,
            state_clamp=hidden_zeros + data,
            nodes_to_sample=[Block(step.model.graph.image_input_nodes), Block(step.model.graph.label_input_nodes)]  #step_model.image_output_block, step_model.label_output_block],
        )

        analytical_image_samples = generate_perturbed_samples(self.key, data[0], dt, image_diffusion_rate, n_trials, self.n_samples)
        analytical_label_samples = generate_perturbed_samples(self.key, data[1], dt, label_diffusion_rate, 1, self.n_samples)

        sampled_image_mean = np.mean(samples[0], axis=(0, 1))
        analytical_image_mean = np.mean(analytical_image_samples, axis=(0, 1))
        computed_image_mean = flip_prob_image * n_trials #because all nodes start at 0, all flip to 1 with prob flip_prob_image
        sampled_label_mean = np.mean(samples[1], axis=(0, 1))
        analytical_label_mean = np.mean(analytical_label_samples, axis=(0, 1))
        computed_label_mean = flip_prob_label
        print("with all initial zeros, sampled image mean:", sampled_image_mean, 
                                        "get_perturbed_data image mean:", analytical_image_mean, 
                                        "computed image mean", computed_image_mean)
        assert (max(sampled_image_mean, analytical_image_mean, computed_image_mean) - min(sampled_image_mean, analytical_image_mean, computed_image_mean))/n_trials < .05
        print("with all initial zeros, sampled label mean:", sampled_label_mean, 
                                        "get_perturbed_data image mean:", analytical_label_mean, 
                                        "computed image mean", computed_label_mean)
        assert (max(sampled_label_mean, analytical_label_mean, computed_label_mean) - min(sampled_label_mean, analytical_label_mean, computed_label_mean)) < .05

        # Case 2, random data
        image_key, label_key = random.split(self.key)
        data_image = random.randint(image_key, shape=(image_size,), minval=0, maxval=n_trials + 1).astype(jnp.bool_)
        data_label = random.bernoulli(label_key, shape=(label_size,)).astype(jnp.bool_)
        data = [data_image, data_label]

        samples = sample_states(
            key=self.key,
            program=prog,
            schedule=samplers[0].schedule,
            init_state_free=init_free,
            state_clamp=hidden_zeros + data,
            nodes_to_sample=[Block(step.model.graph.image_input_nodes), Block(step.model.graph.label_input_nodes)]
        )

        analytical_image_samples = generate_perturbed_samples(self.key, data[0], dt, image_diffusion_rate, n_trials, self.n_samples)
        analytical_label_samples = generate_perturbed_samples(self.key, data[1], dt, label_diffusion_rate, 1, self.n_samples)

        sampled_image_mean = np.mean(samples[0], axis=0)
        analytical_image_mean = np.mean(analytical_image_samples, axis=0)
        computed_image_mean = (n_trials - data[0])*flip_prob_image + data[0]*(1-flip_prob_image) #expected number of 0's flipping to 1 plus expected number of 1's staying
        sampled_label_mean = np.mean(samples[1], axis=0)
        analytical_label_mean = np.mean(analytical_label_samples, axis=0)
        computed_label_mean = (1- data[1]) * flip_prob_label + data[1] * (1 - flip_prob_label)
        image_means = np.stack([sampled_image_mean, analytical_image_mean, computed_image_mean], axis=0)
        image_diffs = np.max(image_means, axis=0) - np.min(image_means, axis=0)
        assert np.all(image_diffs / n_trials < .05)
        label_means = np.stack([sampled_label_mean, analytical_label_mean, computed_label_mean], axis=0)
        label_diffs = np.max(label_means, axis=0) - np.min(label_means, axis=0)
        assert np.all(label_diffs < .05)

if __name__ == "__main__":
    unittest.main(verbosity=2)
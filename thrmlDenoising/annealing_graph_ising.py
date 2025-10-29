from typing import ClassVar, TypeAlias
from typing_extensions import Self
import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree, ArrayLike, Float
from abc import abstractmethod
import jax

from thrmlDenoising.pgm_continued import SparseGraph

from thrml.block_management import Block, BlockSpec, from_global_state
from thrml.conditional_samplers import AbstractConditionalSampler, _State, _SamplerState
from thrml.block_sampling import SamplingSchedule
from thrml.pgm import AbstractNode
from thrml.models.ebm import EBMFactor, AbstractFactorizedEBM
from thrml.interaction import InteractionGroup
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram


FloatScalarLike: TypeAlias = Float[ArrayLike, ""]

class IsingNode(AbstractNode):
    """Represents a binary random variable in Ising models.
    
    Takes values in {-1, 1} for energy calculations but stores as {0, 1} for efficiency.
    In DTM, forms the basis for encoding pixels and labels in graphs.
    """
    pass

class BiasInteraction(eqx.Module):
    """Holds biases and their global indices for node-wise energy contributions.
    Global indices arrays are used in sampling_specs.py for updating interactions at the
    array level when in a jitted function."""
    biases: Array
    bias_global_indices: Array

class BiasFactor(EBMFactor):
    """Factor adding bias terms to node energies in Ising EBMs.
    
    **Attributes:**
    - `biases`: Array of bias values, shape matching block size.
    - `bias_global_indices`: Maps biases to global node positions.
    """
    biases: Array
    bias_global_indices: Array

    def __init__(self, biases: Array, block: Block, bias_global_indices: Array):
        super().__init__([block])
        self.biases = biases
        self.bias_global_indices = bias_global_indices

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        """Computes bias energy for global energy in testing. 
        
        Used only in testing for computing the global energy. Local field energy calculations, 
        unlike this method, never look at the state of the nodes actually being sampled,
        only the state of the nodes it is connected with.  """
        extracted = from_global_state(global_state, block_spec, self.node_groups)
        state = extracted[0]
        s = 2 * state.astype(jnp.float32) - 1
        return -jnp.sum(self.biases * s)

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Produces bias interactions affecting head nodes only."""
        return [InteractionGroup(
            interaction=BiasInteraction(self.biases, self.bias_global_indices),
            head_nodes=self.node_groups[0],
            tail_nodes=[]
        )]

class WeightInteraction(eqx.Module):
    """Holds weights and global indices for pairwise interactions.
    Global indices arrays are used in sampling_spec.py for updating interactions at the
    array level when in a jitted function."""
    weights: Array
    weight_global_indices: Array

class WeightFactor(EBMFactor):
    """Factor for weighted pairwise energies in Ising EBMs.
    
    **Attributes:**
    - `weights`: Edge weights, shape matching number of pairs.
    - `weight_global_indices`: Maps weights to global edge positions.
    """
    weights: Array
    weight_global_indices: Array

    def __init__(self, weights: Array, blocks: tuple[Block, Block], weight_global_indices: Array):
        super().__init__(list(blocks))
        self.weights = weights
        self.weight_global_indices = weight_global_indices

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        """Computes pairwise energy for global energy in testing. 
        
        Used only in testing for computing the global energy. Local field energy calculations, 
        unlike this method, never look at the state of the nodes actually being sampled,
        only the state of the nodes it is connected with.  """
        extracted = from_global_state(global_state, block_spec, self.node_groups)  # List of 2 batched arrays, each (num_states, n_edges)
        state0, state1 = extracted
        s0 = 2 * state0.astype(jnp.float32) - 1
        s1 = 2 * state1.astype(jnp.float32) - 1
        return -jnp.sum(self.weights * s0 * s1)

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Generates bidirectional interactions for undirected edges."""
        return [InteractionGroup(WeightInteraction(self.weights, self.weight_global_indices), self.node_groups[0], [self.node_groups[1]]),
                InteractionGroup(WeightInteraction(self.weights, self.weight_global_indices), self.node_groups[1], [self.node_groups[0]])]

class AbstractAnnealingIsingSampler(AbstractConditionalSampler):
    """Sampling for Ising with annealing over betas.
    
    Supports scalar or scheduled inverse temperatures for controlled sampling.
    
    **Attributes:**
    - `betas`: Inverse temperatures, scalar or array for schedule.
    - `schedule`: Defines warmup and sampling steps.
    """
    betas: ClassVar[Array]
    schedule: ClassVar

    def sample(self, key: Key, interactions: list[PyTree],
               active_flags: list[Array], states: list[list[_State]],
               sampler_state: _SamplerState,
               output_sd: PyTree[jax.ShapeDtypeStruct]) -> tuple[Array, _SamplerState]:
        """Updates block states via local fields from weights/biases.
        
        Computes energies, applies beta (scalar or scheduled), samples Bernoulli in {0,1}.
        """
        itr = sampler_state

        num_nodes = output_sd.shape[0]
        energy = jnp.zeros(shape=(num_nodes),dtype=jnp.float32)

        for active, interaction, state in zip(active_flags, interactions, states):
            if isinstance(interaction, WeightInteraction):
                weights = interaction.weights
                neighbor_binary_state = state[0].astype(jnp.float32) # this is the neighboring state in the 0's and 1's
                neighbor_spin_state = 2 * neighbor_binary_state - 1 # and this is the neighbouring states in -1's and 1's
                assert neighbor_spin_state.ndim == 2, ("neighbor_state should have have two dimensions [n, k]. "
                        "n is the number of nodes that we are updating in parallel during this call to sample and "
                        "k is the maximum number of times any node in the block that is being updated shows up as a head node for this interaction."
                        "A leading batch dimension is not supported in this sampler now, not sure if it should be...")
                energy -= jnp.sum(
                    weights * neighbor_spin_state * active, axis=1 #sum contributions from all interacting nodes
                )
                assert energy.shape == (num_nodes,), f"{energy.shape} != {(num_nodes)}"

            elif isinstance(interaction, BiasInteraction):
                biases = interaction.biases
                bias_contribution = biases * active
                #assert that only one node is interacting through biases
                assert bias_contribution.ndim == 2 and bias_contribution.shape[1] == 1, ("bias_contribution should have have two dimensions [n, 1]. "
                        "n is the number of nodes that we are updating in parallel during this call to sample and "
                        "the axis=1 dimension is the generally the maximum number of times any node in the block that is being updated shows up as a head node for this interaction, which should always be 1 for bias interactions."
                        "A leading batch dimension is not supported in this sampler now, not sure if it should be...")
                bias_contribution = jnp.squeeze(bias_contribution, axis=1) #remove empty dimension meant for max number of nodes affecting updated energy states
                energy -= bias_contribution 
                assert energy.shape == (num_nodes,), f"{energy.shape} != {(num_nodes)}"        
            else:
                raise ValueError("Invalid interaction class")
            
        # if beta is a schedule over warmups steps, get betas[itr], otherwise use the scalar beta
        #   Here annealing is assumed to only be used in generation where we care about the final (and often only) sample after warmup
        if jnp.ndim(self.betas) == 0:
            beta = self.betas
        elif jnp.ndim(self.betas) == 1:
            assert self.betas.ndim == 1
            assert self.betas.shape[0] == self.schedule.n_warmup, "beta schedule assumed to be length of warmup so that first sample is sampled as hottest beta"
            beta = self.betas[itr]
        else:
            raise ValueError(f"betas should be a scalar or a 1D array, got {self.betas}")
        new_sampler_state = itr + 1

        # the 2 in sigmoid assumes BIPOLAR states
        p_per_node = jax.nn.sigmoid(-2*beta*energy)
        updated_binary_state = jax.random.bernoulli(key, p_per_node)

        assert updated_binary_state.shape == output_sd.shape
        return (updated_binary_state), new_sampler_state
    
    def init(self) -> _SamplerState:
        """Starts sampler iteration counter at 0 for annealing."""
        return 0
    
class AbstractIsingEBMwithGraph(AbstractFactorizedEBM):
    """EBM for Ising models on sparse graphs.
    
    Defines energy via biases and weights on nodes/edges.
    
    **Attributes:**
    - `graph`: Sparse graph of nodes/edges.
    - `biases`: Node biases.
    - `weights`: Edge weights.
    """

    graph: SparseGraph
    biases: Array
    weights: Array

    def __init__(self, graph: SparseGraph, biases: Array, weights: Array):
        """Sets up EBM with uniform node types; assumes at least one node."""       
        sd_map = {graph.nodes[0].__class__: jax.ShapeDtypeStruct((), jnp.bool)}

        super().__init__(sd_map)

        self.graph = graph
        self.biases = biases
        self.weights = weights

    @property
    @abstractmethod
    def factors(self) -> list[EBMFactor]:
        """Lists EBM factors for the model."""
        pass

    @abstractmethod
    def update_weights_and_biases(self, new_weights: Array, new_biases: Array) -> Self:
        """Returns updated EBM with new parameters, used under jit in do_epoch."""
        pass

class AnnealingIsingSamplingProgram(FactorSamplingProgram):
    """A very thin wrapper on FactorSamplingProgram that specializes it to the case of an Ising Model with annealing."""

    def __init__(self, ebm: AbstractIsingEBMwithGraph, free_blocks: list[Block], clamped_blocks: list[Block], beta_array: Array, sampler_schedule: SamplingSchedule):
        class AnnealingSampler(AbstractAnnealingIsingSampler):
            betas = beta_array
            schedule = sampler_schedule
        samp = AnnealingSampler()
        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)
        super().__init__(spec, [samp for _ in spec.free_blocks], ebm.factors, [])
    

@eqx.filter_jit
def hinton_init_from_graph(
    key,
    model: AbstractIsingEBMwithGraph,
    blocks: list[Block],
    batch_size: int,
    beta: FloatScalarLike,
) -> list[Array]:
    """
    Initialize binary states for nodes in specified blocks.

    **Arguments:**

    - key: PRNGKey.
    - model: BinomialIsingModel; model.nodes is a list of all nodes.
    - blocks: tuple of Block; each Block represents a group of nodes.
    - batch_size: int; number of samples to initialize.
    - beta: Scalar or Array; the inverse temperature used to compute probabilities.

    **Returns:**

    - List of Arrays; for each Block in `blocks`, returns an Array with
        shape (batch_size, len(block)) where each entry is a binary state sampled
        via a Bernoulli distribution.

    The probabilities are computed using the sigmoid activation on beta * block_biases.
    """

    node_map = model.graph.node_mapping
    data = []
    block_keys = jax.random.split(key, len(blocks))
    for block, block_key in zip(blocks, block_keys):
        block_slice = jnp.array([node_map[node] for node in block], dtype=jnp.int32) #better to keep this all in jax here?
        block_biases = model.biases[block_slice]
        assert issubclass(block.node_type, IsingNode)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        else:
            target_shape = (batch_size, len(block))
            probs = jnp.reshape(jax.nn.sigmoid(beta * block_biases), (1, -1))
        block_data = jax.random.bernoulli(block_key, probs, target_shape)
        assert block_data.shape == (batch_size, len(block))
        data.append(block_data)

    return data
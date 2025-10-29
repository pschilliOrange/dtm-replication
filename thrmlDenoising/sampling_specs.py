import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrmlDenoising.annealing_graph_ising import (
    AbstractIsingEBMwithGraph,
    AnnealingIsingSamplingProgram,
    BiasInteraction,
    WeightInteraction,
)

def assert_no_intrablock_edges(graph, blocks):
    """
    Verifies no connected nodes exist within the same block to ensure accurate sampling.

    Intra-block edges could lead to stale neighbor states during updates.
    Raises ValueError if such edges are found.

    Args:
        graph: The graph containing nodes and edges.
        blocks: List of blocks to check.
    """
    idx = {}
    for b_i, b in enumerate(blocks):
        for n in b:
            idx[n] = b_i
    for e in graph.edges:
        u, v = e.connected_nodes
        bu, bv = idx.get(u), idx.get(v)
        if bu is not None and bu == bv:
            raise ValueError(f"Intra-block edge between two nodes in the block with index {bu} that was handed to assert_no_intrablock_edges. \
                             This is almost certainly caused by an error in the base graph manager, grouping two connected nodes within the same block.")

def _is_interaction(x):
    """Checks if an object is a bias or weight interaction."""
    return isinstance(x, (BiasInteraction, WeightInteraction))

@eqx.filter_jit
def get_new_per_block_interactions(program, weights, biases):
    """
    Updates per-block interactions with new weights and biases.

    Slices global weights/biases ground truth into interaction-specific 
    views, which is useful because it can be done quickly and even in a 
    jitted function as it changes arrays only.

    Args:
        program: The sampling program with per-block interactions.
        weights: Updated global weights array.
        biases: Updated global biases array.

    Returns:
        Updated list of per-block interactions.
    """
    def upd(inter):
        out = inter
        if hasattr(inter, "weights"):
            w = jnp.take(weights, inter.weight_global_indices, axis=0)
            out = eqx.tree_at(lambda i: i.weights, out, w)
        if hasattr(inter, "biases"):
            b = jnp.take(biases, inter.bias_global_indices, axis=0)
            out = eqx.tree_at(lambda i: i.biases, out, b)
        return out

    new_per_block_interactions = jax.tree_util.tree_map(upd, program.per_block_interactions, is_leaf=_is_interaction)
    return new_per_block_interactions


class BinomialIsingTrainingSpec(eqx.Module):
    #doc
    """Contains a complete specification of an Binomial Ising EBM that can be trained using sampling-based gradients.

    Defines sampling programs and schedules that allow for collection of the positive and negative phase samples
    required for monte-carlo estimation of the gradient of the KL-divergence between the model and a data distribution.
    """

    program_positive: AnnealingIsingSamplingProgram
    program_negative: AnnealingIsingSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule
    beta: Array

    def __init__(
        self,
        ebm: AbstractIsingEBMwithGraph,
        data_blocks: list[Block],
        conditioning_block: Block,
        hidden_blocks: list[Block],
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
        beta: Array,
    ):
        """
        Initializes the training specification.

        Args:
            ebm: The Ising EBM model.
            data_blocks: Blocks for data variables.
            conditioning_block: Block for conditioning variables.
            hidden_blocks: Blocks for hidden variables.
            schedule_positive: Sampling schedule for positive phase.
            schedule_negative: Sampling schedule for negative phase.
            beta: Inverse temperature.
        """
        assert_no_intrablock_edges(ebm.graph, hidden_blocks + data_blocks + [conditioning_block])

        self.program_positive = AnnealingIsingSamplingProgram(ebm, hidden_blocks, data_blocks + [conditioning_block], beta, schedule_positive)
        self.program_negative = AnnealingIsingSamplingProgram(ebm, hidden_blocks + data_blocks, [conditioning_block], beta, schedule_negative)

        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative

        self.beta = beta

    def update_weights_and_biases(self, new_weights, new_biases):
        """
        Creates an updated training spec with new weights and biases.

        Used for jitting in training loops by avoiding mutation.

        Args:
            new_weights: Updated weights array.
            new_biases: Updated biases array.

        Returns:
            New training spec instance with updated programs.
        """

        new_pos_per_block_interactions = get_new_per_block_interactions(self.program_positive, new_weights, new_biases)
        new_neg_per_block_interactions = get_new_per_block_interactions(self.program_negative, new_weights, new_biases)
        new_pos_program = eqx.tree_at(lambda p: p.per_block_interactions, self.program_positive, new_pos_per_block_interactions)
        new_neg_program = eqx.tree_at(lambda p: p.per_block_interactions, self.program_negative, new_neg_per_block_interactions)

        return eqx.tree_at(lambda s: (s.program_positive, s.program_negative), self, (new_pos_program, new_neg_program))

class BinomialIsingGenerationSpec(eqx.Module):
    """
    Specification for generation in an Ising EBM.

    Defines free and conditioned sampling programs for generation tasks,
    along with input/output blocks and annealing schedule.
    """
    program_free: AnnealingIsingSamplingProgram
    program_conditioned: AnnealingIsingSamplingProgram
    schedule: SamplingSchedule
    beta_schedule: Array

    input_block: Block
    image_output_blocks: list[Block]
    label_output_blocks: list[Block]

    def __init__(
        self,
        ebm: AbstractIsingEBMwithGraph,
        input_block: Block,
        image_output_blocks: list[Block],
        label_output_blocks: list[Block],
        hidden_blocks: list[Block],
        schedule: SamplingSchedule,
        betas: Array
    ):
        """
        Initializes the generation specification.

        Args:
            ebm: The Ising EBM model.
            input_block: Input block for generation.
            image_output_blocks: Output blocks for image variables.
            label_output_blocks: Output blocks for label variables.
            hidden_blocks: Blocks for hidden variables.
            schedule: Sampling schedule.
            betas: Annealing beta schedule array.
        """
        assert_no_intrablock_edges(ebm.graph, image_output_blocks + label_output_blocks + hidden_blocks + [input_block])

        self.program_free = AnnealingIsingSamplingProgram(ebm, image_output_blocks + label_output_blocks + hidden_blocks, [input_block], betas, schedule)
        self.program_conditioned = AnnealingIsingSamplingProgram(ebm, image_output_blocks + hidden_blocks, label_output_blocks + [input_block], betas, schedule)
        self.schedule = schedule
        self.beta_schedule = betas

        self.image_output_blocks = image_output_blocks
        self.label_output_blocks = label_output_blocks
        self.input_block = input_block
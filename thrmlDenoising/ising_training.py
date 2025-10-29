from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Key

from thrml.models.ising import estimate_moments
from thrml.pgm import AbstractNode
from thrmlDenoising.annealing_graph_ising import AbstractIsingEBMwithGraph, FloatScalarLike, hinton_init_from_graph
from thrmlDenoising.pgm_continued import Edge
from thrmlDenoising.sampling_specs import BinomialIsingTrainingSpec

@eqx.filter_jit
def symmetric_kl_grad(
    key,
    model: AbstractIsingEBMwithGraph,
    training_spec: BinomialIsingTrainingSpec,
    bias_nodes: list[AbstractNode],
    weight_edges: list[Edge],
    free_data_positive: list[Array],
    clamped_data_positive: list[Array],
    free_data_negative: list[Array],
    clamped_data_negative: list[Array],
    beta: Array,
    correlation_penalty: Optional[FloatScalarLike] = None,
):
    """
    Computes gradients for biases and weights using a symmetric approximation to the KL divergence.
    
    This function estimates moments from positive (data-clamped) and negative (model-generated) phases
    and computes gradients as -beta * (positive_moments - negative_moments). The term 'symmetric' refers
    to the fact that the positive and negative phases take in the same init and clamped data structures.
    The positive phase samples hidden variables given clamped data and conditioning, while the negative 
    phase samples both hidden and data variables given only conditioning. 
    
    Returns gradients for all nodes/biases and edges/weights in the model, with zeros for those not
    included in bias_nodes and weight_edges.
    
    Args:
        key: JAX random key for sampling.
        model: The Ising EBM model.
        training_spec: Training specification including positive and negative sampling programs.
        bias_nodes: List of nodes for which to compute bias gradients.
        weight_edges: List of edges for which to compute weight gradients.
        free_data_positive: Initial free states for positive phase (list of arrays).
        clamped_data_positive: Clamped states for positive phase.
        free_data_negative: Initial free states for negative phase.
        clamped_data_negative: Clamped states for negative phase.
        beta: Inverse temperature parameter.
        correlation_penalty: Optional correlation penalty coefficient for regularization.

    Returns:
        Tuple of (weight_gradients, bias_gradients), each as JAX arrays matching model shapes.
    """

    key_pos, key_neg = jax.random.split(key, 2)

    batch_size_positive = free_data_positive[0].shape[0]
    batch_size_negative = free_data_negative[0].shape[0]

    assert clamped_data_positive[0].shape[0] == batch_size_positive, "positive batch size in free_data_positive and clamped_data_positive don't agree."
    if len(clamped_data_negative):
        assert clamped_data_negative[0].shape[0] == batch_size_negative, "negative batch size in free_data_negative and clamped_data_negative don't agree."
        assert len(clamped_data_negative) == len(training_spec.program_negative.gibbs_spec.clamped_blocks)

    assert len(free_data_positive) == len(training_spec.program_positive.gibbs_spec.free_blocks)
    assert len(clamped_data_positive) == len(training_spec.program_positive.gibbs_spec.clamped_blocks)
    assert len(free_data_negative) == len(training_spec.program_negative.gibbs_spec.free_blocks)

    weight_edge_tuples = [(edge.connected_nodes[0], edge.connected_nodes[1]) for edge in weight_edges]

    keys_pos = jax.random.split(key_pos, batch_size_positive)

    moms_b_pos, moms_w_pos = jax.vmap(
        lambda k, i, c: estimate_moments(
            k,
            bias_nodes,
            weight_edge_tuples,
            training_spec.program_positive,
            training_spec.schedule_positive,
            i,
            c,
        )
    )(keys_pos, free_data_positive, clamped_data_positive)

    assert moms_b_pos.shape == (batch_size_positive, len(bias_nodes))
    assert moms_w_pos.shape == (batch_size_positive, len(weight_edges))

    moms_b_pos = jnp.mean(moms_b_pos, axis=0)
    moms_w_pos = jnp.mean(moms_w_pos, axis=0)

    keys_neg = jax.random.split(key_neg, batch_size_negative)

    moms_b_neg, moms_w_neg = jax.vmap(
        lambda k, i, c: estimate_moments(
            k,
            bias_nodes,
            weight_edge_tuples,
            training_spec.program_negative,
            training_spec.schedule_negative,
            i,
            c,
        )
    )(keys_neg, free_data_negative, clamped_data_negative)

    assert moms_b_neg.shape == (batch_size_negative, len(bias_nodes))
    assert moms_w_neg.shape == (batch_size_negative, len(weight_edges))

    moms_b_neg = jnp.mean(moms_b_neg, axis=0)
    moms_w_neg = jnp.mean(moms_w_neg, axis=0)

    if correlation_penalty is not None:
        # build a node to edge spec
        node_map = {node: i for i, node in enumerate(bias_nodes)}
        edges_arr = np.zeros((len(weight_edges), 2), dtype=int)
        for i, edge in enumerate(weight_edges):
            for j in range(2):
                edges_arr[i, j] = node_map[edge.connected_nodes[j]]
        node_to_edge_spec = jnp.array(edges_arr)

        penalty_pos = jnp.prod(moms_b_neg[node_to_edge_spec], axis=1)
        penalty_neg = moms_w_neg
        cp = correlation_penalty * (penalty_pos - penalty_neg)
    else:
        cp = 0.0

    grad_b = -beta * (moms_b_pos - moms_b_neg)
    grad_w = -beta * (moms_w_pos - moms_w_neg + cp)

    # map bias_node and weight_edges gradients to indices in full model.graph
    full_grad_b = jnp.zeros_like(model.biases)
    bias_indices = jnp.array([model.graph.node_mapping[node] for node in bias_nodes])
    full_grad_b = full_grad_b.at[bias_indices].set(grad_b)

    # Map to full weights
    full_grad_w = jnp.zeros_like(model.weights)
    edge_indices = jnp.array([model.graph.edge_mapping[edge] for edge in weight_edges])
    full_grad_w = full_grad_w.at[edge_indices].set(grad_w)

    return full_grad_w, full_grad_b


@eqx.filter_jit
def do_epoch(
    key: Key[Array, ""],
    model: AbstractIsingEBMwithGraph,
    training_spec: BinomialIsingTrainingSpec,
    bias_nodes: list[AbstractNode],
    weight_edges: list[Edge],
    batch_size: int,
    data_positive: tuple[Array, ...],
    data_negative: tuple[Array, ...],
    beta: Array,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    weight_decay: Optional[FloatScalarLike] = None,
    bias_decay: Optional[FloatScalarLike] = None,
    correlation_penalty: Optional[FloatScalarLike] = None,
):
    """
    Performs one training epoch by processing batches and applying updates.
    
    Batches the positive and negative data, then uses a scanned loop to compute
    gradients and update parameters for each batch. Assumes the model and training_spec
    implement update_weights_and_biases methods for creating updated instances under the jit.
    
    Args:
        key: JAX random key for the epoch.
        model: Initial Ising EBM model.
        training_spec: Training specification.
        bias_nodes: Nodes for bias updates.
        weight_edges: Edges for weight updates.
        batch_size: Number of examples per batch.
        data_positive: Positive phase data tuple.
        data_negative: Negative phase data tuple.
        beta: Inverse temperature.
        optim: Optimizer.
        opt_state: Initial optimizer state.
        weight_decay: Optional weight decay coefficient.
        bias_decay: Optional bias decay coefficient.
        correlation_penalty: Optional correlation penalty.

    Returns:
        Updated weights, biases, and optimizer state.
    """

    weight_decay = 0.0 if weight_decay is None else weight_decay
    bias_decay = 0.0 if bias_decay is None else bias_decay

    def batch_data(key, data, _bsz, clamped_blocks):
        data_size = data[0].shape[0]
        _n_batches = data_size // _bsz
        tot_len = _n_batches * _bsz

        key, key_shuffle = jax.random.split(key)
        idxs = jax.random.permutation(key_shuffle, jnp.arange(data_size))
        assert len(clamped_blocks) == len(data)

        batched_data = []
        for arr, block in zip(data, clamped_blocks):
            assert arr.shape == (data_size, len(block))
            arr = jnp.reshape(arr[idxs][:tot_len], (_n_batches, _bsz, len(block)))
            # wrap arr in a tuple, which is supposed to range over SDs
            batched_data.append(arr)

        return batched_data, _n_batches

    key, key_pos, key_neg = jax.random.split(key, 3)

    batched_data_pos, n_batches = batch_data(
        key_pos, data_positive, batch_size, training_spec.program_positive.gibbs_spec.clamped_blocks
    )

    batched_data_neg, n_batches = batch_data(
        key_neg, data_negative, batch_size, training_spec.program_negative.gibbs_spec.clamped_blocks
    )


    @jax.jit
    def body_fun(carry, key_and_data):
        _key, _data_pos, _data_neg = key_and_data

        _opt_state, _params = carry
        _weights, _biases = _params

        _model = model.update_weights_and_biases(_weights, _biases) #this updates the global weights and biases; if we refactor hinton init to not use model's biases we could only update this after each epoch
        _training_spec = training_spec.update_weights_and_biases(_weights, _biases) # this updates the predetermined slices stored in BlockSamplingProgram of weights and biases actually used in sampling
        
        key_train, key_init_pos, key_init_neg = jax.random.split(_key, 3)

        vals_free_pos = hinton_init_from_graph(
            key_init_pos,
            _model,
            _training_spec.program_positive.gibbs_spec.free_blocks,
            batch_size,
            beta,
        )
        vals_free_neg = hinton_init_from_graph(
            key_init_neg,
            _model,
            _training_spec.program_negative.gibbs_spec.free_blocks,
            batch_size,
            beta,
        )

        grad_w, grad_b = symmetric_kl_grad(
            key_train,
            _model,
            _training_spec,
            bias_nodes,
            weight_edges,
            vals_free_pos,
            _data_pos,
            vals_free_neg,
            _data_neg,
            beta,
            correlation_penalty,
        )

        grads = (grad_w, grad_b)
        updates_without_decays, _opt_state = optim.update(grads, _opt_state, _params) # optim handles learning rate decay
        weight_updates_without_decay, bias_updates_without_decay = updates_without_decays
        weight_updates = (weight_updates_without_decay - _weights * weight_decay)
        bias_updates = (bias_updates_without_decay - _biases * bias_decay) 
        masked_updates = (weight_updates, bias_updates)
        _weights, _biases = eqx.apply_updates(_params, masked_updates)

        new_carry = _opt_state, (_weights, _biases)

        return new_carry, None

    params = model.weights, model.biases

    assert opt_state is not None

    init_carry = opt_state, params

    keys = jax.random.split(key, n_batches)
    out_carry, _ = jax.lax.scan(body_fun, init_carry, (keys, batched_data_pos, batched_data_neg))

    opt_state, params = out_carry
    new_weights, new_biases = params

    return new_weights, new_biases, opt_state
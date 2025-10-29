from typing_extensions import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from thrml.block_management import Block
from thrml.models.ebm import EBMFactor
from thrmlDenoising.annealing_graph_ising import AbstractIsingEBMwithGraph, BiasFactor, IsingNode, WeightFactor
from thrmlDenoising.pgm_continued import Edge
from thrmlDenoising.step_graph import DiffusionStepGraph

class DiffusionStepEBM(AbstractIsingEBMwithGraph):
    """Energy-based model for a single diffusion step using an Ising graph.

    Extends AbstractIsingEBMwithGraph with factors for couplings, biases,
    and base graph weights for diffusion-based generation.

    **Attributes:**
    - `graph`: The underlying diffusion step graph structure.
    - `image_coupling_factor`: Factor for image input-output couplings.
    - `label_coupling_factor`: Factor for label input-output couplings.
    - `bias_factor`: Bias factor for output and hidden nodes.
    - `base_graph_weight_factor`: Factor for weights in the base graph edges.
    """
    graph: DiffusionStepGraph

    image_coupling_factor: WeightFactor
    label_coupling_factor: WeightFactor
    bias_factor: BiasFactor #this is a bias factor for all nodes in the graph except the input nodes
    base_graph_weight_factor: WeightFactor

    def __init__(
        self,
        base_graph_image_nodes: list[IsingNode],
        base_graph_label_nodes: list[IsingNode],
        base_graph_hidden_nodes: list[IsingNode],
        base_graph_edges: list[Edge],
    ):
        """Initializes the EBM with factors based on the provided base graph components.

        Constructs the diffusion step graph and initializes factors for image/label couplings,
        biases on non-input nodes, and base graph weights. Calls the superclass initializer
        with zero-initialized biases and weights.

        **Arguments:**
        - `base_graph_image_nodes`: List of Ising nodes for image outputs from the base graph.
        - `base_graph_label_nodes`: List of Ising nodes for label outputs from the base graph.
        - `base_graph_hidden_nodes`: List of Ising nodes for hidden layers from the base graph.
        - `base_graph_edges`: List of edges connecting nodes in the base graph.
        """

        graph = DiffusionStepGraph(base_graph_image_nodes,
                                   base_graph_label_nodes,
                                   base_graph_hidden_nodes,
                                   base_graph_edges,
                                )

        # Make factors
        image_coupling_inds = jnp.array([graph.edge_mapping[e] for e in graph.image_coupling_edges])
        self.image_coupling_factor = WeightFactor(jnp.zeros(len(graph.image_coupling_edges)),
                            (Block([edge.connected_nodes[0] for edge in graph.image_coupling_edges]),
                             Block([edge.connected_nodes[1] for edge in graph.image_coupling_edges])),
                             image_coupling_inds)
        
        label_coupling_inds = jnp.array([graph.edge_mapping[e] for e in graph.label_coupling_edges])
        self.label_coupling_factor = WeightFactor(jnp.zeros(len(graph.label_coupling_edges)),
                            (Block([edge.connected_nodes[0] for edge in graph.label_coupling_edges]),
                             Block([edge.connected_nodes[1] for edge in graph.label_coupling_edges])),
                             label_coupling_inds)
        
        base_graph_edge_inds = jnp.array([graph.edge_mapping[e] for e in graph.base_graph_edges]) 
        self.base_graph_weight_factor = WeightFactor(jnp.zeros(len(graph.base_graph_edges)),
                            (Block([edge.connected_nodes[0] for edge in graph.base_graph_edges]), # it is ok to include repitions of nodes in blocks that are just handed to factors
                             Block([edge.connected_nodes[1] for edge in graph.base_graph_edges])),
                             base_graph_edge_inds)
        
        base_graph_nodes = graph.output_nodes + graph.hidden_nodes
        bias_inds = jnp.array([graph.node_mapping[n] for n in base_graph_nodes])
        self.bias_factor = BiasFactor(jnp.zeros(len(base_graph_nodes)), 
                                                    Block(base_graph_nodes),
                                                    bias_inds)
        
        super().__init__(
            graph,
            jnp.zeros(len(graph.nodes), dtype=jnp.float32),
            jnp.zeros(len(graph.edges), dtype=jnp.float32)
        )

    def set_coupling_weights(self, weights): 
        """Sets the coupling weights for image and label factors.

        Updates the global weights array and propagates the relevant slices to
        the image and label coupling factors.

        **Arguments:**
        - `weights`: New weights array for all edges in the graph.

        **Returns:**
        - Updated DiffusionStepEBM instance with new coupling weights.
        """
        model = eqx.tree_at(lambda m: m.weights, self, weights)

        # Update image_coupling_factor weights (for image_coupling_edges)
        image_coupling_indices = jnp.array([self.graph.edge_mapping[edge] for edge in self.graph.image_coupling_edges])
        new_image_coupling_weights = weights[image_coupling_indices]
        model = eqx.tree_at(lambda m: m.image_coupling_factor.weights, model, new_image_coupling_weights)

        # Update label_coupling_factor weights (for label_coupling_edges)
        label_coupling_indices = jnp.array([self.graph.edge_mapping[edge] for edge in self.graph.label_coupling_edges])
        new_label_coupling_weights = weights[label_coupling_indices]
        model = eqx.tree_at(lambda m: m.label_coupling_factor.weights, model, new_label_coupling_weights)

        return model

    def update_weights_and_biases(self, new_weights: Array, new_biases: Array) -> Self:
        """Updates the model's weights and biases.

        Replaces the current weights and biases with new values, returning
        an updated model instance. Used in training loops under JIT.

        **Arguments:**
        - `new_weights`: Updated array of edge weights.
        - `new_biases`: Updated array of node biases.

        **Returns:**
        - Updated DiffusionStepEBM instance with new weights and biases.
        """
        
        model = eqx.tree_at(lambda m: m.weights, self, new_weights)
        model = eqx.tree_at(lambda m: m.biases, model, new_biases)
        
        return model
    
    @property
    def factors(self) -> list[EBMFactor]:
        """Lists all EBM factors in the model.
        """
        return [self.image_coupling_factor, self.label_coupling_factor, self.bias_factor, self.base_graph_weight_factor]

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array

from thrml.block_management import Block
from thrmlDenoising.base_graphs.abstract_base_graph_manager import AbstractBaseGraphManager
from thrmlDenoising.annealing_graph_ising import IsingNode
from thrmlDenoising.pgm_continued import Edge

class PoissonBinomialIsingGraphManager(AbstractBaseGraphManager):
    """Handles graphs with Poisson-binomial encoding for grayscale images.
    
    Represents each pixel as n_trials independent Ising nodes (Bernoulli trials), summed for grayscale value.
    This structure lets the model learn arbitrary pmfs over levels: by tuning energies, it shapes probabilities
    for number of 'on' spins per pixel, enabling flexible distributions beyond uniform binomial.

    This offers continuity in grayscale transitions, but the Ising state space (2^{n_trials} per pixel) explodes 
    compared to actual levels (n_trials + 1), which is undesirable especially as n grows.
    
    Places all image trials in one block, labels in another. Builds bipartite grids for block Gibbs.
    
    **Attributes:**
    - `n_trials`: Trials per pixel, setting max grayscale (0 to n_trials).
    """

    n_trials: int

    def __init__(self, n_image_pixels: int,
                n_label_nodes: int,
                n_trials: int,
                ):
        """
        Sets up manager with single-block lengths for images (pixels * trials) and labels.
        """
        self.n_trials = n_trials

        image_output_block_lengths = [n_image_pixels * n_trials]  # Single block containing all image nodes
        label_output_block_lengths = [n_label_nodes]  # Single block for labels

        super().__init__(n_image_pixels, n_label_nodes, image_output_block_lengths, label_output_block_lengths)


    def convert_output_blocks_to_pixels(self, ising_data: list[Array]):
        """Sums binomial trials per pixel to recover integer grayscale values from single image block."""
        assert len(ising_data) == 1, "There is only one image output block in this base graph."
        ising_image_block_data = ising_data[0]
        assert ising_image_block_data.shape[-1] == self.n_image_pixels * self.n_trials, (
                    f"expected last dim {self.n_image_pixels * self.n_trials}, got {ising_image_block_data.shape[-1]}"
                )
        ising_image_block_data = ising_image_block_data.reshape(ising_image_block_data.shape[:-1] + (self.n_image_pixels, self.n_trials))
        
        return jnp.sum(ising_image_block_data, axis=-1, dtype=np.min_scalar_type(self.n_trials))


    def convert_pixels_to_output_blocks(self, image_data: Array) -> list[Array]:
        """Encodes pixels into binomial Ising states in one flat block for sampling."""
        assert image_data.shape[-1] == self.n_image_pixels, (
            f"Expected arrays of shape (..., {self.n_image_pixels}), got {image_data.shape}"
        )

        if self.n_trials == 1:
            return [image_data]
        
        return [jnp.reshape(
            (image_data[..., None] > jnp.arange(self.n_trials)).astype(jnp.bool_), 
            (image_data.shape[:-1] + (self.n_image_pixels * self.n_trials,))
        )]

    def convert_pixels_and_labels_to_input_block(self, image_data: Array, label_data: Array) -> Array:
        """Combines binomial-encoded pixels and labels into one input block for diffusion step inputs."""
        assert image_data.shape[-1] == self.n_image_pixels, (
            f"Expected arrays of shape (..., {self.n_image_pixels}), got {image_data.shape}"
        )
        assert label_data.shape[-1] == self.n_label_nodes, (
            f"Expected arrays of shape (..., {self.n_label_nodes}), got {label_data.shape}"
        )
        
        ising_image_data = jnp.reshape(
            (image_data[..., None] > jnp.arange(self.n_trials)).astype(jnp.bool_), 
            (image_data.shape[:-1] + (self.n_image_pixels * self.n_trials,))
        )
        input_data = jnp.concatenate([ising_image_data, label_data], axis=-1)
        assert input_data.shape[-1] == self.n_image_pixels * self.n_trials + self.n_label_nodes
        return input_data

    
    def convert_label_out_blocks_to_label(self, ising_data: list[Array]) -> Array:
        """Directly returns one-hot labels from single label block."""
        assert len(ising_data) == 1, "In this graph there is only one label output block"
        return ising_data[0]

    def convert_label_to_label_out_blocks(self, label_data: Array) -> list[Array]:
        """Wraps one-hot labels into single block for output consistency."""
        return [label_data]

    def make_base_graph(
        self,
        key,
        graph_preset_arch,
        n_image_pixels: int,
        n_label_nodes: int,
        torus: bool = False,
    ):
        """Builds bipartite square grid graph for block-parallel sampling.
        
        Preset defines side length and edge jumps ensuring bipartiteness (odd Manhattan distance).
        Grid colored chessboard-style: even-sum coords 'upper', odd 'lower'.
        All lower nodes are hidden. Visible nodes (image trials + labels) placed randomly in upper half.
        Remaining upper nodes become upper hidden. Edges connect upper-lower only.
        Blocks: one for all image nodes, one for labels, two hidden (upper/lower) for parallel updates.
        
        Torus wraps edges if enabled, requiring even side for bipartiteness.
        """
        if graph_preset_arch not in graph_preset_architectures:
            raise ValueError(f"Graph preset key '{graph_preset_arch}' not present in dictionary.")
        side_len, jumps = graph_preset_architectures[graph_preset_arch]

        size = side_len**2

        assert n_image_pixels > 0, "Nonzero image pixels should be used for each step graph."

        n_image_nodes = n_image_pixels * self.n_trials

        n_visible_nodes = n_image_nodes + n_label_nodes

        assert n_visible_nodes < (size // 2), (
            f"n_visible={n_visible_nodes} > size/2={size // 2}. The code's logic assumes all visible nodes can fit on the 'upper' half of the grid."
        )

        if torus:
            assert side_len % 2 == 0, "Torus is set to true and grid side length is odd, which will break bipartitness"

        for dx, dy in jumps:
            assert (dx + dy) % 2 == 1, (f"To ensure bipartitness for parallel sampling, jumps on the grid are assumed to only connect the same colors with a chessboard-like coloring scheme. Jump {(dx,dy)} violates this assumption.")

        def get_idx(i, j):
            if torus:
                i = (i + 10 * side_len) % side_len
                j = (j + 10 * side_len) % side_len

            cond = (i >= side_len) | (j >= side_len) | (i < 0) | (j < 0)
            return jnp.where(cond, -1, i * side_len + j)

        def get_coords(idx):
            return idx // side_len, (idx + side_len) % side_len

        def make_edge_single(idx, di, dj):
            def upper_node_first(_):
                return jnp.array([idx, get_idx(i + di, j + dj)])
            def lower_node_first(_):
                return jnp.array([get_idx(i + di, j + dj), idx])
            i, j = get_coords(idx)
            upper_first = (i + j) % 2 == 0
            return lax.cond(upper_first, upper_node_first, lower_node_first, None)

        make_edge_arr = jax.jit(
            jax.vmap(make_edge_single, in_axes=(0, None, None), out_axes=0)
        )

        indices = jnp.arange(size)
        edge_arrs_list = []

        for dx, dy in jumps:
            edges_I = make_edge_arr(indices, dx, dy)
            edges_II = make_edge_arr(indices, -dy, dx)
            edges_III = make_edge_arr(indices, -dx, -dy)
            edges_IV = make_edge_arr(indices, dy, -dx)
            edge_arrs_list.append(edges_I)
            edge_arrs_list.append(edges_II)
            edge_arrs_list.append(edges_III)
            edge_arrs_list.append(edges_IV)


        edge_array = np.concatenate(edge_arrs_list, axis=0)
        assert edge_array.shape == (4 * len(jumps) * size, 2)

        visible_indices = jr.permutation(key, jnp.arange((size // 2) - 1))[
            :n_visible_nodes
        ]
        visible_ind_set = set(visible_indices.tolist())

        image_indices = set(visible_indices[:n_image_nodes].tolist())
        label_indices = set(visible_indices[n_image_nodes:].tolist())

        nodes_upper = []
        nodes_lower = []
        all_grid_nodes = []
        full_to_upper = {}
        upper_count = 0
        for i in range(size):
            new_node = IsingNode()
            if (i // side_len + i % side_len) % 2 == 0:
                nodes_upper.append(new_node)
                full_to_upper[i] = upper_count
                upper_count += 1
            else:
                nodes_lower.append(new_node)
            all_grid_nodes.append(new_node)

        edges = []
        seen = set()
        for upper_ind_in_full, lower_ind_in_full in edge_array:
            if upper_ind_in_full == -1 or lower_ind_in_full == -1:
                continue
            upper_node = all_grid_nodes[upper_ind_in_full]
            lower_node = all_grid_nodes[lower_ind_in_full]
            edge = Edge((upper_node, lower_node))
            if edge in seen: 
                continue
            seen.add(edge)
            edges.append(edge)

        grid_edges = edges

        image_output_nodes = [nodes_upper[i] for i in image_indices]
        label_output_nodes = [nodes_upper[i] for i in label_indices]

        image_output_blocks = [Block(image_output_nodes)]
        label_output_blocks = [Block(label_output_nodes)]

        upper_hidden_indices = [
            i for i in range(len(nodes_upper)) if i not in visible_ind_set
        ]

        upper_hidden_nodes = [nodes_upper[i] for i in upper_hidden_indices]
        lower_hidden_nodes = nodes_lower
        hidden_nodes = upper_hidden_nodes + lower_hidden_nodes

        hidden_blocks = [Block(upper_hidden_nodes), Block(lower_hidden_nodes)]

        return image_output_nodes, label_output_nodes, hidden_nodes, grid_edges, image_output_blocks, label_output_blocks, hidden_blocks

graph_preset_architectures = {
    3_4: (3, [(0, 1), 
                ]
        ),   
    4_4: (4, [(0, 1), 
                ]
        ),    
    6_4: (6, [(0, 1), 
                ]
        ),   
    8_8: (8, [(0, 1), 
                (4, 1)
                ]
        ),
    20_8: (20, [(0, 1), 
                (4, 1)
                ]
        ),
    42_8: (42, [(0, 1), 
                  (4, 1)
                  ]
            ),
    44_8: (44, [(0, 1), 
        (4, 1), 
        ]
    ),
    44_12: (44, [(0, 1), 
        (4, 1), 
        (10, 9),
        ]
    ),
    44_16: (44, [(0, 1), 
            (4, 1), 
            (10, 9),
            (11, 14),
            ]
    ),
    44_20: (44, [(0, 1), 
            (4, 1), 
            (10, 9),
            (11, 14),
            (23, 6),
            ]
    ),
    50_8: (50, [(0, 1), 
        (4, 1), 
        ]
    ),
    50_12: (50, [(0, 1), 
        (4, 1), 
        (10, 9),
        ]
    ),
    50_16: (50, [(0, 1), 
            (4, 1), 
            (10, 9),
            (11, 14),
            ]
    ),
    50_20: (50, [(0, 1), 
            (4, 1), 
            (10, 9),
            (11, 14),
            (23, 6),
            ]
    ),
    60_8: (60, [(0, 1), 
            (4, 1), 
            ]
        ),
    60_12: (60, [(0, 1), 
               (4, 1), 
               (10, 9),
               ]
        ),
    60_16: (60, [(0, 1), 
        (4, 1), 
        (10, 9),
        (11, 14),
        ]
    ),
    60_20: (60, [(0, 1), 
               (4, 1), 
               (10, 9),
               (11, 14),
               (23, 6),
               ]
        ),
    60_32: (60, [(0, 1), 
               (4, 1), 
               (10, 9),
               (11, 14),
               (23, 6),
               (3, 28),
               (6, 33),
               (20, 35),
               ]
        ),
    60_40: (
        60,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
        ],
    ),
    60_44: (
        60,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
            (4, 5),
        ],
    ),
    70_8: (70, [(0, 1), 
        (4, 1), 
        ]
    ),
    70_12: (70, [(0, 1), 
               (4, 1), 
               (10, 9),
               ]
        ),
    70_16: (70, [(0, 1), 
        (4, 1), 
        (10, 9),
        (11, 14),
        ]
    ),
    70_20: (70, [(0, 1), 
            (4, 1), 
            (10, 9),
            (11, 14),
            (23, 6),
            ]
    ),
    70_24: (70, [(0, 1), 
               (4, 1), 
               (10, 9),
               (11, 14),
               (23, 6),
               (3, 28),
               ]
        ),
    70_36: (
        70,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
        ],
    ),
    70_52: (
        70,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
            (4, 5),
            (2, 9),
            (1, 14),
        ],
    ),
    80_8: (80, [(0, 1), 
        (4, 1), 
        ]
    ),
    80_12: (80, [(0, 1), 
        (4, 1), 
        (10, 9),
        ]
    ),
    80_16: (80, [(0, 1), 
        (4, 1), 
        (10, 9),
        (11, 14),
        ]
    ),
    80_20: (80, [(0, 1), 
        (4, 1), 
        (10, 9),
        (11, 14),
        (23, 6),
        ]
    ),
    80_24: (
        80,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
        ],
    ),
    80_36: (
        80,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
        ],
    ),
    80_48: (
        80,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
            (4, 5),
            (2, 9),
        ],
    ),
    80_60: (
        80,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
            (4, 5),
            (2, 9),
            (1, 14),
            (17, 8),
            (8, 21),
        ],
    ),
    90_8: (90, [(0, 1), 
        (4, 1), 
        ]
    ),
    90_12: (90, [(0, 1), 
        (4, 1), 
        (10, 9),
        ]
    ),
    90_16: (90, [(0, 1), 
        (4, 1), 
        (10, 9),
        (11, 14),
        ]
    ),
    90_20: (90, [(0, 1), 
        (4, 1), 
        (10, 9),
        (11, 14),
        (23, 6),
        ]
    ),
    100_40: (
        100,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
        ],
    ),
    100_60: (
        100,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
            (4, 5),
            (2, 9),
            (1, 14),
            (17, 8),
            (8, 21),
        ],
    ),
    100_76: (
        100,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
            (47, 20),
            (4, 5),
            (2, 9),
            (1, 14),
            (17, 8),
            (8, 21),
            (11, 26),
            (21, 24),
            (31, 18),
            (23, 34),
        ],
    ),
    120_36: (
        120,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
            (6, 33),
            (20, 35),
            (34, 29),
        ],
    ),
}
# === thrmlDenoising/base_graphs/convolved_poisson_binomial_ising_graph_manager.py ===
from __future__ import annotations

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

class ConvolvedPoissonBinomialIsingGraphManager(AbstractBaseGraphManager):
    """
    """

    n_trials: int

    # Precomputed index partitions for fast (de)serialization of blocks
    upper_pixel_indices: jnp.ndarray
    lower_pixel_indices: jnp.ndarray

    upper_image_trials_len: int
    lower_image_trials_len: int
    upper_label_len: int
    lower_label_len: int

    def __init__(
        self,
        n_image_pixels: int,
        n_label_nodes: int,
        n_trials: int,
    ):
        assert n_trials >= 1, "n_trials must be >= 1."
        self.n_trials = n_trials

        # Partition pixels by image-space parity (requires square images).
        side = int(np.sqrt(n_image_pixels))
        assert side * side == n_image_pixels, (
            f"Convolved manager requires square images; got n_image_pixels={n_image_pixels}"
        )
        img_idx = jnp.arange(n_image_pixels, dtype=jnp.int32)
        img_i = img_idx // side
        img_j = img_idx % side
        upper_mask = ((img_i + img_j) % 2) == 0

        self.upper_pixel_indices = img_idx[upper_mask]
        self.lower_pixel_indices = img_idx[~upper_mask]

        n_upper_pixels = int(self.upper_pixel_indices.shape[0])
        n_lower_pixels = int(self.lower_pixel_indices.shape[0])

        # Trials per pixel
        self.upper_image_trials_len = n_upper_pixels * n_trials
        self.lower_image_trials_len = n_lower_pixels * n_trials

        # Deterministic label split (first half to upper, remainder to lower).
        self.upper_label_len = n_label_nodes // 2
        self.lower_label_len = n_label_nodes - self.upper_label_len

        image_output_block_lengths = [self.upper_image_trials_len, self.lower_image_trials_len]
        label_output_block_lengths = [self.upper_label_len, self.lower_label_len]

        super().__init__(
            n_image_pixels,
            n_label_nodes,
            image_output_block_lengths,
            label_output_block_lengths,
            noise_image_data_in_pixel_space=True,
        )

    # ----------------- Image/Label data <-> block helpers (sum-of-trials) -----------------

    def convert_output_blocks_to_pixels(self, ising_data: list[Array]) -> Array:
        """
        Decode two image blocks (upper-trials, lower-trials) back to integer pixels.

        Each block is reshaped to (..., n_pixels_in_that_block, T) and summed along the last axis.
        We then concatenate (upper, lower) and permute back to **pixel order 0..P-1**.
        """
        assert len(ising_data) == 2, "Expected exactly 2 image output blocks (upper, lower)."
        upper_block, lower_block = ising_data

        # Reshape to (..., #pixels_in_block, T) then sum trials
        if self.n_trials == 1:
            upper_vals = upper_block
            lower_vals = lower_block
        else:
            upper_vals = upper_block.reshape(upper_block.shape[:-1] + (self.upper_image_trials_len // self.n_trials, self.n_trials)).sum(
                axis=-1, dtype=np.min_scalar_type(self.n_trials)
            )
            lower_vals = lower_block.reshape(lower_block.shape[:-1] + (self.lower_image_trials_len // self.n_trials, self.n_trials)).sum(
                axis=-1, dtype=np.min_scalar_type(self.n_trials)
            )

        # Concatenate in (upper, lower) block order
        concatenated = jnp.concatenate([upper_vals, lower_vals], axis=-1)

        # Build a permutation that maps (upper,lower) order → global pixel order [0..P-1].
        # We precompute the concatenated index list (upper first), then argsort to get
        # positions that yield pixel-major order.
        concat_positions = jnp.concatenate([self.upper_pixel_indices, self.lower_pixel_indices], axis=0)
        perm = jnp.argsort(concat_positions)

        pixels = concatenated[..., perm]
        return pixels

    def convert_pixels_to_output_blocks(self, image_data: Array) -> list[Array]:
        """
        Encode integer pixel values into **two Ising blocks** (upper, lower), using
        front-filled trials per pixel.

        Returns:
            [upper_block, lower_block] with shapes (..., #upper_pixels*T) and (..., #lower_pixels*T).
        """
        assert image_data.shape[-1] == self.n_image_pixels, (
            f"Expected arrays of shape (..., {self.n_image_pixels}), got {image_data.shape}"
        )

        upper_vals = image_data[..., self.upper_pixel_indices]
        lower_vals = image_data[..., self.lower_pixel_indices]

        if self.n_trials == 1:
            # For the binarized case, the single "trial" is just the 0/1 pixel value.
            upper_block = upper_vals
            lower_block = lower_vals
        else:
            trials = jnp.arange(self.n_trials)
            upper_bits = (upper_vals[..., None] > trials).astype(jnp.bool_)
            lower_bits = (lower_vals[..., None] > trials).astype(jnp.bool_)
            upper_block = upper_bits.reshape(upper_vals.shape[:-1] + (self.upper_image_trials_len,))
            lower_block = lower_bits.reshape(lower_vals.shape[:-1] + (self.lower_image_trials_len,))

        return [upper_block, lower_block]

    def convert_pixels_and_labels_to_input_block(self, image_data: Array, label_data: Array) -> Array:
        """
        Build the **conditioning input block** in the canonical order used by the model’s
        image/label input nodes:

        - Image trials come **in global pixel-major then trial-major order** across *all pixels*,
          i.e. not separated by halves. (This matches the order of `image_output_nodes` we build.)
        - Then raw label nodes in index order `0..n_label_nodes-1`.

        This order matches how the diffusion model creates coupling edges
        (each image/label input node couples 1-to-1 to the corresponding output node).
        """
        assert image_data.shape[-1] == self.n_image_pixels, (
            f"Expected arrays of shape (..., {self.n_image_pixels}), got {image_data.shape}"
        )
        assert label_data.shape[-1] == self.n_label_nodes, (
            f"Expected arrays of shape (..., {self.n_label_nodes}), got {label_data.shape}"
        )

        if self.n_trials == 1:
            ising_image_flat = image_data
        else:
            trials = jnp.arange(self.n_trials)
            bits = (image_data[..., None] > trials).astype(jnp.bool_)
            ising_image_flat = bits.reshape(image_data.shape[:-1] + (self.n_image_pixels * self.n_trials,))

        input_data = jnp.concatenate([ising_image_flat, label_data], axis=-1)
        assert input_data.shape[-1] == self.n_image_pixels * self.n_trials + self.n_label_nodes
        return input_data

    def convert_label_out_blocks_to_label(self, ising_data: list[Array]) -> Array:
        """
        Merge two label blocks (upper, lower) back to label index order.

        Because we split deterministically (first half upper, remainder lower),
        concatenating in that order already yields the correct label order.
        """
        assert len(ising_data) == 2, "Expected exactly 2 label output blocks (upper, lower)."
        upper_blk, lower_blk = ising_data
        return jnp.concatenate([upper_blk, lower_blk], axis=-1)

    def convert_label_to_label_out_blocks(self, label_data: Array) -> list[Array]:
        """
        Split labels by our deterministic partition:
        first `upper_label_len` indices to the upper label block, remainder to the lower block.
        """
        assert label_data.shape[-1] == self.n_label_nodes
        upper = label_data[..., : self.upper_label_len]
        lower = label_data[..., self.upper_label_len :]
        return [upper, lower]

    # ----------------- Graph builder -----------------

    def make_base_graph(
        self,
        key,
        graph_preset_arch,
        n_image_pixels: int,
        n_label_nodes: int,
        torus: bool = False,
    ):
        """
        Construct a **bipartite base grid** with **image-layer neighbor edges**, while
        ensuring **no block mixes upper & lower nodes**:

        Steps:

        1) Read `(side, jumps, image_jumps)` from the preset and assert each `(dx,dy)` has
           odd `(dx+dy)` to preserve bipartiteness (both for base-grid edges and image-neighbor edges).

        2) Build the base grid of size `side^2` and generate all **upper↔lower** base edges
           using `jumps` and 4 rotations (`(dx,dy),(-dy,dx),(-dx,-dy),(dy,-dx)`). Optional torus
           wrapping is supported **for base-grid** indices only.

        3) Split **image pixels by image parity** `((row+col)%2)`. All trials for an upper-parity
           pixel must be placed in the **upper** half of the base grid; all trials for a lower-parity
           pixel in the **lower** half. We **randomly permute** positions within each half so data
           nodes are randomly scattered but never violate parity.

        4) Split labels deterministically (`upper_label_len = n_label_nodes//2`); randomly place those
           upper/lower label nodes within their designated halves.

        5) Create **image-jump edges** for each `(dx,dy)` in `image_jumps` and for each **trial layer k**:
           connect trial-k of pixel A to trial-k of pixel B (if B exists). Parity and our placement
           guarantee these edges are **upper↔lower** only.

        6) Assemble:
            - `image_output_nodes` in **global pixel-major then trial-major order**.
            - `label_output_nodes` in **label index order**.
            - `image_output_blocks = [Block(upper_image_trials), Block(lower_image_trials)]`.
            - `label_output_blocks = [Block(upper_labels), Block(lower_labels)]`.
            - `hidden_blocks = [Block(upper_hidden), Block(lower_hidden)]`.

           Because each block contains nodes from **only one half**, and all edges are
           **upper↔lower**, `assert_no_intrablock_edges` will hold.

        Notes:
        - Image-space “torus” wrapping for `image_jumps` is **not** applied; out-of-bounds neighbors
          are simply ignored (as in the earlier commented prototype). Base-grid torus remains supported.
        """
        if graph_preset_arch not in graph_preset_architectures:
            raise ValueError(f"Graph preset key '{graph_preset_arch}' not present in dictionary.")
        base_side, base_jumps, image_jumps = graph_preset_architectures[graph_preset_arch]

        size = base_side ** 2
        n_lower_nodes = size // 2
        n_upper_nodes = size - n_lower_nodes

        # Sanity: capacity for data in each half
        assert (self.upper_image_trials_len + self.upper_label_len) < n_upper_nodes, (
            f"Upper data nodes ({self.upper_image_trials_len + self.upper_label_len}) must fit in upper half ({n_upper_nodes})."
        )
        assert (self.lower_image_trials_len + self.lower_label_len) < n_lower_nodes, (
            f"Lower data nodes ({self.lower_image_trials_len + self.lower_label_len}) must fit in lower half ({n_lower_nodes})."
        )

        if torus:
            assert base_side % 2 == 0, "Torus is true but base-grid side length is odd, which breaks bipartiteness."

        # Parity assertions for both sets of jumps
        for dx, dy in base_jumps:
            assert (dx + dy) % 2 == 1, (
                f"Base-grid jump {(dx, dy)} must have odd (dx+dy) to preserve upper↔lower bipartiteness."
            )
        for dx, dy in image_jumps:
            assert (dx + dy) % 2 == 1, (
                f"Image jump {(dx, dy)} must have odd (dx+dy) so image neighbors are opposite parity."
            )

        # --- helpers for the base grid (full graph) ---
        def get_idx(i, j):
            if torus:
                i = (i + 10 * base_side) % base_side
                j = (j + 10 * base_side) % base_side
            cond = (i >= base_side) | (j >= base_side) | (i < 0) | (j < 0)
            return jnp.where(cond, -1, i * base_side + j)

        def get_coords(idx):
            return idx // base_side, idx % base_side

        def make_edge_single(idx, di, dj):
            # Pair is (upper, lower) for determinism.
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

        full_indices = jnp.arange(size, dtype=jnp.int32)

        # Base-grid edges (4 rotations).
        edge_arrs_list = []
        for dx, dy in base_jumps:
            edges_I = make_edge_arr(full_indices, dx, dy)
            edges_II = make_edge_arr(full_indices, -dy, dx)
            edges_III = make_edge_arr(full_indices, -dx, -dy)
            edges_IV = make_edge_arr(full_indices, dy, -dx)
            edge_arrs_list.extend([edges_I, edges_II, edges_III, edges_IV])

        # Split full grid into halves by parity.
        full_i = full_indices // base_side
        full_j = full_indices % base_side
        upper_mask = ((full_i + full_j) % 2) == 0
        upper_indices = full_indices[upper_mask]
        lower_indices = full_indices[~upper_mask]
        assert int(upper_indices.shape[0]) == n_upper_nodes
        assert int(lower_indices.shape[0]) == n_lower_nodes

        # ---------------- Place DATA nodes randomly in their designated halves ----------------
        key_u, key_l = jr.split(key, 2)
        upper_perm = jr.permutation(key_u, upper_indices)
        lower_perm = jr.permutation(key_l, lower_indices)

        # Upper half: first image trials, then label nodes
        upper_image_full = upper_perm[: self.upper_image_trials_len]
        upper_label_full = upper_perm[
            self.upper_image_trials_len : self.upper_image_trials_len + self.upper_label_len
        ]

        # Lower half: first image trials, then label nodes
        lower_image_full = lower_perm[: self.lower_image_trials_len]
        lower_label_full = lower_perm[
            self.lower_image_trials_len : self.lower_image_trials_len + self.lower_label_len
        ]

        # Build (pixel, trial) -> full index mapping with front-fill trial order
        T = self.n_trials
        n_upper_pixels = self.upper_image_trials_len // T
        n_lower_pixels = self.lower_image_trials_len // T

        upper_image_full_2d = (
            upper_image_full.reshape(n_upper_pixels, T) if n_upper_pixels > 0 else jnp.zeros((0, T), dtype=jnp.int32)
        )
        lower_image_full_2d = (
            lower_image_full.reshape(n_lower_pixels, T) if n_lower_pixels > 0 else jnp.zeros((0, T), dtype=jnp.int32)
        )

        # Global mapping pixel->(trial indices in full grid)
        image_trial_to_full = jnp.full((self.n_image_pixels, T), -1, dtype=jnp.int32)
        # Fill upper pixels
        if n_upper_pixels > 0:
            image_trial_to_full = image_trial_to_full.at[self.upper_pixel_indices, :].set(upper_image_full_2d)
        # Fill lower pixels
        if n_lower_pixels > 0:
            image_trial_to_full = image_trial_to_full.at[self.lower_pixel_indices, :].set(lower_image_full_2d)

        # ---------------- Image-jump edges (layer-wise) ----------------
        img_side = int(np.sqrt(self.n_image_pixels))
        img_indices = jnp.arange(self.n_image_pixels, dtype=jnp.int32)

        def get_img_coords(idx_img):
            return idx_img // img_side, idx_img % img_side

        def get_img_idx(i, j):
            # No torus in image space; out-of-bounds → -1
            cond = (i >= img_side) | (j >= img_side) | (i < 0) | (j < 0)
            return jnp.where(cond, -1, i * img_side + j)

        def make_edge_for_image_single_trial(idx_img, di, dj, t):
            # Map image-neighborhood to full-graph nodes; keep a consistent order (upper first).
            i, j = get_img_coords(idx_img)
            nbr_img = get_img_idx(i + di, j + dj)

            idx_in_full = image_trial_to_full[idx_img, t]
            nbr_in_full = jnp.where(nbr_img == -1, jnp.int32(-1), image_trial_to_full[nbr_img, t])

            upper_first = ((i + j) % 2) == 0
            return jnp.where(
                upper_first,
                jnp.stack([idx_in_full, nbr_in_full]),
                jnp.stack([nbr_in_full, idx_in_full]),
            )

        vmapped_img_edges = jax.jit(
            jax.vmap(make_edge_for_image_single_trial, in_axes=(0, None, None, None), out_axes=0)
        )

        # For each trial layer, add the rotated image-jump edges
        for t in range(T):
            for dx, dy in image_jumps:
                edges_I = vmapped_img_edges(img_indices, dx, dy, t)
                edges_II = vmapped_img_edges(img_indices, -dy, dx, t)
                edges_III = vmapped_img_edges(img_indices, -dx, -dy, t)
                edges_IV = vmapped_img_edges(img_indices, dy, -dx, t)
                edge_arrs_list.extend([edges_I, edges_II, edges_III, edges_IV])

        # ---------------- Create nodes (one per full-graph location) ----------------
        all_nodes = [IsingNode() for _ in range(size)]

        # Image output nodes (global list): pixel-major, then trial-major
        image_output_nodes = []
        for p in range(self.n_image_pixels):
            for t in range(T):
                full_idx = int(image_trial_to_full[p, t])
                image_output_nodes.append(all_nodes[full_idx])

        # Label output nodes: build label_idx -> full index mapping (deterministic split)
        # upper labels: indices [0, upper_label_len), lower labels: [upper_label_len, n_labels)
        label_idx_to_full = jnp.full((self.n_label_nodes,), -1, dtype=jnp.int32)
        if self.upper_label_len > 0:
            label_idx_to_full = label_idx_to_full.at[: self.upper_label_len].set(upper_label_full)
        if self.lower_label_len > 0:
            label_idx_to_full = label_idx_to_full.at[self.upper_label_len :].set(lower_label_full)

        label_output_nodes = [all_nodes[int(x)] for x in np.asarray(label_idx_to_full)]

        # Hidden nodes are the remaining nodes (upper + lower halves) not used for image or labels
        used_full_upper = set(np.asarray(jnp.concatenate([upper_image_full, upper_label_full]), dtype=int).tolist())
        used_full_lower = set(np.asarray(jnp.concatenate([lower_image_full, lower_label_full]), dtype=int).tolist())

        upper_hidden_nodes = [all_nodes[int(x)] for x in np.asarray(upper_indices) if int(x) not in used_full_upper]
        lower_hidden_nodes = [all_nodes[int(x)] for x in np.asarray(lower_indices) if int(x) not in used_full_lower]
        hidden_nodes = upper_hidden_nodes + lower_hidden_nodes

        # ---------------- Build unique Edge objects from all proposed edges ----------------
        edge_array = np.concatenate(edge_arrs_list, axis=0)
        # Expected rows: base grid + image jumps (each with 4 rotations)
        expected_rows = 4 * len(base_jumps) * size + 4 * len(image_jumps) * self.n_image_pixels * T
        assert edge_array.shape == (expected_rows, 2), (
            f"Edge matrix shape mismatch: got {edge_array.shape}, expected {(expected_rows, 2)}"
        )

        grid_edges = []
        seen = set()
        for idx0, idx1 in edge_array:
            if (idx0 == -1) or (idx1 == -1):
                continue
            n0 = all_nodes[int(idx0)]
            n1 = all_nodes[int(idx1)]
            e = Edge((n0, n1))
            if e in seen:
                continue
            seen.add(e)
            grid_edges.append(e)

        # ---------------- Blocks: split by half so no block mixes connected nodes ----------------
        # Image blocks (upper trials, lower trials) in per-block pixel-major/trial-major order
        upper_image_nodes_block = []
        for p in list(np.asarray(self.upper_pixel_indices)):
            for t in range(T):
                full_idx = int(image_trial_to_full[p, t])
                upper_image_nodes_block.append(all_nodes[full_idx])

        lower_image_nodes_block = []
        for p in list(np.asarray(self.lower_pixel_indices)):
            for t in range(T):
                full_idx = int(image_trial_to_full[p, t])
                lower_image_nodes_block.append(all_nodes[full_idx])

        image_output_blocks = [Block(upper_image_nodes_block), Block(lower_image_nodes_block)]

        # Label blocks (upper then lower) in label index order
        upper_label_nodes_list = [all_nodes[int(x)] for x in np.asarray(upper_label_full)]
        lower_label_nodes_list = [all_nodes[int(x)] for x in np.asarray(lower_label_full)]
        label_output_blocks = [Block(upper_label_nodes_list), Block(lower_label_nodes_list)]

        hidden_blocks = [Block(upper_hidden_nodes), Block(lower_hidden_nodes)]

        return (
            image_output_nodes,
            label_output_nodes,
            hidden_nodes,
            grid_edges,
            image_output_blocks,
            label_output_blocks,
            hidden_blocks,
        )

# ----------------- Architectures (with image_jumps) -----------------
# Tuples are: (side_len, jumps, image_jumps)
graph_preset_architectures = {
    6_4_1: (
        6,
        [
            (0, 1),
        ],
        [
            (0, 1),  # image 4-neighborhood (via rotations)
        ],
    ),
    8_8_1: (
        8,
        [
            (0, 1),
            (4, 1),
        ],
        [
            (0, 1),  # image 4-neighborhood (via rotations)
        ],
    ),
    60_12_1: (
        60,
        [
            (0, 1),
            (4, 1),
            (10, 9),
        ],
        [
            (0, 1),
        ],
    ),
    60_24_1: (
        60,
        [
            # Use a 24-degree set similar in spirit to the 80_24 config from other managers.
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
        ],
        [
            (0, 1),
        ],
    ),
    70_36_1: (
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
        [
            (0, 1),
        ],
    ),
    80_12_1: (
        80,
        [
            (0, 1),
            (4, 1),
            (10, 9),
        ],
        [
            (0, 1),
        ],
    ),
    80_24_1: (
        80,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
        ],
        [
            (0, 1),
        ],
    ),
    90_36_1: (
        90,
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
        [
            (0, 1),
        ],
    ),
    90_44_1: (
        90,
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
        [
            (0, 1),
        ],
    ),
    108_60_1: (
        108,
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
        [
            (0, 1),
        ],
    ),
    120_24_1: (
        120,
        [
            (0, 1),
            (6, 1),
            (12, 5),
            (11, 14),
            (23, 6),
            (3, 28),
        ],
        [
            (0, 1),
        ],
    ),
    120_36_1: (
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
        [
            (0, 1),
        ],
    ),
    120_72_1: (
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
            (47, 20),
            (4, 5),
            (2, 9),
            (1, 14),
            (17, 8),
            (8, 21),
            (11, 26),
            (21, 24),
            (31, 18),
        ],
        [
            (0, 1),
        ],
    ),
    180_36_1: (
        180,
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
        [
            (0, 1),
        ],
    ),
     180_72_1: (
        180,
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
        ],
        [
            (0, 1),
        ],
    ),
}
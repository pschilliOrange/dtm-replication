import abc
import numpy as np
from jax import lax
from jaxtyping import ArrayLike, Array, PRNGKeyArray
from typing import Optional
import jax.random as jr
import equinox as eqx
import optax

import jax
import jax.numpy as jnp
from jaxtyping import Array

from thrmlDenoising.annealing_graph_ising import IsingNode
from thrml.block_management import Block
from thrmlDenoising.pgm_continued import Edge

class AbstractBaseGraphManager(eqx.Module):
    """
    Abstract base class for managing base graphs in Ising-based models for diffusion processes.

    This class provides an interface for constructing sparse graphs (nodes and edges) and handling conversions
    between pixel/label representations and block structures used in sampling programs. It is designed to support
    different encoding strategies (e.g., Poisson binomial or binary representations) for image and label data.

    Subclasses must implement the abstract methods to define specific graph construction and conversion logic.

    **Attributes:**
    - `n_image_pixels` (int): Number of pixels in each image.
    - `n_label_nodes` (int): Number of nodes representing labels.
    - `image_output_block_lengths` (list[int]): Length of each image output block.
    - `label_output_block_lengths` (list[int]): Length of each label output block.
    - `noise_image_data_in_pixel_space` (bool): Flag indicating whether to apply noise to image data in pixel space (defaults to True).
        Setting to false would require an extra method to convert from ising image and label to input block
        and calling that method in DiffusionStep's _make_training_data, which it currently does not.
    """

    n_image_pixels: int
    n_label_nodes: int

    image_output_block_lengths: list[int]
    label_output_block_lengths: list[int]

    noise_image_data_in_pixel_space: bool

    def __init__(self, n_image_pixels, n_label_nodes, image_output_block_lengths, label_output_block_lengths, noise_image_data_in_pixel_space: Optional[bool] = None):
        """
        Initialize the graph manager with image and label dimensions.
        """
        self.n_image_pixels = n_image_pixels
        self.n_label_nodes = n_label_nodes
        self.image_output_block_lengths = image_output_block_lengths
        self.label_output_block_lengths = label_output_block_lengths
        if noise_image_data_in_pixel_space is not None:
            self.noise_image_data_in_pixel_space = noise_image_data_in_pixel_space
        else:
            self.noise_image_data_in_pixel_space = True

    @abc.abstractmethod
    def convert_output_blocks_to_pixels(self, ising_data: list[Array]) -> Array:
        """
        Convert Ising-encoded data from output blocks back to pixel values.

        This method aggregates data from multiple output blocks into a single array of pixel values.

        **Arguments:**
        - `ising_data` (list[Array]): List of arrays containing ising-encoded data from output blocks.

        **Returns:**
        - `Array`: Array of pixel values, with shape agnostic to all dimensions except the last (pixels).
        """
        return NotImplemented

    @abc.abstractmethod
    def convert_pixels_to_output_blocks(self, image_data: Array) -> list[Array]:
        """
        Convert pixel values to ising-encoded data in output block format.

        This method splits pixel data into blocks suitable for sampling (e.g., encoding integers into binary or binomial representations).

        **Arguments:**
        - `image_data` (Array): Array of pixel values, with last dimension equal to `n_image_pixels`.

        **Returns:**
        - `list[Array]`: List of arrays representing the data in output block format.
        """
        return NotImplemented

    @abc.abstractmethod
    def convert_pixels_and_labels_to_input_block(self, image_data: Array, label_data: Array) -> Array:
        """
        Convert pixel and label data into a single input block array.

        This combines image pixels and labels into a format suitable for input to a diffusion step or sampling program.

        **Arguments:**
        - `image_data` (Array): Array of pixel values, with last dimension equal to `n_image_pixels`.
        - `label_data` (Array): Array of label values, with last dimension equal to `n_label_nodes`.

        **Returns:**
        - `Array`: Combined input block array.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def convert_label_out_blocks_to_label(self, ising_data: list[Array]) -> Array:
        """
        Convert Ising-encoded label data from output blocks back to label values.

        **Arguments:**
        - `ising_data` (list[Array]): List of arrays containing Ising-encoded label data from output blocks.

        **Returns:**
        - `Array`: Array of label values.
        """
        pass

    @abc.abstractmethod
    def convert_label_to_label_out_blocks(self, label_data: Array) -> list[Array]:
        """
        Convert label values to Ising-encoded data in output block format.

        **Arguments:**
        - `label_data` (Array): Array of label values, with last dimension equal to `n_label_nodes`.

        **Returns:**
        - `list[Array]`: List of arrays representing the label data in output block format.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def make_base_graph(self, *args, **kwargs) -> tuple[list[IsingNode], list[IsingNode], list[IsingNode], list[Edge], list[Block], list[Block], list[Block]]:
        """
        Construct the base graph structure, including nodes, edges, and blocks. The base graph structure consists of
        all output data nodes' connections with all hidden nodes.

        This method generates the underlying sparse graph (nodes and edges) and partitions nodes into blocks
        for image outputs, label outputs, and hidden nodes. If two nodes connected by an edge are passed in the 
        same block there will a ValueError in sampling_specs which checks for this to ensure proper sampling.

        Loading and saving models assumes that the graph is deterministically determined by a PRNG key in its input.
        If a graph breaks this assumption, (eg. by passing lists to sets) loading and saving will break.

        **Arguments:**
        - `*args`: Variable positional arguments to allow for potentially different arguments across managers
        - `**kwargs`: Variable keyword arguments to allow for potentially different arguments across managers

        **Returns:**
        - `tuple`: Containing image output nodes, label output nodes, hidden nodes, edges, image output blocks,
          label output blocks, and hidden blocks.
        """
        return NotImplemented
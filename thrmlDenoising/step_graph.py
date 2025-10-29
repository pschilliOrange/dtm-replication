from thrmlDenoising.annealing_graph_ising import IsingNode
from thrmlDenoising.pgm_continued import Edge, SparseGraph 

class DiffusionStepGraph(SparseGraph):
    """A specialized sparse graph for a single diffusion step in an Ising-based model.

    Extends SparseGraph by adding input nodes for images and labels, creating coupling edges
    between input and output nodes, and organizing nodes/edges into categorized attributes
    for easier access in diffusion processes. The base graph provides the output (visible)
    and hidden nodes, while this class adds corresponding input nodes and bidirectional
    couplings to model diffusion interactions.

    **Attributes:**
    - `input_nodes`: All input nodes (image + label inputs).
    - `image_input_nodes`: Input nodes for image data.
    - `label_input_nodes`: Input nodes for label data.
    - `output_nodes`: All output nodes (image + label outputs from base graph).
    - `image_output_nodes`: Output nodes for image data (from base graph).
    - `label_output_nodes`: Output nodes for label data (from base graph).
    - `hidden_nodes`: Hidden nodes (from base graph).
    - `coupling_edges`: All coupling edges (image + label couplings).
    - `image_coupling_edges`: Edges connecting image input to output nodes.
    - `label_coupling_edges`: Edges connecting label input to output nodes.
    - `base_graph_edges`: Edges from the provided base graph.
    """
    #nodes
    input_nodes: list[IsingNode]
    image_input_nodes: list[IsingNode]
    label_input_nodes: list[IsingNode]

    output_nodes: list[IsingNode]
    image_output_nodes: list[IsingNode]
    label_output_nodes: list[IsingNode]

    hidden_nodes: list[IsingNode]

    #edges
    coupling_edges: list[Edge]
    image_coupling_edges: list[Edge]
    label_coupling_edges: list[Edge]

    base_graph_edges: list[Edge]

    def __init__(self, 
            base_graph_image_nodes,
            base_graph_label_nodes,
            base_graph_hidden_nodes,
            base_graph_edges,
        ):
        """Initializes the diffusion step graph by extending the base graph.

        Creates input nodes matching the number of image and label output nodes,
        adds coupling edges between each input-output pair, and initializes the
        superclass with the combined nodes and edges.

        **Arguments:**
        - `base_graph_image_nodes`: List of Ising nodes representing image outputs from the base graph.
        - `base_graph_label_nodes`: List of Ising nodes representing label outputs from the base graph.
        - `base_graph_hidden_nodes`: List of Ising nodes representing hidden nodes from the base graph.
        - `base_graph_edges`: List of edges connecting nodes in the base graph.
        """

        self.image_output_nodes = base_graph_image_nodes
        self.label_output_nodes = base_graph_label_nodes
        self.output_nodes = self.image_output_nodes + self.label_output_nodes

        self.hidden_nodes = base_graph_hidden_nodes
        self.base_graph_edges = base_graph_edges

        self.image_input_nodes = [IsingNode() for _ in range(len(base_graph_image_nodes))]
        self.label_input_nodes = [IsingNode() for _ in range(len(base_graph_label_nodes))]
        self.input_nodes = self.image_input_nodes + self.label_input_nodes

        self.image_coupling_edges = [
            Edge((input_image_node, output_image_node)) 
            for input_image_node, output_image_node in zip(self.image_input_nodes, base_graph_image_nodes)
        ]
        self.label_coupling_edges = [
            Edge((input_label_node, output_label_node)) 
            for input_label_node, output_label_node in zip(self.label_input_nodes, base_graph_label_nodes)
        ]
        self.coupling_edges = self.image_coupling_edges + self.label_coupling_edges

        super().__init__(self.input_nodes + self.output_nodes + self.hidden_nodes, self.coupling_edges + self.base_graph_edges)
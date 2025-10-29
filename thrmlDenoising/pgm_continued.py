from dataclasses import dataclass
from typing import Sequence

from thrml.pgm import AbstractNode, _UniqueID

@dataclass(frozen=True, eq=False)
class Edge:
    """The basic undirected Edge class."""

    __slots__ = ("_connected_nodes",)
    _connected_nodes: tuple[AbstractNode, AbstractNode]

    @property
    def connected_nodes(self) -> tuple[AbstractNode, AbstractNode]:
        return self._connected_nodes

    def __hash__(self):
        return hash(frozenset(self.connected_nodes))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return frozenset(self.connected_nodes) == frozenset(other.connected_nodes)
    
"""
Even though thrml does not natively support a SparseGraph class, it is included in 
    thrmlDenoising for use of node and edge mapping as a global reference.
"""
class SparseGraph(_UniqueID):
    """
    A SparseGraph is merely a collection of nodes and edges, as well
    as supporting incidence and index mapping.

    Attributes:
        nodes: the list of nodes
        edges: the list of edges
        edge_incidence: a list over the nodes, in which each edge_incidence[i]
            is a list over which edges that node is connected to
        node_incidence: a list over the nodes, in which each node_incidence[i]
            is a list over which nodes that node is connected to
        node_mapping: a dictionary that maps each node to its order in the
            node list
        edge_mapping: a dictionary which maps each edge to its position in the
            edge list
    """

    nodes: list[AbstractNode]
    edges: list[Edge]
    edge_incidence: list[list[Edge]]
    node_incidence: list[list[AbstractNode]]
    node_mapping: dict[AbstractNode, int]
    edge_mapping: dict[Edge, int]

    def __init__(
        self, nodes: Sequence[AbstractNode], edges: list[Edge]
    ) -> None:
        node_mapping = {node: i for i, node in enumerate(nodes)}

        node_node_sets = {node: set() for node in nodes}

        # Note: edge_incidence[node] and node_incidence[node] must be in the
        # same order to each other. This is why we not only create the sets
        # (which are used for membership checks) but also the lists (which
        # make sure the order is preserved).
        node_edge_lists = {node: [] for node in nodes}
        node_node_lists = {node: [] for node in nodes}

        unique_edges = []
        edge_set = set()

        def add_incidence(edge, from_node, to_node):
            if to_node not in node_node_sets[from_node]:
                node_node_sets[from_node].add(to_node)
                node_node_lists[from_node].append(to_node)
                node_edge_lists[from_node].append(edge)
                if edge not in edge_set:
                    unique_edges.append(edge)
                    edge_set.add(edge)

        for edge in edges:
            this_node, other_node = edge.connected_nodes
            add_incidence(edge, this_node, other_node)
            add_incidence(edge, other_node, this_node)

        assert len(unique_edges) == len(edges), "edges passed to SparseGraph are not unique, which might cause incorrect use of mappings"

        edge_mapping = dict(zip(unique_edges, list(range(len(unique_edges)))))
        edge_incidence = [node_edge_lists[x] for x in nodes]
        node_incidence = [node_node_lists[x] for x in nodes]

        self.nodes = nodes
        self.edges = edges
        self.edge_incidence = edge_incidence
        self.node_incidence = node_incidence
        self.node_mapping = node_mapping
        self.edge_mapping = edge_mapping
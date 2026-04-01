from typing import List, Dict, Tuple
from rips.utils import FKComponent

__all__ = [
    # Undirected graph
    'VertexList', 'Vertex', 'EdgeList', 'Edge', 'EdgeIndex', 'Graph',
    # Directed graph
    'NodeList', 'Node', 'ArcList', 'Arc', 'ArcIndex', 'DiGraph']

Node = int
NodeList = List[int]
ArcIndex = int
Arc = Tuple[Node, Node]
ArcList = List[Arc]


class DiGraph(FKComponent):
    """Base class for directed graphs."""

    def __init__(self, arcs: ArcList, **kwargs):
        self.arcs = arcs
        self.nodes = self._get_nodes()
        self.successors = self._getsuccessors()
        self.predecessors = self._getpredecessors()
        super().__init__(**kwargs)

    def _get_nodes(self):
        n = max(node + 1 for e in self.arcs for node in e)
        return list(range(n))

    @property
    def num_arcs(self) -> int:
        """Number of arcs."""
        return len(self.arcs)

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def _getsuccessors(self) -> List[Dict[Node, ArcIndex]]:
        node_successors = [{} for _ in self.nodes]
        for e, (u, v) in enumerate(self.arcs):
            node_successors[u][v] = e
        return node_successors

    def _getpredecessors(self) -> List[Dict[Node, ArcIndex]]:
        node_predecessors = [{} for _ in self.nodes]
        for e, (u, v) in enumerate(self.arcs):
            node_predecessors[v][u] = e
        return node_predecessors

    def visitor(self, source: Node, arc_up: List[bool]) -> List[bool]:
        """List testing that ith node is visited.

        Args
            source: source node.
            arc_up: state of arcs (up = True, down = False).

        """
        visited = [False] * self.num_nodes
        visited[source] = True
        queue = [source]
        while not not queue:
            node = queue.pop()
            for neighbor in self.successors[node]:
                if not visited[neighbor]:
                    index = self.successors[node][neighbor]
                    if arc_up[index]:
                        queue.append(neighbor)
                        visited[neighbor] = True
        return visited


Vertex = int
VertexList = List[Vertex]
EdgeIndex = int
Edge = Tuple[Vertex, Vertex]
EdgeList = List[Edge]


class Graph(FKComponent):
    """Base class for undirected graphs."""

    def __init__(self, edges: EdgeList, **kwargs):
        self.edges = edges
        self.vertices = self._get_vertices()
        self.adj = self._getadj()
        super().__init__(**kwargs)

    def _get_vertices(self):
        n = max(vertex + 1 for e in self.edges for vertex in e)
        return list(range(n))

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    def _getadj(self) -> List[List[Tuple[Vertex, EdgeIndex]]]:
        adj = [[] for _ in self.vertices]
        for e, (u, v) in enumerate(self.edges):
            adj[u].append((v, e))
            adj[v].append((u, e))
        return adj

    def visitor(self, source: Vertex, edge_up: List[bool]) -> List[bool]:
        """List testing that ith vertex is visited.

        Args
            source: source vertex.
            edge_up: state of edges (up = True, down = False).

        """
        visited = [False] * self.num_vertices
        visited[source] = True
        queue = [source]
        while not not queue:
            node = queue.pop()
            for neighbor, index in self.adj[node]:
                if not visited[neighbor]:
                    if edge_up[index]:
                        queue.append(neighbor)
                        visited[neighbor] = True
        return visited

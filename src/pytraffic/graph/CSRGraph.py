import numpy as np
import heapq
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CSRGraph:
    """
    CSR формат для направденного графа.

    Nodes: 0 .. n-1
    Edges: 0 .. m-1
    """
    n: int                      # number of nodes
    m: int                      # number of edges
    tail: np.ndarray            # (m,) int32, edge start
    head: np.ndarray            # (m,) int32, edge end
    first_out: np.ndarray       # (n+1,) int32, CSR pointer
    out_eid: np.ndarray         # (m,) int32, edge ids sorted by tail


    @staticmethod
    def from_edges(n_nodes: int, tail: np.ndarray, head: np.ndarray) -> "CSRGraph":
        """
        Build CSRGraph from arrays tail/head.
        """
        tail = np.asarray(tail, dtype=np.int32)
        head = np.asarray(head, dtype=np.int32)
        m = int(tail.size)

        # sort edges by tail (stable)
        order = np.argsort(tail, kind="stable").astype(np.int32)
        tail_sorted = tail[order]

        # build CSR indptr
        first_out = np.zeros(n_nodes + 1, dtype=np.int32)
        np.add.at(first_out, tail_sorted + 1, 1)
        first_out = np.cumsum(first_out, dtype=np.int32)

        return CSRGraph(
            n=int(n_nodes),
            m=m,
            tail=tail,
            head=head,
            first_out=first_out,
            out_eid=order,
        )

    def dijkstra(self, weight: np.ndarray, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dijkstra shortest paths from source.

        weight : (m,) edge weights
        source : start node

        return:
          dist[v] — shortest distance to v
          pe[v]   — parent edge id to reach v (or -1)
        """
        weight = np.asarray(weight, dtype=np.float64)

        dist = np.full(self.n, np.inf, dtype=np.float64)
        pe = np.full(self.n, -1, dtype=np.int32)

        s = int(source)
        dist[s] = 0.0
        pq = [(0.0, s)]

        while pq:
            du, u = heapq.heappop(pq)
            if du != dist[u]:
                continue
            
            # цикл по исходящим из вершины рёбрам
            for i in range(self.first_out[u], self.first_out[u + 1]):
                e = int(self.out_eid[i])
                v = int(self.head[e])
                nd = du + weight[e]
                # Если значение изменилось, то вершина не помечена и возможно изменился минимум =>
                # она возможный кандидат на минимум
                if nd < dist[v]:
                    dist[v] = nd
                    pe[v] = e
                    heapq.heappush(pq, (nd, v))

        return dist, pe
    
    def to_networkx(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))

        for u in range(self.n):
            for eid in self.out_eid[self.first_out[u]: self.first_out[u + 1]]:
                v = int(self.head[eid])
                G.add_edge(u, v, eid=int(eid))

        return G


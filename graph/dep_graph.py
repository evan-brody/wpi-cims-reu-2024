# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import numpy as np
from itertools import chain, compress, product
from PyQt5.QtWidgets import QGraphicsRectItem

class DepGraph:
    MAX_VERTICES = 512
    DEFAULT_EDGE_WEIGHT = 1
    DEFAULT_DR = 0.25
    J = np.ones((MAX_VERTICES, MAX_VERTICES), np.uint8)
    I = np.identity(MAX_VERTICES, np.uint8)

    def __init__(self) -> None:
        self.refi = {} # Maps QGraphicsRectItems to indices
        self.iref = np.empty((self.MAX_VERTICES,), QGraphicsRectItem) # Maps indices to QGraphicsRectItems

        # How many vertices we have
        self.n = 0
        # Direct risk vector
        self.r0 = np.empty((self.MAX_VERTICES,), np.double)
        # Full risk vector
        self.r = np.empty((self.MAX_VERTICES,), np.double)
        # self.is_AND[i] stores whether vi is an AND gate
        self.is_AND = np.empty((self.MAX_VERTICES,), bool)
        # Adjacency matrix
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        # Transitive closure of A
        self.A_tc = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        # [i, j] = Count of paths j -> i with weight = 1
        # Probably doesn't need to be 64-bit but that can be figured out later
        self.one_count = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.uint64)

    # P(a U b)
    def scl_or_scl(self, a: float, b: float) -> float:
        return 1 - (1 - a) * (1 - b)
    
    # a is the probability of OR{b, ...}
    # b is the event to remove
    def or_inv(self, a: float, b: float) -> float:
        return (a - b) / (1 - b)
    
    def vec_or_vec(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        n = self.n
        return self.J[0, :n] - np.multiply(self.J[0, :n] - v1, self.J[0, :n] - v2)

    def mat_or_vec(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        lenv = len(v)
        return self.J[0, :lenv] - np.prod(self.J[:lenv, :lenv] - np.multiply(a, v), axis=1)

    def mat_or_mat(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res = np.empty((a.shape[0], b.shape[1]), np.double)
        
        for i, j in product(range(a.shape[0]), range(b.shape[1])):
            res[i, j] = 1 - np.prod(self.J[0, :a.shape[1]] - np.multiply(a[i], b.T[j]))
        
        return res
    
    def calc_Ac_full(self) -> np.ndarray:
        n = self.n
        Ac_full = np.empty((n, n), np.double)
        for i, j in product(range(n), repeat=2):
            Ac_full[i, j] = max(self.A_tc[i, j], int(bool(self.one_count[i, j])))
        
        return Ac_full
    
    def add_vertices(self, refs: list[QGraphicsRectItem], direct_risks: list[float]=None) -> None:
        n = self.n
        d = len(refs)

        for i, ref in enumerate(refs):
            self.refi[ref] = n + i
            self.iref[n + i] = ref

        if direct_risks:
            self.r0[n:n + d] = direct_risks
        else:
            self.r0[n:n + d] = self.DEFAULT_DR

        self.is_AND[n:n + d] = False

        self.A[n:n + d, :n + d] = 0
        self.A[:n, n:n + d] = 0

        self.A_tc[n:n + d, :n + d] = 0
        self.A_tc[:n, n:n + d] = 0

        self.one_count[n:n + d, :n + d] = 0
        self.one_count[:n, n:n + d] = 0

        self.n += d

    def add_vertex(self, ref: QGraphicsRectItem, direct_risk: float=DEFAULT_DR) -> None:
        n = self.n
        self.refi[ref] = n
        self.iref[n] = ref

        self.r0[n] = direct_risk
        self.is_AND[n] = False

        self.A[n, :n + 1] = 0
        self.A[:n, n] = 0

        self.A_tc[n, :n + 1] = 0
        self.A_tc[:n, n] = 0

        self.one_count[n, :n + 1] = 0
        self.one_count[:n, n] = 0

        self.n += 1

    def add_AND_gate(self, ref: QGraphicsRectItem) -> None:
        n = self.n
        self.refi[ref] = n
        self.iref[n] = ref

        self.r0[n] = 0
        self.is_AND[n] = True

        self.A[n, :n + 1] = 0
        self.A[:n, n] = 0

        self.A_tc[n, :n + 1] = 0
        self.A_tc[:n, n] = 0

        self.n += 1

    # edge is a tuple (a, b) where a -> b
    def add_edge(self, edge: tuple[QGraphicsRectItem], weight: float=DEFAULT_EDGE_WEIGHT) -> None:
        n = self.n
        a, b = self.refi[edge[0]], self.refi[edge[1]]
        self.A[b, a] = weight

        # Add to A-collapse by combining with existing connections
        if 1 == weight:
            self.one_count[b, a] += 1
        else:
            self.A_tc[b, a] = self.scl_or_scl(
                self.A_tc[b, a], weight
            )

        Ac_full = self.calc_Ac_full()

        # Collapse paths starting at a and passing through b
        # If we're dealing with an AND gate as B, we should
        # only collapse paths leading to other AND gates
        # Skip b because we don't care about loops
        to_update_to = np.copy(self.is_AND[:n]) if self.is_AND[b] else [True] * n
        to_update_to[b] = False
        for i in compress(range(n), to_update_to):
            # When a != AND & b = AND don't incorporate
            # (b -> i) in (a -> b -> i)_c
            new_path = weight
            if not self.is_AND[b] or self.is_AND[a]:
                new_path *= Ac_full[i, b]

            if 1 == new_path:
                self.one_count[i, a] += 1
            else:
                # a -> i OR (a -> b AND b -> i)
                self.A_tc[i, a] = self.scl_or_scl(
                    self.A_tc[i, a], new_path
                )

        # Make sure a doesn't loop on itself
        self.A_tc[a, a] = 0
        self.one_count[a, a] = 0
        
        # Collapse other paths that pass through a to b
        # Skip a's and b's columns. A's because we already
        # calculated its values, b's because we don't care
        # about loops
        # If we're dealing with an AND gate as a, we only want to
        # collapse paths of the form (i -> a -> AND)
        to_update_to = np.copy(self.is_AND[:n]) if self.is_AND[a] else [True] * n
        to_update_to[a] = False
        to_update_from = [True] * n
        to_update_from[a] = False
        to_update_from[b] = False
        for j in compress(range(n), to_update_from):
            for i in compress(range(n), to_update_to):
                # j -> i OR (j -> a AND a -> i)
                new_path = Ac_full[a, j]
                if not self.is_AND[a] or self.is_AND[j]:
                    new_path *= Ac_full[i, a]
                
                if 1 == new_path:
                    self.one_count[i, j] += 1
                else:
                    self.A_tc[i, j] = self.scl_or_scl(self.A_tc[i, j], new_path)

        # Remove any loops we've created
        np.fill_diagonal(self.A_tc[:n, :n], 0)
        np.fill_diagonal(self.one_count[:n, :n], 0)
    
    def add_edges(self, edges: list[tuple[QGraphicsRectItem]], weights: list[float]=None) -> None:
        if None == weights:
            for e in edges:
                self.add_edge(e)
        else:
            for e, w in zip(edges, weights):
                self.add_edge(e, w)

    # edge is a tuple of integers (a, b) where (a -> b)
    def update_edge_i(self, edge: tuple[int], new_weight: float) -> None:
        n = self.n
        a, b = edge

        # We need to add the identity matrix so our calculations
        # for broken_path_weight are accurate when i or j = a or b
        Ac_full = self.I[:n, :n] + self.calc_Ac_full()
        old_weight = Ac_full[b, a]
        if old_weight == new_weight:
            return
        self.A[b, a] = new_weight

        # Handle the edge itself directly if a is an AND gate
        # In all other cases, this is handled in the for-loop below
        if self.is_AND[a]:
            if 1 == old_weight:
                self.one_count[b, a] -= 1
            else:
                self.A_tc[b, a] = self.or_inv(
                    self.A_tc[b, a], old_weight
                )
            
            if 1 == new_weight:
                self.one_count[b, a] += 1
            else:
                self.A_tc[b, a] = self.scl_or_scl(
                    self.A_tc[b, a], new_weight
                )
        
        # Remove influence of deleted edge on other paths
        # then add influence of new edge weight
        # Skip diagonal because we don't allow those edges
        # If the edge involves an AND gate, we should only update
        # connections through it to other AND gates
        to_update_to = self.is_AND[:n] if self.is_AND[a] or self.is_AND[b] else [True] * n
        rangen = range(n)
        for i, j in filter(lambda t: t[0] != t[1], product(rangen, compress(rangen, to_update_to))):
            # (i -> a) AND (a -> b) AND (b -> j)
            # Note that (a -> b) is not all possible paths (a -> b),
            # but the specific edge we're updating
            broken_path_weight = Ac_full[a, i] * old_weight * Ac_full[j, b]
            # Remove influence of (a -> b) on (i -> j)
            if 1 == broken_path_weight:
                self.one_count[j, i] -= 1
            else:
                self.A_tc[j, i] = self.or_inv(
                    self.A_tc[j, i], broken_path_weight
                )

            # Add influence of new weight
            if 1 == new_weight:
                self.one_count[j, i] += 1
            else:
                self.A_tc[j, i] = self.scl_or_scl(
                    self.A_tc[j, i], new_weight
                )

    # edge is a tuple of references (a, b) where (a -> b)
    def update_edge(self, edge: tuple[QGraphicsRectItem], new_weight: float) -> None:
        self.update_edge_i((self.refi[edge[0]], self.refi[edge[1]]), new_weight)

    def update_edges(self, edges: list[tuple[QGraphicsRectItem]], new_weights: list[float]) -> None:
        for e, w in zip(edges, new_weights):
            self.update_edge(e, w)

    def update_vertex(self, ref: QGraphicsRectItem, new_weight: float) -> None:
        self.r0[self.refi[ref]] = new_weight

    def update_vertices(self, refs: list[QGraphicsRectItem], new_weights: list[float]) -> None:
        for ref, nw in zip(refs, new_weights):
            self.update_vertex(ref, nw)

    # edge is a tuple of integers (a, b) where (a -> b)
    def delete_edge_i(self, edge: tuple[int]) -> None:
        self.update_edge_i(edge, 0)
            
    # edge is a tuple of references (a, b) where (a -> b)
    def delete_edge(self, edge: tuple[QGraphicsRectItem]) -> None:
        self.delete_edge_i((self.refi[edge[0]], self.refi[edge[1]]))

    def delete_edges(self, edges: list[tuple[QGraphicsRectItem]]) -> None:
        for e in edges:
            self.delete_edge(e)

    # This works for AND gates too
    def delete_vertex(self, ref: QGraphicsRectItem) -> None:
        n = self.n
        vi = self.refi[ref]

        # Delete edges before we lose their information
        for i, j in chain(product((vi,), range(n)), product(range(n), (vi,))):
            if self.A[i, j]:
                self.delete_edge_i((j, i))

        del self.refi[ref]
        for key in self.refi.keys():
            index = self.refi[key]
            if index > vi:
                self.refi[key] = index - 1

        self.iref[vi:n - 1] = self.iref[vi + 1:n]
        self.r0[vi:n - 1] = self.r0[vi + 1:n]
        self.is_AND[vi:n - 1] = self.is_AND[vi + 1:n]

        self.A[vi:n - 1, :n] = self.A[vi + 1:n, :n]
        self.A[:n - 1, vi:n - 1] = self.A[:n - 1, vi + 1:n]

        self.A_tc[vi:n - 1, :n] = self.A_tc[vi + 1:n, :n]
        self.A_tc[:n - 1, vi:n - 1] = self.A_tc[:n - 1, vi + 1:n]

        self.one_count[vi:n - 1, :n] = self.one_count[vi + 1:n, :n]
        self.one_count[:n - 1, vi:n - 1] = self.one_count[:n - 1, vi + 1:n]

        self.n -= 1

    def delete_vertices(self, refs: list[QGraphicsRectItem]) -> None:
        for ref in refs:
            self.delete_vertex(ref)

    def update_AND_weights(self) -> None:
        n = self.n
        Ac_full = self.calc_Ac_full()

        AND_indices = compress(range(n), self.is_AND[:n])
        comp_bools = np.logical_not(self.is_AND[:n])
        comp_indices = compress(range(n), comp_bools)
        for i in AND_indices:
            # If an AND gate isn't connected to any components,
            # we calculate its risk separately and mark it as 0
            # for now
            if not np.any(Ac_full[i, comp_bools]):
                self.r0[i] = 0
                continue

            self.r0[i] = 1
            for j in comp_indices:
                # (j -> i)
                path_weight = Ac_full[i, j]
                # Include vertex weight if j is a component
                if not self.is_AND[j]:
                    path_weight *= self.r0[j]
                if path_weight:
                    self.r0[i] *= path_weight

        for i, j in filter(lambda t: t[0] != t[1], product(AND_indices, repeat=2)):
            # Only consider risk from AND gates that are
            # connected to a component. AND gates that
            # have no connected components will have an r0
            # value of 0
            path_weight = self.r0[j] * Ac_full[i, j]
            if path_weight:
                self.r0[i] *= path_weight

    # Note: self.r values for AND gates are garbage values
    def calc_r(self) -> None:
        n = self.n
        self.update_AND_weights()
        self.r = self.mat_or_vec(self.I[:n, :n] + self.calc_Ac_full(), self.r0[:n])
        return self.r
    
    def get_edge_weight_A(self, edge: tuple[QGraphicsRectItem]) -> float:
        return self.A_tc[self.refi[edge[1]], self.refi[edge[0]]]

    def get_edge_weight_Ac(self, edge: tuple[QGraphicsRectItem]) -> float:
        return self.A_tc[self.refi[edge[1]], self.refi[edge[0]]]

    def get_vertex_weight(self, ref: QGraphicsRectItem) -> float:
        return self.r0[self.refi[ref]]
    
    def get_total_risk(self, ref: QGraphicsRectItem) -> float:
        return self.r[self.refi[ref]]
    
    def get_r_dict(self) -> dict:
        n = self.n
        self.calc_r()
        return { self.iref[i] : risk for i, risk in compress(enumerate(self.r), np.logical_not(self.is_AND[:n])) }

if __name__ == "__main__":
    ########### Testing code ################
    # Test 1
    def test_suite_1():
        dg = DepGraph()

        dg.add_vertices(['s', 'c', 'v', 'p'], [0.25] * 4)
        dg.add_edges([('s', 'v'), ('c', 'v'), ('v', 'p')], [1 / 3] * 3)

        print(dg.calc_r())

        # Test 2
        dg = DepGraph()
        dg.add_vertices(['a', 'b', 'c', 'd'], [0.25] * 4)
        dg.add_edge(('b', 'd'))
        dg.add_edge(('c', 'd'))
        dg.add_edge(('a', 'b'))
        dg.add_edge(('a', 'c'))
        n = dg.n

        print(dg.A[:n, :n])
        print(dg.A_tc[:n, :n])
        print(dg.member_paths[3, 0])

        dg.delete_edge(('b', 'd'))
        dg.delete_edge(('c', 'd'))

        print()
        print(dg.A[:n, :n])
        print(dg.A_tc[:n, :n])
        print(dg.member_paths[3, 0])
        print(dg.calc_r())

        # Test 3
        dg = DepGraph()
        dg.add_vertices(['a', 'b', 'c', 'd'], [0.25] * 4)
        dg.add_edge(('b', 'd'))
        dg.add_edge(('c', 'd'))
        dg.add_edge(('a', 'b'))
        dg.add_edge(('a', 'c'))
        n = dg.n

        print(dg.A[:n, :n])
        print(dg.A_tc[:n, :n])
        print(dg.member_paths[3, 0])
        print("\nCALC_R")
        print(dg.calc_r())

        dg.delete_vertex('a')
        n = dg.n

        print()
        print(dg.A[:n, :n])
        print(dg.A_tc[:n, :n])
        print(dg.member_paths[3, 0])
        print("\nCALC_R")
        print(dg.calc_r())

        print("TEST 4")

        dg = DepGraph()
        dg.add_vertices(['a', 'b', 'c', 'd'], [0.5] * 4)
        dg.add_edge(('c', 'd'), 1)
        dg.add_edge(('b', 'c'), 1)
        dg.add_edge(('a', 'b'), 1)

        n = dg.n
        print("\nInitial paths:")
        print(dg.A_tc[:n, :n])
        print(dg.one_count[:n, :n])

        print("\nInitial r:")
        print(dg.calc_r())

        dg.delete_edge(('b', 'c'))

        print("\nFinal paths:")
        print(dg.A_tc[:n, :n])
        print(dg.one_count[:n, :n])

        print("\nFinal r:")
        print(dg.calc_r())

        a = np.array([[1, 2],
                        [3, 4]])
        b = np.array([[5, 6],
                        [7, 8]])
        c = np.vectorize(lambda i, j: max(i, j))

        print(c(a, b))
        print()

    def test_suite_2():
        print("TEST 1")
        dg = DepGraph()
        dg.add_vertices(['a', 'b', 'c'], [0.5] * 3)
        dg.add_AND_gate("AND")
        dg.add_edge(('AND', 'c'), 1)
        dg.add_edge(('a', 'AND'), 1)
        dg.add_edge(('b', 'AND'), 1)
        n = dg.n

        print(dg.calc_Ac_full()[:n, :n] + dg.I[:n, :n])

        print("AND calc_r:")
        print(dg.calc_r())

        print("\nTEST2")
        dg = DepGraph()
        dg.add_vertices(['a', 'b', 'c', 'd'], [0.75] * 4)
        dg.add_AND_gate('A')
        dg.add_AND_gate('B')
        dg.add_edge(('B', 'd'), 0.75)
        dg.add_edge(('A', 'B'), 0.75)
        dg.add_edge(('a', 'A'), 1)
        dg.add_edge(('b', 'A'), 1)
        dg.add_edge(('c', 'B'), 1)
        n = dg.n

        print()
        print(dg.calc_Ac_full()[:n, :n])
        print(dg.calc_r())

    test_suite_2()
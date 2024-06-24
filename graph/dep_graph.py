# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import numpy as np
from functools import reduce
from itertools import chain, product
from PyQt5.QtWidgets import QGraphicsRectItem

# TODO: figure out if we still need member_paths

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
        # Adjacency matrix
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        # Path information. Updated when calc_r is called
        self.A_collapse = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        # [i, j] = Count of paths j -> i with weight = 1
        # Probably doesn't need to be 64-bit but that can be figured out later
        self.one_count = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.uint64)
        # For index [i, j] possible paths from j -> i with weights
        self.member_paths = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), dict)

    # For two sets of paths p_a and p_b, returns possible connections
    # {pa0 + pb0, pa0 + pb1, ..., pa1 + pb0, pa1 + pb1, ..., ...}
    def connect_paths(self, p_a: dict, p_b: dict) -> dict:
        return { p1 + p2[1:] : p_a[p1] * p_b[p2] for p1, p2 in product(p_a.keys(), p_b.keys()) }
    
    # For a set of path p, returns OR{p_0, p_1, ...}
    def combine_paths(self, pathset: dict) -> float:
        return 1 - reduce(lambda a,b: (1 - a) * (1 - b), pathset.values(), 1)

    # Returns if a is a subtuple of b
    def subtuple_match(self, a: tuple, b: tuple) -> bool:
        # This method could be a one-liner, but it's
        # significantly slower that way because of Python
        lena = len(a)
        for i in range(len(b) - lena + 1):
            if a == b[i:lena + i]:
                return True
        return False

    # P(a U b)
    def scl_or_scl(self, a: float, b: float) -> float:
        return 1 - (1 - a) * (1 - b)
    
    # a is the probability of OR{b, ...}
    # b is the event to remove
    def or_inv(self, a: float, b: float) -> float:
        if b == 1: raise Exception("or_inv passed b = 1")
        return (a - b) / (1 - b)
    
    # TODO: make these methods modify the arguments in-place ?
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
    
    def calc_A_c_full(self) -> np.ndarray:
        n = self.n
        A_c_full = np.ndarray((n, n), np.double)
        for i, j in product(range(n), repeat=2):
            A_c_full[i, j] = max(self.A_collapse[i, j], int(bool(self.one_count[i, j])))
        
        return A_c_full
    
    def add_vertices(self, refs: list, direct_risks: list=None) -> None:
        n = self.n
        d = len(refs)

        for i, ref in enumerate(refs):
            self.refi[ref] = n + i
            self.iref[n + i] = ref

        if direct_risks:
            self.r0[n:n + d] = direct_risks
        else:
            self.r0[n:n + d] = self.DEFAULT_DR

        self.A[n:n + d, :n + d] = 0
        self.A[:n, n:n + d] = 0

        self.A_collapse[n:n + d, :n + d] = 0
        self.A_collapse[:n, n:n + d] = 0

        self.one_count[n:n + d, :n + d] = 0
        self.one_count[:n, n:n + d] = 0

        # This needs to be a for-loop so that it's
        # not all the same dictionary
        for i, j in product(range(n, n + d), range(n + d)):
            self.member_paths[i, j] = dict()
        for i, j in product(range(n), range(n, n + d)):
            self.member_paths[i, j] = dict()

        self.n += d

    def add_vertex(self, ref: QGraphicsRectItem, direct_risk: float=None) -> None:
        n = self.n
        direct_risk = direct_risk if direct_risk else self.DEFAULT_DR
        self.refi[ref] = n
        self.iref[n] = ref

        self.r0[n] = direct_risk

        self.A[n, :n + 1] = 0
        self.A[:n, n] = 0

        self.A_collapse[n:n + 1, :n + 1] = 0
        self.A_collapse[:n, n:n + 1] = 0

        # This needs to be a for-loop so that it's
        # not all the same dictionary
        for j in range(n + 1):
            self.member_paths[n, j] = dict()
        for i in range(n):
            self.member_paths[i, n] = dict()

        self.n += 1

    # edge is a tuple (a, b) where a -> b
    def add_edge(self, edge: tuple, weight: float=None) -> None:
        n = self.n
        a, b = self.refi[edge[0]], self.refi[edge[1]]
        weight = weight if weight else self.DEFAULT_EDGE_WEIGHT
        self.A[b, a] = weight
        self.member_paths[b, a][(a, b)] = weight

        # Add to A-collapse by combining with existing connections
        if 1 == weight:
            self.one_count[b, a] += 1
        else:
            self.A_collapse[b, a] = self.scl_or_scl(
                self.A_collapse[b, a], weight
            )

        A_c_full = self.calc_A_c_full()

        # Collapse paths starting at a and passing through b
        for i in range(n):
            new_path = A_c_full[i, b]
            if new_path:
                new_path *= weight
                if 1 == new_path:
                    self.one_count[i, a] += 1
                else:
                    # a -> i OR (a -> b AND b -> i)
                    self.A_collapse[i, a] = self.scl_or_scl(
                        self.A_collapse[i, a], new_path
                    )

                # P[a -> i] U P[a -> b -> i]
                self.member_paths[i, a].update(
                    self.connect_paths(self.member_paths[b, a], self.member_paths[i, b])
                )

        # Make sure a doesn't loop on itself
        self.A_collapse[a, a] = 0
        self.one_count[a, a] = 0
        
        # Collapse other paths that pass through a to b
        # Skip a's and b's columns. A's because we already
        # calculated its values, b's because we don't care
        # about loops
        lesser_i, greater_i = min(a, b), max(a, b)
        for j in chain(range(lesser_i), \
                       range(lesser_i + 1, greater_i), \
                       range(greater_i + 1, n)):
            for i in range(n):
                # j -> i OR (j -> a AND a -> i)
                new_path = A_c_full[a, j] * A_c_full[i, a]
                if not new_path:
                    continue
                
                if 1 == new_path:
                    self.one_count[i, j] += 1
                else:
                    self.A_collapse[i, j] = self.scl_or_scl(self.A_collapse[i, j], new_path)
                
                # P[j -> i] U P[j -> a -> i] 
                self.member_paths[i, j].update(
                    self.connect_paths(self.member_paths[a, j], self.member_paths[i, a])
                )

        # Remove any loops we've created
        np.fill_diagonal(self.A_collapse[:n, :n], 0)
        np.fill_diagonal(self.one_count[:n, :n], 0)

    def add_edges(self, edges: list, weights: list=None) -> None:
        if None == weights:
            for e in edges:
                self.add_edge(e)
        else:
            for e, w in zip(edges, weights):
                self.add_edge(e, w)

    # edge is a tuple of integers (a, b) where a -> b
    def delete_edge_i(self, edge: tuple[int]) -> None:
        n = self.n
        a, b = edge

        A_c_full = self.calc_A_c_full()
        weight = A_c_full[b, a]
        self.A[b, a] = 0

        to_delete = []
        for i, j in product(range(n), repeat=2):
            for key in self.member_paths[i, j].keys():
                if not self.subtuple_match(edge, key):
                    continue

                to_delete.append((i, j, key))

        for i, j, key in to_delete:
            del self.member_paths[i, j][key]

        # Remove influence of deleted edge on other paths
        # Skip i = a, j = b because we'll deal with that separately
        for i, j in product(chain(range(a), range(a + 1, n)), chain(range(b), range(b + 1, n))):
            # (i -> a) AND (a -> b) AND (b -> j)
            # Note that (a -> b) is not all possible paths (a -> b),
            # but the specific edge we're deleting
            broken_path_weight = A_c_full[a, i] * weight * A_c_full[j, b]
            # Remove influence of (a -> b) on (i -> j)
            if 1 == broken_path_weight:
                self.one_count[j, i] -= 1
            else:
                self.A_collapse[j, i] = self.or_inv(
                    self.A_collapse[j, i], broken_path_weight
                )
    
        # Dealing with paths (a -> j)
        for j in chain(range(a), range(a + 1, n)):
            broken_path_weight = weight * A_c_full[j, b]
            if 1 == broken_path_weight:
                self.one_count[j, a] -= 1
            else:
                self.A_collapse[j, a] = self.or_inv(
                    self.A_collapse[j, a], broken_path_weight
                )

        # Dealing with paths (i -> b)
        for i in chain(range(b), range(b + 1, n)):
            broken_path_weight = A_c_full[a, i] * weight
            if 1 == broken_path_weight:
                self.one_count[b, i] -= 1
            else:
                self.A_collapse[b, i] = self.or_inv(
                    self.A_collapse[b, i], broken_path_weight
                )

        # Settling (a -> b)
        if 1 == weight:
            self.one_count[b, a] -= 1
        else:
            self.A_collapse[b, a] = self.or_inv(
                self.A_collapse[b, a], broken_path_weight
            )
            
    # edge is a tuple of references (a, b) where (a -> b)
    def delete_edge(self, edge: tuple[QGraphicsRectItem]) -> None:
        self.delete_edge_i((self.refi[edge[0]], self.refi[edge[1]]))

    def delete_edges(self, edges: list) -> None:
        for e in edges:
            self.delete_edge(e)

    def delete_vertex(self, ref: QGraphicsRectItem) -> None:
        n = self.n
        vi = self.refi[ref]
        del self.refi[ref]

        self.iref[vi:n - 1] = self.iref[vi + 1:n]
        self.r0[vi:n - 1] = self.r0[vi + 1:n]

        # Delete edges before we lose their information
        for i, j in chain(product((vi,), range(n)), product(range(n), (vi,))):
            if self.A[i, j]:
                self.delete_edge_i((j, i))

        self.A[vi:n - 1, :n] = self.A[vi + 1:n, :n]
        self.A[:n - 1, vi:n - 1] = self.A[:n - 1, vi + 1:n]

        self.A_collapse[vi:n - 1, :n] = self.A_collapse[vi + 1:n, :n]
        self.A_collapse[:n - 1, vi:n - 1] = self.A_collapse[:n - 1, vi + 1:n]

        self.one_count[vi:n - 1, :n] = self.one_count[vi + 1:n, :n]
        self.one_count[:n - 1, vi:n - 1] = self.one_count[:n - 1, vi + 1:n]

        # Delete the dictionaries before copying back
        self.member_paths[vi, :n] = None
        self.member_paths[:n, vi] = None
        self.member_paths[vi:n - 1, :n] = self.member_paths[vi + 1:n, :n]
        self.member_paths[:n - 1, vi:n - 1] = self.member_paths[:n - 1, vi + 1:n]

        self.n -= 1

    def delete_vertices(self, refs: list) -> None:
        for ref in refs:
            self.delete_vertex(ref)
    
    def calc_r(self) -> np.ndarray:
        n = self.n
        A_c_full = self.calc_A_c_full()

        self.r = self.mat_or_vec(self.I[:n, :n] + A_c_full, self.r0[:n])
        return self.r
    
    def get_r_dict(self) -> dict:
        return { self.iref[i] : risk for i, risk in enumerate(self.r) }

if __name__ == "__main__":
    ########### Testing code ################
    # Test 1
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
    print(dg.A_collapse[:n, :n])
    print(dg.member_paths[3, 0])

    dg.delete_edge(('b', 'd'))
    dg.delete_edge(('c', 'd'))

    print()
    print(dg.A[:n, :n])
    print(dg.A_collapse[:n, :n])
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
    print(dg.A_collapse[:n, :n])
    print(dg.member_paths[3, 0])

    dg.delete_vertex('a')
    n = dg.n

    print()
    print(dg.A[:n, :n])
    print(dg.A_collapse[:n, :n])
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
    print(dg.A_collapse[:n, :n])
    print(dg.one_count[:n, :n])

    print("\nInitial r:")
    print(dg.calc_r())

    dg.delete_edge(('b', 'c'))

    print("\nFinal paths:")
    print(dg.A_collapse[:n, :n])
    print(dg.one_count[:n, :n])

    print("\nFinal r:")
    print(dg.calc_r())
# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import numpy as np
from functools import reduce
from itertools import repeat, chain, product
from PyQt5.QtWidgets import QGraphicsRectItem

import timeit

# TODO: edge removal (necessary for vertex removal)
# TODO: vertex removal
class DepGraph_CPUOptimized:
    MAX_VERTICES = 512
    DEFAULT_EDGE_WEIGHT = 1
    DEFAULT_DR = 0.05
    J = np.ones((MAX_VERTICES, MAX_VERTICES), np.uint8)
    I = np.identity(MAX_VERTICES, np.uint8)

    def __init__(self) -> None:
        self.refi = {} # Maps QGraphicsRectItems to indices
        self.iref = np.empty((self.MAX_VERTICES,), QGraphicsRectItem) # Maps indices to QGraphicsRectItems

        self.n = 0 # How many vertices we have
        self.m = 1 # Largest power of A we care about
        self.r0 = np.empty((self.MAX_VERTICES,), np.double) # Direct risk vector
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        self.A_collapse = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        self.member_paths = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), set)

    def connect_paths(self, p_a: set, p_b: set) -> set:
        return { p[0] + p[1][1:] for p in product(p_a, p_b) }

    def scl_or_scl(self, a: float, b: float) -> float:
        return 1 - (1 - a) * (1 - b)
    
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
    
    def add_vertices(self, refs, direct_risks=None) -> None:
        m = self.m
        n = self.n
        d = len(refs)

        for i, ref in enumerate(refs):
            self.refi[ref] = n + i
            self.iref[n + i] = ref

        if direct_risks is None:
            self.r0[n:n + d] = self.DEFAULT_DR
        else:
            self.r0[n:n + d] = direct_risks

        self.A[:n:n + d, :n + d] = 0
        self.A[:n, n:n + d] = 0

        self.A_collapse[n:n + d, :n + d] = 0
        self.A_collapse[:n, n:n + d] = 0

        # This needs to be a for-loop so that it's
        # not all the same set
        for i, j in product(range(n, n + d), range(n + d)):
            self.member_paths[i, j] = set()
        for i, j in product(range(n), range(n, n + d)):
            self.member_paths[i, j] = set()

        self.n += d

    def delete_vertices(self, refs) -> None:
        pass

    def add_edges(self, edges, weights=None) -> None:
        weights = weights if weights else [None] * len(edges)
        for e, w in zip(edges, weights):
            self.add_edge(e, w)
    
    # edge is a tuple (a, b) where a -> b
    def add_edge(self, edge, weight=None) -> None:
        n = self.n
        a, b = self.refi[edge[0]], self.refi[edge[1]]
        weight = weight if weight else self.DEFAULT_EDGE_WEIGHT
        self.A[b, a] = weight
        self.member_paths[b, a].add((a, b))

        # Add to A-collapse by combining with existing connections
        self.A_collapse[b, a] = self.scl_or_scl(
            self.A_collapse[b, a], weight
        )

        # Collapse paths starting at a and passing through b
        # self.A_collapse[:n, a] = self.vec_or_vec(
        #     self.A_collapse[:n, a], weight * self.A_collapse[:n, b]
        # )

        for i in range(n):
            new_path = self.A_collapse[i, b]
            if new_path:
                new_path *= weight
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
                new_path = self.A_collapse[a, j] * self.A_collapse[i, a]
                if new_path:
                    self.A_collapse[i, j] = self.scl_or_scl(self.A_collapse[i, j], new_path)
                    # P[j -> i] U P[j -> a -> i] 
                    self.member_paths[i, j].update(
                        self.connect_paths(self.member_paths[a, j], self.member_paths[i, a])
                    )

            # self.A_collapse[:n, j] = self.vec_or_vec(
            #     self.A_collapse[:n, j], self.A_collapse[a, j] * self.A_collapse[:n, a]
            # )

        # Remove any loops we've created
        np.fill_diagonal(self.A_collapse, 0)

        # TODO: store participating edges / vertices
        # TODO: optimize loop removal ?

    def delete_edges(self, edges) -> None:
        pass
    
    def calc_r(self) -> np.ndarray:
        n = self.n
        return self.mat_or_vec(self.I[:n, :n] + self.A_collapse[:n, :n], self.r0[:n])

if __name__ == "__main__":
    # Testing code
    dg = DepGraph_CPUOptimized()

    dg.add_vertices(['s', 'c', 'v', 'p'], [0.25, 0.25, 0.25, 0.25])
    dg.add_edges([('s', 'v'), ('c', 'v'), ('v', 'p')], [1/3, 1/3, 1/3])

    # print(dg.calc_r())
    n = dg.n

    # print(dg.A_collapse[:n, :n])
    # print(dg.member_paths[:n, :n])

    p_a = set([(1, 2, 3), (1, 4, 3)])
    p_b = set([(5, 6, 7), (5, 8, 7)])
    dg.connect_paths(p_a, p_b)

    # print(dg.member_paths[0, 3])
    dg = DepGraph_CPUOptimized()
    dg.add_vertices(['a', 'b', 'c', 'd'], [0.25] * 4)
    dg.add_edge(('b', 'd'), None)
    dg.add_edge(('c', 'd'), None)
    dg.add_edge(('a', 'b'), None)
    dg.add_edge(('a', 'c'), None)
    n = dg.n

    print(dg.A_collapse[:n, :n])
    print(dg.member_paths[3, 0])


####### DON'T USE ###############
class DepGraph_RAMOptimized:
    MAX_VERTICES = 512
    DEFAULT_EDGE_WEIGHT = 1
    DEFAULT_DR = 0.05

    def __init__(self) -> None:
        self.refi = {} # Maps QGraphicsRectItems to indices
        self.iref = np.empty((self.MAX_VERTICES,), QGraphicsRectItem) # Maps indices to QGraphicsRectItems

        self.r0 = [] # Direct risk vector
        self.A = [[]] # Adjacency matrix
    
    def scl_or_scl(self, a: float, b: float) -> float:
        return 1 - (1 - a) * (1 - b)
    
    def vec_or_vec(self, v1: list, v2: list) -> list:
        n = len(self.A[0])
        return [1] * n - np.multiply([1] * n - v1, [1] * n - v2)

    def mat_or_vec(self, a: list[list], v: list) -> list:
        lenv = len(v)
        return [1] * lenv - np.prod([ [1] * lenv for _ in repeat(None, lenv) ] - np.multiply(a, v), axis=1)

    def mat_or_mat(self, a: list[list], b: list[list]) -> list[list]:
        res = [[0] * len(b[0])]

        for _ in repeat(None, len(a) - 1):
            res.append([0] * len(b[0]))
        
        for i, j in product(range(len(a)), range(len(b[0]))):
            res[i][j] = 1 - np.prod([1] * len(a[0]) - np.multiply(a[i], np.transpose(b)[j]))
        
        return res

    def mat_or_mat_c(self, a: list[list], b: list[list]) -> list:
        res = [[0] * len(b[0])]

        for _ in repeat(None, len(a) - 1):
            res.append([0] * len(b[0]))
        
        for i, j in product(range(len(a)), range(len(b[0]))):
            res[i][j] = np.prod([1] * len(a[0]) - np.multiply(a[i], np.transpose(b)[j]))
        
        return res

    def exp_A(self, p) -> list[list]:
        n = len(self.A[0])

        res = np.identity(n)
        for _ in repeat(None, p):
            res = self.mat_or_mat(self.A, res)

        return res
    
    def exp_A_c(self, p) -> list[list]:
        n = len(self.A[0])
        res = np.identity(n)
        if 0 == p:
            return res

        for _ in repeat(None, p - 1):
            res = self.mat_or_mat(self.A, res)

        return self.mat_or_mat_c(self.A, res)
    
    def add_vertices(self, refs, direct_risks=None) -> None:
        n = len(self.A[0])
        d = len(refs)

        for i, ref in enumerate(refs):
            self.refi[ref] = n + i
            self.iref[n + i] = ref

        if direct_risks is None:
            self.r0 += [self.DEFAULT_DR] * d
        else:
            self.r0 += direct_risks
        
        if 0 == n:
            self.A.pop()

        for _ in repeat(None, d):
            self.A.append([0] * (n + d))
        
        for i in range(n):
            self.A[i] += [0] * d

    # edges is a list of tuples (a, b) where a -> b
    # with the weight in weights whose index matches the tuple's
    def add_edges(self, edges, weights=None) -> None:
        if weights:
            for k, pair in enumerate(edges):
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i][j] = weights[k]
        else:
            for pair in edges:
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i][j] = self.DEFAULT_EDGE_WEIGHT

    def calc_r(self, m) -> list:
        n = len(self.A[0])
        return self.mat_or_vec(np.identity(n, np.uint8) + np.ones((n, n), np.uint8) - reduce(np.multiply, [ self.exp_A_c(i) for i in range(1, m + 1) ]), self.r0)
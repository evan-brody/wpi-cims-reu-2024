# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import numpy as np
from functools import reduce
from itertools import repeat, chain, product
from PyQt5.QtWidgets import QGraphicsRectItem

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
    
    def vec_or_vec(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        n = len(self.A[0])
        return [1] * n - np.multiply([1] * n - v1, [1] * n - v2)

    def mat_or_vec(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        lenv = len(v)
        return [1] * lenv - np.prod([ [1] * lenv for _ in repeat(None, lenv) ] - np.multiply(a, v), axis=1)

    def mat_or_mat(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res = [[0] * len(b[0])]

        for _ in repeat(None, len(a) - 1):
            res.append([0] * len(b[0]))
        
        for i, j in product(range(len(a)), range(len(b[0]))):
            res[i][j] = 1 - np.prod([1] * len(a[0]) - np.multiply(a[i], np.transpose(b)[j]))
        
        return res

    def mat_or_mat_c(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res = [[0] * len(b[0])]

        for _ in repeat(None, len(a) - 1):
            res.append([0] * len(b[0]))
        
        for i, j in product(range(len(a)), range(len(b[0]))):
            res[i][j] = np.prod([1] * len(a[0]) - np.multiply(a[i], np.transpose(b)[j]))
        
        return res

    def exp_A(self, p) -> np.ndarray:
        n = len(self.A[0])

        res = np.identity(n)
        for _ in repeat(None, p):
            res = self.mat_or_mat(self.A, res)

        return res
    
    def exp_A_c(self, p) -> np.ndarray:
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

    def calc_r(self, m) -> np.ndarray:
        n = len(self.A[0])
        return self.mat_or_vec(np.identity(n, np.uint8) + np.ones((n, n), np.uint8) - reduce(np.multiply, [ self.exp_A_c(i) for i in range(1, m + 1) ]), self.r0)

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
        self.r0 = np.empty((self.MAX_VERTICES,), np.double) # Direct risk vector
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        self.A_collapse = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)

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
        n = self.n
        d = len(refs)

        for i, ref in enumerate(refs):
            self.refi[ref] = n + i
            self.iref[n + i] = ref

        if direct_risks is None:
            self.r0[n:n + d] = self.DEFAULT_DR
        else:
            self.r0[n:n + d] = direct_risks
        
        self.A[n:n + d, :n + d] = 0
        self.A[:n, n:n + d] = 0
        self.A_collapse[n:n + d, :n + d] = 0
        self.A_collapse[:n, n:n + d] = 0

        self.n += d

    # edges is a list of tuples (a, b) where a -> b
    # with the weight in weights whose index matches the tuple's
    def add_edges(self, edges, weights=None) -> None:
        n = self.n
        if weights:
            for k, pair in enumerate(edges):
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i, j] = weights[k]
        else:
            for pair in edges:
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i, j] = self.DEFAULT_EDGE_WEIGHT
        
        # Auto-update / caching
        for k, pair in enumerate(edges):
            weightk = weights[k]
            starti, endi = self.refi[pair[0]], self.refi[pair[1]]

            # Collapse paths starting at a and passing through b
            self.A_collapse[endi, starti] = weightk
            self.A_collapse[:n, starti] = self.vec_or_vec(
                self.A_collapse[:n, starti], weightk * self.A_collapse[:n, endi]
            )

            # Collapse other paths that pass through a to b
            for j in chain(range(starti), range(starti + 1, n)):
                self.A_collapse[:n, j] = self.vec_or_vec(
                    self.A_collapse[:n, j], self.A_collapse[starti, j] * self.A_collapse[:n, starti]
                )
    
    def calc_r(self) -> np.ndarray:
        n = self.n
        return self.mat_or_vec(self.I[:n, :n] + self.A_collapse[:n, :n], self.r0[:n])

if __name__ == "__main__":
    # Testing code
    dg = DepGraph_CPUOptimized()

    dg.add_vertices(['s', 'c', 'v', 'p'], [0.25, 0.25, 0.25, 0.25])
    dg.add_edges([('s', 'v'), ('c', 'v'), ('v', 'p')], [1/3, 1/3, 1/3])

    print(dg.calc_r())
    
    dg = DepGraph_RAMOptimized()

    dg.add_vertices(['s', 'c', 'v', 'p'], [0.25, 0.25, 0.25, 0.25])
    dg.add_edges([('s', 'v'), ('c', 'v'), ('v', 'p')], [1/3, 1/3, 1/3])

    m = 2
    print(dg.calc_r(m))
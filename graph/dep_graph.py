# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import sys # TODO: this is for RAM usage analysis. remove when not needed anymore
import numpy as np
import itertools
import functools
from PyQt5.QtWidgets import QGraphicsRectItem

# TODO: remove vertices
# TODO: remove edges
# TODO: optimize

class DepGraph:
    MAX_VERTICES = 512
    DEFAULT_EDGE_WEIGHT = 1
    DEFAULT_DR = 0.05
    J = np.ones((MAX_VERTICES, MAX_VERTICES), np.uint8)
    I = np.identity(MAX_VERTICES, np.uint8)

    def __init__(self, cpu_optimized=False) -> None:
        # User will be able to set this. This will
        # use up (significantly) more RAM with the benefit
        # of (significantly) faster risk calculations
        self.cpu_optimized = cpu_optimized

        self.r0 = np.empty((self.MAX_VERTICES,), np.double) # Direct risk vector
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double) # Adjacency matrix

        self.m = 0 # Max length path
        self.n = 0 # How many vertices we have
        self.refi = {} # Maps QGraphicsRectItems to indices
        self.iref = np.empty((self.MAX_VERTICES,), QGraphicsRectItem) # Maps indices to QGraphicsRectItems

        if not self.cpu_optimized:
            return

        self.Ap = np.empty((self.MAX_VERTICES,), np.ndarray) # Caches powers of A
        self.Ap[0] = np.identity(self.MAX_VERTICES, np.uint8)
        self.A_collapse = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double)
        self.r = np.empty((self.MAX_VERTICES,), np.double)

    def set_cpu_optimized(self, cpu_optimized: bool) -> None:
        self.cpu_optimized = cpu_optimized

        if self.cpu_optimized:
            pass # TODO: Cache relevant information on enabling CPU optimization

    def get_cpu_optimized(self) -> bool:
        return self.cpu_optimized

    def prob_or(self, a: float, b: float) -> float:
        return 1 - (1 - a) * (1 - b)

    def prob_mat_vec_mul(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        lenv = len(v)
        return self.J[0, :lenv] - np.prod(self.J[:lenv, :lenv] - np.multiply(a, v), axis=1)

    def prob_mat_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res = np.empty((a.shape[0], b.shape[1]), np.double)
        
        for i, j in itertools.product(range(a.shape[0]), range(b.shape[1])):
            res[i, j] = 1 - np.prod(self.J[0, :a.shape[1]] - np.multiply(a[i], b.T[j]))
        
        return res
    
    def prob_mat_mul_c(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res = np.empty((a.shape[0], b.shape[1]), np.double)
        
        for i, j in itertools.product(range(a.shape[0]), range(b.shape[1])):
            res[i, j] = np.prod(self.J[0, :a.shape[1]] - np.multiply(a[i], b.T[j]))
        
        return res
    
    # This is slow
    # TODO: speed up
    def exp_A(self, p) -> np.ndarray:
        n = self.n

        res = np.identity(n)
        A1 = self.A[:n, :n]
        for _ in range(p):
            res = self.prob_mat_mul(A1, res)

        return res
    
    def exp_A_c(self, p) ->np.ndarray:
        n = self.n
        if 0 == p:
            return self.I[:n, :n]

        res = np.identity(n)
        A1 = self.A[:n, :n]
        for _ in range(p - 1):
            res = self.prob_mat_mul(A1, res)

        return self.prob_mat_mul_c(A1, res)
    
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

        if not self.cpu_optimized:
            self.n += d
            return
        
        m = self.m
        
        for i in range(m + 1):
            self.Ap[i][n:n + d, :n + d] = 0
            self.Ap[i][:n, n:n + d] = 0
        self.A_collapse[n:n + d, :n + d] = 1
        self.A_collapse[:n, n:n + d] = 1

        self.n += d

    # edges is a list of tuples (a, b) where a -> b
    # with the weight in weights whose index matches the tuple's
    def add_edges(self, edges, weights=None) -> None:
        if weights:
            for k, pair in enumerate(edges):
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i, j] = weights[k]
        else:
            for pair in edges:
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i, j] = self.DEFAULT_EDGE_WEIGHT

        if not self.cpu_optimized:
            return
        
        # Auto-update / caching
        n = self.n

        for k, pair in enumerate(edges):
            starti, endi = self.refi[pair[0]], self.refi[pair[1]]
            weightk_c = 1 - weights[k]
            self.A_collapse[endi, starti] = weightk_c
            self.A_collapse[:n, starti] = self.A_collapse[:n, endi] * weightk_c
            # TODO: propagate from vertices connecting to a

    # This is slow! Shouldn't be used unless necessary
    # TODO: this class should calculate m. could be tricky: it's NP-hard
    def calc_r(self, m) -> np.ndarray:
        n = self.n
        return self.prob_mat_vec_mul(self.I[:n, :n] + self.J[:n, :n] - functools.reduce(np.multiply, [ self.exp_A_c(i) for i in range(1, m + 1) ]), self.r0[:n])
    
    def calc_r_quick(self, m) -> np.ndarray:
        if not self.cpu_optimized:
            return
        
        n = self.n
        # TODO: take into account a_collapse using complement
        # return self.prob_mat_vec_mul(self.Ap[0][:n, :n] + self.A_collapse[:n, :n], self.r0[:n])

if __name__ == "__main__":
    # Testing code
    dg = DepGraph(True)

    dg.add_vertices(['s', 'c', 'v', 'p'], [0.25, 0.25, 0.25, 0.25])
    dg.add_edges([('s', 'v'), ('c', 'v'), ('v', 'p')], [1/3, 1/3, 1/3])

    m = 2

    print(dg.calc_r(m))
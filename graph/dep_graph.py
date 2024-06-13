# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import sys # TODO: this is for RAM usage analysis. remove when not needed anymore
import numpy as np
import itertools
import functools
from PyQt5.QtWidgets import QGraphicsRectItem

class DepGraph:
    MAX_VERTICES = 4096
    DEFAULT_EDGE_WEIGHT = 1
    DEFAULT_DR = 0.05
    # 17 MB
    J = np.ones((MAX_VERTICES, MAX_VERTICES), np.uint8)
    I = np.identity(MAX_VERTICES, np.uint8)

    def __init__(self) -> None:
        # User will be able to set this. I anticipate that
        # auto updates could be very slow, so there'll be
        # an option to manually recalculate whenever the user
        # wants as opposed to auto updates
        self.auto_update = False

        # 33 KB
        self.r0 = np.empty((self.MAX_VERTICES,), np.double) # Direct risk vector
        # 134 MB
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double) # Adjacency matrix

        self.n = 0 # How many vertices we have
        self.refi = {} # Maps QGraphicsRectItems to indices
        # 33 KB
        self.iref = np.empty((self.MAX_VERTICES,), QGraphicsRectItem) # Maps indices to QGraphicsRectItems

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
    

    def set_auto_update(self, bval) -> None:
        self.auto_update = bval

    def get_auto_update(self) -> bool:
        return self.auto_update
    
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

        self.n += d

    # edges is a list of tuples (a, b) where a -> b
    # with the weight in weights whose index matches the tuple's
    def add_edges(self, edges, weights=None) -> None:
        if weights is None:
            for pair in edges:
                i, j = self.refi[pair[1]], self.refi[pair[0]]
                self.A[i, j] = self.DEFAULT_EDGE_WEIGHT
            return
        
        for k, pair in enumerate(edges):
            i, j = self.refi[pair[1]], self.refi[pair[0]]
            self.A[i, j] = weights[k]

        if not self.auto_update:
            return
        
        # TODO: add auto update for r

    # This is slow! Shouldn't be used unless necessary
    # TODO: this class should calculate m. could be tricky: it's NP-hard
    def calc_r(self, m) -> np.ndarray:
        n = self.n
        return self.prob_mat_vec_mul(self.I[:n, :n] + self.J[:n, :n] - functools.reduce(np.multiply, [ self.exp_A_c(i) for i in range(1, m + 1) ]), self.r0[:n])

if __name__ == "__main__":
    # Testing code
    dg = DepGraph()

    dg.add_vertices(['s', 'c', 'v', 'p'], [0.25, 0.25, 0.25, 0.25])
    dg.add_edges([('s', 'v'), ('c', 'v'), ('v', 'p')], [1/3, 1/3, 1/3])

    m = 2

    print(dg.calc_r(m))
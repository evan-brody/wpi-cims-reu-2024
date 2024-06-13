# @file dep_graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import sys # TODO: this is for RAM usage analysis. remove when not needed anymore
import numpy as np
import itertools
from PyQt5.QtWidgets import QGraphicsRectItem

class DepGraph:
    MAX_VERTICES = 4096
    DEFAULT_EDGE_WEIGHT = 1
    DEFAULT_DR = 0.05
    # 17 MB
    J = np.ones((MAX_VERTICES, MAX_VERTICES), np.uint8)

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

    def prob_mat_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        assert a.shape[1] == b.shape[0]

        res = np.empty((a.shape[0], b.shape[1]), np.double)
        
        for i, j in itertools.product(range(a.shape[0]), range(b.shape[1])):
            res[i, j] = 1 - np.prod(self.J[0, :a.shape[1]] - np.multiply(a[i], b.T[j]))
        
        return res

    def prob_mat_vec_mul(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        assert a.shape[1] == v.shape[0]

        lenv = len(v)
        return self.J[0, :lenv] - np.prod(self.J[:lenv, :lenv] - np.multiply(a, v), axis=1)

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
            self.r0[n:n + d] = direct_risks
        else:
            self.r0[n:n + d] = self.DEFAULT_DR
        self.A[n:n + d, :n + d] = 0
        self.A[:n, n:n + d] = 0

        self.n += d

    # edges is a list of tuples (a, b) where a -> b
    # with the weight in weights whose index matches the tuple's
    def add_edges(self, edges, weights=None):
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
        
        # TODO

    def calc_r(self):
        pass

if __name__ == "__main__":
    # Testing code
    dg = DepGraph()

    dg.add_vertices(['a', 'b'], [1, 2])
    dg.add_vertices(['c'], [3])
    dg.add_vertices(['d', 'e', 'f'], [4, 5, 6])

    dg.add_edges([('a', 'c')])
    dg.add_edges([('f', 'e')], [0.5])

    # print(dg.A[:9, :9])
    # print(dg.r0[:9])

    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    b = np.array([1, 2, 3])

    dg.prob_mat_vec_mul(a, b)
    print(dg.prob_mat_mul(a, a))

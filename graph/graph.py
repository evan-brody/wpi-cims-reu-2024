# @file graph.py
# @author Evan Brody
# @brief Provides backend graph functionality for dependency analysis

import numpy as np

class DepGraph:
    MAX_VERTICES = 2 ** 12

    def __init__(self) -> None:
        self.auto_update = False
        self.r0 = np.empty((self.MAX_VERTICES,), np.double) # Direct risk vector
        self.A = np.empty((self.MAX_VERTICES, self.MAX_VERTICES), np.double) # Adjacency matrix
        self.n = 0

    def set_auto_update(self, bval) -> None:
        self.auto_update = bval

    def get_auto_update(self) -> bool:
        return self.auto_update
    
    def add_vertices(self, direct_risks) -> None:
        n = self.n
        d = len(direct_risks)

        self.r0[n:n + d] = 0
        self.A[n:n + d, :n + d] = 0
        self.A[:n, n:n + d] = 0

        self.n += d

if __name__ == "__main__":
    # Testing code
    dg = DepGraph()


    dg.add_vertices([1, 2])
    dg.add_vertices([2])
    dg.add_vertices([1, 2, 2])
    print(dg.A[:9, :9])
    print(dg.r0[:9])
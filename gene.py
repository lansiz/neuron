# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np


class Gene(object):
    def __init__(self, N, conn_ratio=.2):
        self.connections_number = np.min([(N ** 2 - N) // 2, int(N ** 2 * conn_ratio)])
        self.connections = np.array([0] * (N ** 2), dtype=np.float64).reshape((N, N))
        for _ in range(self.connections_number):
            i = np.random.choice(range(N))
            j = np.random.choice(range(N))
            # get rid of self-connections and circle connections
            if self.connections[i][j] or self.connections[j][i] or (i == j):
                continue
            self.connections[i][j] = 1

    def info(self):
        print(self.connections)

    @classmethod
    def creations(cls, P, N):
        # let brain know its connections
        gene_pool = [Gene(N, np.random.rand() / 2) for i in range(P)]
        return gene_pool


if __name__ == "__main__":
    N = 20
    gene_pool = Gene.creations(3, N)
    for gene in gene_pool:
        print('============================================')
        print(gene.connections)
        print(gene.connections.sum() / float(N ** 2) * 100)

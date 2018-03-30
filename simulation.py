# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
from environment import Environment
from neural_network import NeuralNetwork
from stimulus import StimuliPool
from gene import Gene

# configs
N = 4  # neurons number in a neural netowrk
S = 5  # stimuli pool size
G = 2  # generations of evolution
P = 20  # population of neural network pool
I = 10  # iterations of stimuli on neural network
strengthen_rate = 0.00005
#
np.random.seed()
stimu_pool = StimuliPool(N, S)
env = Environment()
# nn_pool = NeuralNetwork.creations(P, N)
gene_pool = Gene.creations(P, N)
for g in range(G):
    # One Generation
    print('----------------------------- Generation %s -----------------------------' % g)
    # continiously stimulate NNs with randomly picked stimuli for I iterations
    stats_l = env.evaluate_fitness(gene_pool, stimu_pool, NeuralNetwork, I, strengthen_rate)
    # env.output_stats(nn_pool)
    # env.store_stats(nn_pool)
    print(stats_l[0]['accuracy'])
    mating_pool = env.select_mating_pool(gene_pool, stats_l)
    new_gene_pool = env.reproduce_nn_pool(mating_pool, P)
    gene_pool = new_gene_pool
# for nn in nn_pool: print nn.i
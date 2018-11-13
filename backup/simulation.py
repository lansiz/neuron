# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
from environment import Environment
from neural_network import NeuralNetwork
from stimulus import StimuliPool
from gene import Gene

# configs
P = 40  # population of neural network pool
N = 10  # neurons number in a neural netowrk
S = 1  # stimuli pool size
G = 30 # generations of evolution
I = 10 * 10 ** 3   # iterations of stimuli on neural network
strengthen_rate = 0.001
#
print('#population: %s #neuron: %s #propagation: %s #generation: %s #iteration: %s lr: %s' % (
    P, N, S, G, I, strengthen_rate))
np.random.seed()
stimu_pool = StimuliPool(N, S)
print((' stimulation ').center(100, '-'))
stimu_pool.info()
env = Environment()
# nn_pool = NeuralNetwork.creations(P, N)
gene_pool = Gene.creations(P, N)

for g in range(G):
    # One Generation
    print((' generation %s ' % g).center(100, '-'))
    # continiously stimulate NNs with randomly picked stimuli for I iterations
    stats_l = env.evaluate_fitness(gene_pool, stimu_pool, NeuralNetwork, I, strengthen_rate)
    # env.output_stats(nn_pool)
    # env.store_stats(nn_pool)
    # print(stats_l[0]['accuracy'])
    new_gene_pool = env.build_new_gene_pool(P, gene_pool, stats_l, strength_threshhold=.15)
    if not new_gene_pool:
        break
    # print(mating_pool)
    gene_pool = new_gene_pool
# for nn in nn_pool: print nn.i

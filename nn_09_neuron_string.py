import numpy as np
import multiprocessing
from nn import NeuralNetwork
from gene import Gene
import strengthen_functions
import argparse

N = 40
trails = 20
pf = strengthen_functions.PF80

# bulid a neuron string
connection_matrix = np.zeros(N ** 2).reshape((N, N))
for i in range(N - 1):
    connection_matrix[i][i + 1] = 1


def seek_fp(x):
    np.random.seed()
    nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
    nn.set_strengthen_functions(pf)
    nn.initialize_synapses_strength(.5, .1)
    # training
    for _ in range(100000):
        if x > np.random.rand():
            neurons_stimulated = set([0])
        else:
            neurons_stimulated = set([])
        nn.propagate_once(neurons_stimulated)
    # testing
    l_ = []
    for _ in range(300000):
        neurons_stimulated = set([0])
        l_.append(nn.propagate_test(neurons_stimulated))
    return np.array(l_)


xs = np.linspace(0, 1, trails)

results_l = [seek_fp(x) for x in xs]
print('mean=', [i.mean() for i in results_l])
print('std=', [i.std() for i in results_l])

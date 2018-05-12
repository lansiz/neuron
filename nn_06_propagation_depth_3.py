import numpy as np
import multiprocessing
from nn import NeuralNetwork
from gene import Gene
import strengthen_functions
import argparse

N = 40
trails = 20

parser = argparse.ArgumentParser()
parser.add_argument('-a', action="store", dest="a", default=3)
args = parser.parse_args()
a = int(args.a)

np.random.seed()
# connection_matrix = Gene(N, .7).connections
connection_matrix = np.zeros(N ** 2).reshape((N, N))
for i in range(N-1):
    connection_matrix[i][i + 1] = 1

pf = lambda x: (np.exp(a * x) / (1 + np.exp(a * x)) - .5) * 1.8 + .05


def seek_fp(x, a):
    nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
    nn.set_strengthen_functions(pf)
    nn.initialize_synapses_strength(.5, .1)
    for _ in range(100000):
        if x > np.random.rand():
            neurons_stimulated = set([0])
        else:
            neurons_stimulated = set([])
        nn.propagate_once(neurons_stimulated)
    l = []
    for _ in range(100000):
        neurons_stimulated = set([0])
        l.append(nn.propagate_test(neurons_stimulated))
    return np.array(l)


xs = np.linspace(0, 1, trails)

results_l = [seek_fp(x, a) for x in xs]
print '%s:' % a, [i.mean() for i in results_l]

import numpy as np
from nn import NeuralNetwork
from gene import Gene
import strengthen_functions
'''
import matplotlib as mpl
mpl.use('Agg', warn=False)
'''
import matplotlib.pyplot as plt

N = 8
# connection_matrix = Gene(N, .7).connections
connection_matrix = np.array([
    [0, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0]])
neurons_stimulated_probs = np.random.normal(.6, .1, N)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
for i in range(11):
    nn = NeuralNetwork(connection_matrix, transmission_history_len=10**3)
    nn.set_strengthen_functions(strengthen_functions.PF34)
    nn.initialize_synapses_strength(.1 * i, .1)
    strength_stats = []
    for _ in range(60000):
        neurons_stimulated = set(np.where(neurons_stimulated_probs > np.random.rand(N))[0])
        nn.propagate_once(neurons_stimulated)
        strength_stats.append(nn.stats()['strength'])
    ax.plot(strength_stats)
    ax.set_ylim(0, 1)
# plt.grid(True)
plt.savefig('nn_02.png')
plt.show()

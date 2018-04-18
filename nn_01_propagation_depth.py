import numpy as np
from nn import NeuralNetwork
# import strengthen_functions
'''
import matplotlib as mpl
mpl.use('Agg', warn=False)
'''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

connection_matrix = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# pf = strengthen_functions.PF34


def pf_factory(a, b, c):
    def pf(x):
        return (np.exp(a * x) / (1 + np.exp(a * x)) - .5) * b + c
    return pf


for pf, color in zip(
    [pf_factory(i, 1.8, .05) for i in range(1, 8)],
    ('red', 'gray', 'blue', '#CD8500', 'green', 'purple', '#C5C1AA')):
    frequency_l = []
    for i in range(10):
        nn = NeuralNetwork(connection_matrix, transmission_history_len=10**3)
        nn.initialize_synapses_strength(.5, .1)
        nn.set_strengthen_functions(pf)
        for j in range(15000):
            nn.propagate_once(set([0]))
        frequency_matrix = nn.get_transmission_frequency()
        del nn
        frequency_l.append([
            frequency_matrix[0][1],
            frequency_matrix[1][2],
            frequency_matrix[2][3],
            frequency_matrix[3][4],
            frequency_matrix[4][5],
            frequency_matrix[5][6],
            frequency_matrix[6][7],
            frequency_matrix[7][8],
            frequency_matrix[8][9],
            frequency_matrix[9][10],
            frequency_matrix[10][11]])
    frequency_a = np.array(frequency_l).mean(axis=0)
    for l in frequency_l:
        ax.plot(l, 'o', color=color, alpha=.6, zorder=1)
    ax.plot(frequency_a, color=color, linewidth=2, alpha=.4, zorder=0)
plt.savefig('./nn_01.png')
plt.show()

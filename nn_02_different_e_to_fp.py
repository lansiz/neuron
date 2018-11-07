import numpy as np
import multiprocessing
from nn import NeuralNetwork
# from gene import Gene
import strengthen_functions
'''
import matplotlib as mpl
mpl.use('Agg', warn=False)
'''
import matplotlib.pyplot as plt

N = 8
trails = 10
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
# neurons_stimulated_probs = np.random.normal(.6, 0, N)
neurons_stimulated_probs = np.array([.8, .3, .2, .6, .5, .7, .4, .9])


def seek_fp(x):
    nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
    nn.set_strengthen_functions(strengthen_functions.PF12)
    nn.initialize_synapses_strength(x, .1)
    strength_stats = []
    for _ in range(200000):
        neurons_stimulated = set(
            np.where(neurons_stimulated_probs > np.random.rand(N))[0])
        nn.propagate_once(neurons_stimulated)
        strength_stats.append(nn.stats()['strength'])
    return strength_stats


np.random.seed()
worker_pool = multiprocessing.Pool(processes=4)
xs = np.linspace(0, 1, trails)
results_l = [worker_pool.apply_async(seek_fp, (x,)).get() for x in xs]

fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
for t in range(trails):
    ax.plot(results_l[t])
ax.tick_params(labelsize=8)

# ax.set_ylim(0, 1)
plt.savefig('nn_02_pf12.png')
plt.show()

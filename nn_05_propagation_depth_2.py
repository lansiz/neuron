import numpy as np
import multiprocessing
from nn import NeuralNetwork
from gene import Gene
import strengthen_functions
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt

N = 20
trails = 10
# connection_matrix = Gene(N, .7).connections
connection_matrix = np.zeros(N ** 2).reshape((N, N))
for i in range(N-1):
    connection_matrix[i][i + 1] = 1

a = 3
pf = lambda x: (np.exp(a * x) / (1 + np.exp(a * x)) - .5) * 1.8 + .05

def seek_fp(x): 
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
    for _ in range(10000):
        neurons_stimulated = set([0])
        l.append(nn.propagate_test(neurons_stimulated))
    print l
    return np.array(l)

worker_pool = multiprocessing.Pool(processes=4)
xs = np.linspace(0, 1, trails)
results_l = [worker_pool.apply_async(seek_fp, (x,)).get() for x in xs]

fig, ax = plt.subplots(1, 1, figsize=(3, 2))
# for t in range(trails):
ax.plot(xs, [i.mean() for i in results_l], '.-')
ax.tick_params(labelsize=8)

# ax.set_ylim(0, 1)
plt.savefig('nn_03.png')
plt.show()

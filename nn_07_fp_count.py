import numpy as np
import multiprocessing
from nn import NeuralNetwork
# from gene import Gene
import strengthen_functions
import matplotlib.pyplot as plt

N = 4
trails = 20
connection_matrix = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]])
# neurons_stimulated_probs = np.array([.8, .3, .2, .6, .5, .7, .4, .9])
# neurons_stimulated_probs = np.array([.8, .3, .2, .6])
neurons_stimulated_probs = np.array([.8, .3, .8, .2])
pf13 = strengthen_functions.PF13  # 4 fp
pf12 = strengthen_functions.PF12  # 2 fp
pf32 = strengthen_functions.PF32  # 1 fp
# self.strengthen_functions_m = np.where(self.connection_matrix == 1, pf, '----')
strengthen_functions_m = np.array([
    [0, pf12, pf32, 0],
    [0, 0, pf32, pf32],
    [0, pf12, 0, 0],
    [pf32, 0, pf32, 0]])


def seek_fp(x):
    nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
    # nn.set_strengthen_functions(strengthen_functions.PF32)
    nn.strengthen_functions_m = strengthen_functions_m
    nn.initialize_synapses_strength_uniform()
    strength_stats = []
    for _ in range(200000):
        neurons_stimulated = set(
            np.where(neurons_stimulated_probs > np.random.rand(N))[0])
        nn.propagate_once(neurons_stimulated)
        strength_stats.append(nn.stats()['strength'])
    print(nn.connection_strength_m.round(2))
    print('')
    return strength_stats


np.random.seed()
worker_pool = multiprocessing.Pool(processes=4)
xs = np.linspace(0, 1, trails)
results_l = [worker_pool.apply_async(seek_fp, (x,)).get() for x in xs]

fig, ax = plt.subplots(1, 1, figsize=(3, 5))
for t in range(trails):
    ax.plot(results_l[t])
ax.tick_params(labelsize=12)

# ax.set_ylim(0, 1)
# plt.savefig('fig_07_4*2*2*1.png')
plt.savefig('nn_07_2*2*1.png')
plt.show()

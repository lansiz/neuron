import numpy as np
from neural_network import NeuralNetwork
from gene import Gene
from stimulus import StimuliPool
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
N = 10
S = 2
stimu_pool = StimuliPool(N, S)
g = Gene(N, .5)
stimu_pool.data = [[set([0]), [set([0, 6])]]]
g.connections = np.array([
        [0,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,1,0,0]])
g.connections_number = g.connections.sum()
print(' gene (connection matrix) '.center(100, '-'))
g.info()
nn = NeuralNetwork(g)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions()
print(' strength matrix to start '.center(100, '-'))
print(nn.connection_strength_m_origin.round(4))
print(' stimuli pool '.center(100, '-'))
stimu_pool.info()
print(' strengthen functions matrix '.center(100, '-'))
# print(nn.strengthen_functions_m)
nn.show_strengthen_functions_matrix()
accuracy_stats = []
strength_stats = []
for _ in range(60000):
    nn.propagate_once(stimu_pool, strengthen_rate=0.001)
    strength_stats.append(nn.stats()['strength'])
    if _ % 1000 == 0:
        # nn.evaluate_accuracy(stimu_pool)
        # accuracy_stats.append(nn.stats()['accuracy'])
        strength_stats.append(nn.stats()['strength'])
print(' strength matrix at fixed point '.center(100, '-'))
print(nn.connection_strength_m.round(4))
print(' transmission frequency at fixed point '.center(100, '-'))
print(nn.get_transmission_frequency())
print(' srength '.center(100, '-'))
# print(nn.accuracy)
plt.plot(strength_stats)
plt.savefig('./simu_05.png')


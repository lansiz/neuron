import numpy as np
from nn import NeuralNetwork
import strengthen_functions
'''
import matplotlib as mpl
mpl.use('Agg', warn=False)
'''
# import matplotlib.pyplot as plt
N = 2
connection_matrix = np.array([
[0, 1],
[1, 0]])
'''
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]])
'''
pf = strengthen_functions.PF34
# neurons_stimulated_probs = np.array([.3, .8, .2, .6])
neurons_stimulated_probs = np.array([.8, .1])


nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions(pf)
for j in range(15000):
    neurons_stimulated = set(np.where(neurons_stimulated_probs > np.random.rand(N))[0])
    nn.propagate_once(neurons_stimulated)
frequency_matrix = nn.get_transmission_frequency()
print(frequency_matrix)


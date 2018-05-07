import numpy as np
from nn import NeuralNetwork
import strengthen_functions
'''
import matplotlib as mpl
mpl.use('Agg', warn=False)
'''
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 1, figsize=(6, 3))

N = 7
connection_matrix = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]])
pf = strengthen_functions.PF35


neurons_stimulated_probs = np.array([.9, .5, 0, 0, 0, 0, 0])
nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions(pf)
for j in range(60000):
    # nn.propagate_once(set([0]))
    # nn.propagate_once(set([1]))
    neurons_stimulated = set(np.where(neurons_stimulated_probs > np.random.rand(N))[0])
    nn.propagate_once(neurons_stimulated)
print(nn.connection_strength_m)
print(nn.get_transmission_frequency())


nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions(pf)
for j in range(60000):
    nn.propagate_once(set([0]))
    nn.propagate_once(set([1]))
print(nn.connection_strength_m)
print(nn.get_transmission_frequency())


nn = NeuralNetwork(connection_matrix, transmission_history_len=10**4)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions(pf)
for j in range(60000):
    nn.propagate_once(set([0, 1]))
    nn.propagate_once(set())
print(nn.connection_strength_m)
print(nn.get_transmission_frequency())

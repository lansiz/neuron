import numpy as np
from nn import NeuralNetwork
import strengthen_functions
'''
import matplotlib as mpl
mpl.use('Agg', warn=False)
'''
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 1, figsize=(6, 3))

connection_matrix = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]])
pf = strengthen_functions.PF35

nn = NeuralNetwork(connection_matrix, transmission_history_len=10**3)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions(pf)
for j in range(15000):
    nn.propagate_once(set([0]))
    nn.propagate_once(set([1]))
print(nn.connection_strength_m)
print(nn.get_transmission_frequency())


nn = NeuralNetwork(connection_matrix, transmission_history_len=10**3)
nn.initialize_synapses_strength(.5, .1)
nn.set_strengthen_functions(pf)
for j in range(15000):
    nn.propagate_once(set([0, 1]))
    nn.propagate_once(set())
print(nn.connection_strength_m)
print(nn.get_transmission_frequency())

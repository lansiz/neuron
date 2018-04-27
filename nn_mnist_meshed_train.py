
from __future__ import print_function
import numpy as np
import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_meshed import NeuralNetwork
import mnist
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-m', action="store", dest="mnist_number")
parser.add_argument('-s', action="store", dest="memory_neurons_number", default=500, help="default: 500")
parser.add_argument('-c', action="store", dest="connections_per_sensor", default=2, help="default: 2")
parser.add_argument('-r', action="store", dest="connections_per_memory_neuron", default=5, help="default: 5")
parser.add_argument('-i', action="store", dest="iterations", default=20000, help="default: 20000")
args = parser.parse_args()
mnist_number = int(args.mnist_number)
memory_neurons_number = int(args.memory_neurons_number)
connections_per_sensor = int(args.connections_per_sensor)
iterations = int(args.iterations)
connections_per_memory_neuron = int(args.connections_per_memory_neuron)
parameters_l = (mnist_number, memory_neurons_number, connections_per_sensor, connections_per_memory_neuron)
affix = '%s_%s_%s_%s' % parameters_l

print('mnist_number: %s memory_neurons_number: %s connections_per_sensor: %s connections_per_memory_neuron: %s' % parameters_l)
print('iterations: %s' % iterations)

nn = NeuralNetwork(
    connections_per_memory_neuron=connections_per_memory_neuron,
    memory_neurons_number=memory_neurons_number,
    connections_per_sensor=connections_per_sensor)
nn.initialize_synapses_strength()
nn.set_strengthen_functions()

imgs = mnist.get_imgs_by_number(mnist_number)
img = imgs[100][1]
stimulated = set(np.where(img.flatten() > 0)[0])

start_time = datetime.datetime.now()

plotting_strength = False
if plotting_strength: strength_stats = []
for i in range(iterations):
    nn.propagate_once(stimulated, debug=False)
    if plotting_strength:
        if i % 10 == 0: strength_stats.append(nn.stats()['strength'])

end_time = datetime.datetime.now()
print('start time:', start_time)
print('stop time: ', end_time)

if plotting_strength:
    plt.plot(strength_stats)
    plt.savefig('./nn_mnist_meshed_%s.png' % affix)
utils.write_pickle(nn.connection_strength_m, './pkl/nn_mnist_meshed_%s.pkl' % affix)


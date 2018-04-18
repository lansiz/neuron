
from __future__ import print_function
import numpy as np
import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import mnist
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="mnist_number")
parser.add_argument('-i', action="store", dest="iterations", default=20000, help="default: 20000")
args = parser.parse_args()
mnist_number = int(args.mnist_number)
iterations = int(args.iterations)

nn = NeuralNetwork()

imgs = mnist.get_imgs_by_number(mnist_number)
img = imgs[100][1]

start_time = datetime.datetime.now()

plotting_strength = True
if plotting_strength: strength_stats = []
for i in range(iterations):
    stimulated = set(np.where(img.flatten() > 0)[0])
    nn.propagate_once(stimulated)
    if plotting_strength:
        if i % 10 == 0: strength_stats.append(nn.stats()['strength'])

end_time = datetime.datetime.now()
print('start time:', start_time)
print('stop time: ', end_time)

if plotting_strength:
    plt.plot(strength_stats)
    plt.savefig('./nn_mnist_jellyfish_%s.png' % mnist_number)
utils.write_pickle(nn.connections_matrix, 'nn_mnist_jellyfish_%s.pkl' % mnist_number)

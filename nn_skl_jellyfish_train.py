
from __future__ import print_function
import numpy as np
import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import skl
import utils
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="mnist_number")
parser.add_argument('-i', action="store", dest="iterations", default=25000, help="default: 25000")
args = parser.parse_args()
mnist_number = int(args.mnist_number)
iterations = int(args.iterations)

# images choice indexes = [181, 100, 333, 100, 3282, 5239, 5070, 893, 2117, 5712]

pf = lambda x: (1 / (1 + np.exp(-1 * 10 * x)) - .5) * 1.8 + .05
nn = NeuralNetwork(strength_function=pf, image_scale=8)

imgs = skl.get_imgs_by_number(mnist_number)
i = random.choice(range(len(imgs)))
img = imgs[i][1]
print('%s: #%s' % (mnist_number, i))

start_time = datetime.datetime.now()

plotting_strength = True
if plotting_strength: strength_stats = []

for i in range(iterations):
    nn.propagate_once(img, gray_max=16)
    if plotting_strength:
        if i % 10 == 0: strength_stats.append(nn.stats()['strength'])

end_time = datetime.datetime.now()
print('start time:', start_time)
print('stop time: ', end_time)

if plotting_strength:
    plt.plot(strength_stats)
    plt.savefig('./png/nn_mnist_jellyfish_%s.png' % mnist_number)

utils.write_pickle(nn.connections_matrix, './pkl/nn_mnist_jellyfish_%s.pkl' % mnist_number)


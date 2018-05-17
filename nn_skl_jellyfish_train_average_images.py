
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
parser.add_argument('-n', action="store", dest="number")
parser.add_argument('-i', action="store", dest="iterations", default=25000, help="default: 25000")
args = parser.parse_args()
number = int(args.number)
iterations = int(args.iterations)

pf = lambda x: (1 / (1 + np.exp(-1 * 10 * x)) - .5) * 1.8 + .05
nn = NeuralNetwork(strength_function=pf, image_scale=8)

imgs = skl.get_imgs_by_number(number)
size = float(len(imgs))
img = np.zeros([8, 8])
for label, i in imgs:
    img += i
img = img / size 

start_time = datetime.datetime.now()
for i in range(iterations):
    nn.propagate_once(img, gray_max=16)
end_time = datetime.datetime.now()

print('%s: ' % number, 'start time:', start_time, 'stop time: ', end_time)

utils.write_pickle(nn.connections_matrix, './pkl/nn_mnist_jellyfish_%s.pkl' % number)


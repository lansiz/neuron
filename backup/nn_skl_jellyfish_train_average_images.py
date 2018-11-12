
from __future__ import print_function
import numpy as np
# import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
# import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import skl
import utils
# import sys
import strengthen_functions

parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="number")
parser.add_argument('-i', action="store", dest="iterations", default=30000, help="default: 30000")
args = parser.parse_args()
number = int(args.number)
iterations = int(args.iterations)

pf = strengthen_functions.PF80
nn = NeuralNetwork(strength_function=pf, image_scale=8, transmission_history_len=10**4, propagation_depth=1)

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


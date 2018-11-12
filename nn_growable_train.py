
from __future__ import print_function
# import numpy as np
# import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
# import matplotlib.pyplot as plt
from nn_growable import NeuralNetwork
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

pf = strengthen_functions.PF81
nn = NeuralNetwork(strength_function=pf, image_scale=8, transmission_history_len=10**4)

train_imgs, _ = skl.load_data()
train_imgs = train_imgs[number]
size = len(train_imgs)

start_time = datetime.datetime.now()
for i in range(iterations):
    i_ = i % size
    nn.propagate_once(train_imgs[i_], gray_max=16)
end_time = datetime.datetime.now()

print('%s: ' % number, 'start time:', start_time, 'stop time: ', end_time)

utils.write_pickle(nn.connections_matrix, './pkl/nn_growable_%s.pkl' % number)



from __future__ import print_function
import numpy as np
import random
import sys
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import mnist
import utils

# the number of 0~9 to train NN with
parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="test_number")
parser.add_argument('-m', action="store", dest="train_number")
parser.add_argument('-p', action="store", dest="train_pickle_file_name")
parser.add_argument('-i', action="store", dest="iterations", default=6000)
parser.add_argument('-t', action="store", dest="method", default=1)
args = parser.parse_args()
train_number = int(args.train_number)
test_number = int(args.test_number)
method = int(args.method)
train_pickle_file_name = str(args.train_pickle_file_name)
iterations = int(args.iterations)

imgs_train = mnist.get_imgs_by_number(train_number)
imgs_test = mnist.get_imgs_by_number(test_number)
strength_matrix = utils.read_pickle(train_pickle_file_name)

if strength_matrix is None:
    print('cannot find %s' % train_pickle_file_name)
    sys.exit(1)
score = .0
for _ in range(iterations):
    conns_l, strength_l = NeuralNetwork.validate(random.choice(imgs_train)[1], strength_matrix)
    if method == 1:
        stats_train = len(conns_l)
    else:
        stats_train = np.array(strength_l).sum()
    conns_l, strength_l = NeuralNetwork.validate(random.choice(imgs_test)[1], strength_matrix)
    if method == 1:
        stats_test = len(conns_l)
    else:
        stats_test = np.array(strength_l).sum()
    if stats_train > stats_test:
        score += 1
print('%s %s %s' % (train_number, test_number, score / iterations))

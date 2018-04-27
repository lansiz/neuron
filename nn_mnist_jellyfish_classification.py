
from __future__ import print_function
import numpy as np
import random
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import mnist
import utils

# the number of 0~9 to train NN with
parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", dest="iterations", default=10000)
args = parser.parse_args()
iterations = int(args.iterations)
# print('train', train_number, 'test', test_number)

imgs = mnist.get_imgs_by_number()
strength_matrix_l = [utils.read_pickle('pkl/nn_mnist_jellyfish_' + str(i) + '.pkl') for i in range(10)]

correct = .0
for _ in range(iterations):
    label, img = random.choice(imgs)
    scores_a = np.array([len(NeuralNetwork.validate(img, strength_matrix)[0]) for strength_matrix in strength_matrix_l])
    if scores_a[label] == scores_a.max():
        correct += 1
print(correct / iterations)

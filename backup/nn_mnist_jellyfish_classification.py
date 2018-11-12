
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
parser.add_argument('-i', action="store", dest="iterations", default=30000)
parser.add_argument('-j', action="store", dest="num", default=-1)
args = parser.parse_args()
iterations = int(args.iterations)
num = int(args.num)

if num >= 0:
    imgs = mnist.get_imgs_by_number(num)
else:
    imgs = mnist.get_imgs_by_number()

strength_matrix_l = [utils.read_pickle('pkl/nn_mnist_jellyfish_' + str(i) + '.pkl') for i in range(10)]

correct = .0
trails = .0
# for label, img in imgs[:iterations]:
for label, img in imgs[:iterations]:
    scores_a = np.array([NeuralNetwork.validate(img, strength_matrix) for strength_matrix in strength_matrix_l])
    if label == random.choice(np.where(scores_a == scores_a.max())[0]):
        # print(scores_a)
        correct += 1
    trails += 1

print(correct / trails)


from __future__ import print_function
import numpy as np
import random
import argparse
from nn_growable import NeuralNetwork
import skl
import utils

# the number of 0~9 to train NN with
parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", dest="iterations", default=50000)
parser.add_argument('-j', action="store", dest="num", default=-1)
args = parser.parse_args()
iterations = int(args.iterations)
num = int(args.num)

threshhold = 0.3
power = 2
weight = 20

# print('threshhold %s power %s weight %s' % (threshhold, power, weight))

if num >=0:
    imgs = skl.get_imgs_by_number(num)
else:
    imgs = skl.get_imgs_by_number()
# print('test imgs #', size)

strength_matrix_l = [utils.read_pickle('pkl/nn_growable_' + str(i) + '.pkl') for i in range(10)]

correct = .0
trails = .0
for i in range(iterations):
    label, img = random.choice(imgs)
    scores_a = np.array([NeuralNetwork.validate_linear(
        img, strength_matrix, gray_max=16, threshhold=threshhold, power=power, weight=weight) for strength_matrix in strength_matrix_l])
    if label == random.choice(np.where(scores_a == scores_a.max())[0]):
        correct += 1
        if not (i % 1000):
            # print(label, scores_a)
            pass
    trails += 1

print(round(correct / trails, 4))


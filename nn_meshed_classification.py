
from __future__ import print_function
import numpy as np
import random
import argparse
from nn_meshed import NeuralNetwork
import skl
import utils

# the number of 0~9 to train NN with
parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", dest="iterations", default=1000)
parser.add_argument('-j', action="store", dest="num", default=-1)
args = parser.parse_args()
iterations = int(args.iterations)
num = int(args.num)

'''
threshhold = 0.8
weight = 100
print('threshhold %s weight %s' % (threshhold, weight))
'''

if num >=0:
    imgs = skl.get_imgs_by_number(num)
else:
    imgs = skl.get_imgs_by_number()

strength_matrix_l = [utils.read_pickle('pkl/nn_meshed_' + str(i) + '.pkl') for i in range(10)]

correct = .0
trails = .0
for i in range(iterations):
    trails += 1
    label, img = random.choice(imgs)
    scores_a = np.array([NeuralNetwork.validate(img, strength_matrix, gray_max=16.) for strength_matrix in strength_matrix_l])
    if label == random.choice(np.where(scores_a == scores_a.max())[0]):
        correct += 1
        if not (i % 10) and  i > 0:
            # print(round(correct / trails * 100, 2), label, scores_a)
            pass

print(round(correct / trails, 4))


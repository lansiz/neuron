from __future__ import print_function
import numpy as np
import random
import argparse
from nn_growable import NeuralNetwork
import skl
import utils

random.seed()

iterations = 50000

validators = [
    lambda img, matrix: NeuralNetwork.validate_original(img, matrix),
    lambda img, matrix: NeuralNetwork.validate_step(img, matrix, threshold=.6),
    lambda img, matrix: NeuralNetwork.validate_linear(img, matrix, power=3, weight=100),
    lambda img, matrix: NeuralNetwork.validate_threshold(img, matrix, power=3, threshhold=.2, weight=10)]

imgs = skl.get_imgs_by_number(1)
label, img = imgs[100]

strength_matrix_l = [utils.read_pickle('pkl/nn_growable_' + str(i) + '.pkl') for i in range(10)]

trails = .0
targets_l = [0] * 10
for i in range(iterations):
    # label, img = random.choice(imgs)
    scores_a = np.array([validators[3](img, strength_matrix) for strength_matrix in strength_matrix_l])
    target = random.choice(np.where(scores_a == scores_a.max())[0])
    targets_l[target] += 1
    trails += 1

targets_a = np.array(targets_l)
print((targets_a / trails * 100).round(2).tolist())



from __future__ import print_function
import numpy as np
# import random
# import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_growable import NeuralNetwork
# import mnist
import skl
import utils
# import seaborn as sns
# import sys
import random

iters = 5 * 10 ** 4
num = 6

imgs_l = [skl.get_imgs_by_number(i) for i in range(10)]
strength_matrix_l = [utils.read_pickle('pkl/nn_growable_' + str(i) + '.pkl') for i in range(10)]

fig, axes = plt.subplots(4, 1, figsize=(6, 7), sharex=True, sharey=False)
axes = axes.flatten()

validators = [
    lambda img, matrix: NeuralNetwork.validate_original(img, matrix),
    lambda img, matrix: NeuralNetwork.validate_step(img, matrix, threshold=.6),
    lambda img, matrix: NeuralNetwork.validate_linear(img, matrix, power=3, weight=10),
    lambda img, matrix: NeuralNetwork.validate_threshold(img, matrix, power=3, threshhold=.2, weight=10)]

imgs = imgs_l[num]
for k, vldt in enumerate(validators):
    matrix = strength_matrix_l[num]
    results_l = [[], [], [], [], [], [], [], [], [], []]
    for matrix, result in zip(strength_matrix_l, results_l):
        for i in range(iters):
            img = random.choice(imgs)[1]
            result.append(vldt(img, matrix))

    # get the mu and std for z-scores
    results_l_part = [i for i in results_l if i != num]
    neg = []
    for i in results_l_part:
        neg += i
    neg = np.array(neg)
    neg_mu = neg.mean()
    neg_std = neg.std()
    pos = np.array(results_l[num])
    print(vldt)
    print('negative:', round(neg_mu, 2), round(neg_std, 2), 'positive:', round(pos.mean(), 2), round(pos.std(), 2))

    results_a = [np.array(i) for i in results_l]
    for i, result in enumerate(results_a):
        # turn to z-score
        result = (result - neg_mu) / neg_std
        if i != num:
            color = 'gray'
            zorder = 0
            line = 1
        else:
            color = 'red'
            zorder = 1
            line = 2
            print('pos z-score:', round(result.mean(), 2), round(result.std(), 2))
        axes[k].hist(result, histtype='step', density=False, color=color, linewidth=line, zorder=zorder, bins=20)
        axes[k].tick_params(labelsize=8)
        axes[k].set_xlim(-4, 4)
        # axes[j].set_xlim(0, 50)
        axes[k].tick_params(labelsize=10)
plt.savefig('./nn_growable_6_digit.png')
# plt.show()



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

imgs_l = [skl.get_imgs_by_number(i) for i in range(10)]
strength_matrix_l = [utils.read_pickle('pkl/nn_growable_' + str(i) + '.pkl') for i in range(10)]

fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=False)
axes = axes.flatten()

pfs = [
    lambda img, matrix: NeuralNetwork.validate_linear(img, matrix, gray_max=16, power=0, weight=10),
    lambda img, matrix: NeuralNetwork.validate_linear(img, matrix, gray_max=16, power=1, weight=10),
    lambda img, matrix: NeuralNetwork.validate_linear(img, matrix, gray_max=16, power=3, weight=10),
    lambda img, matrix: NeuralNetwork.validate_threshold(img, matrix, gray_max=16, power=3, threshhold=.2, weight=10)]

# if True:
# for j, matrix in enumerate(strength_matrix_l):
for k, pf in enumerate(pfs):
    j = 6
    matrix = strength_matrix_l[j]
    results_l = [[], [], [], [], [], [], [], [], [], []]
    for imgs, result in zip(imgs_l, results_l):
        for i in range(iters):
            img = random.choice(imgs)[1]
            result.append(pf(img, matrix))

    # get the mu and std for z-scores
    results_l_part = [i for i in results_l if i != j]
    temp = []
    for i in results_l_part:
        temp += i
    temp = np.array(temp)
    mu = temp.mean()
    std = temp.std()
    # print(mu, std)

    results_a = [np.array(i) for i in results_l]
    for i, result in enumerate(results_a):
        # turn to z-score
        result = (result - mu) / std
        if i != j:
            color = 'gray'
            zorder = 0
            line = 1
        else:
            color = 'red'
            zorder = 1
            line = 2
            print(result.mean(), result.std())
        # sns.kdeplot(result, color=color, shade=False, shade_lowest=False, alpha=1, zorder=zorder)
        axes[k].hist(result, histtype='step', density=False, color=color, linewidth=line, zorder=zorder, bins=20)
        # axes.axvline(x=mu, linewidth=1, color=color, zorder=zorder)
        axes[k].tick_params(labelsize=8)
        axes[k].set_xlim(-4, 4)
        # axes[j].set_xlim(0, 50)
        axes[k].tick_params(labelsize=10)
plt.savefig('./nn_growable_6_NN.png')
# plt.show()


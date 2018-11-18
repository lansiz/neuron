from __future__ import print_function
import random
import utils
import skl
from nn_meshed import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
# import random
# import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
# import mnist
# import seaborn as sns
# import sys

iters = 1 * 10 ** 4
j = 6

imgs_l = [skl.get_imgs_by_number(i) for i in range(10)]
strength_matrix_l = [utils.read_pickle(
    'pkl/nn_meshed_' + str(i) + '.pkl') for i in range(10)]

fig, axes = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
# axes = axes.flatten()

# for j, matrix in enumerate(strength_matrix_l):
if True:
    matrix = strength_matrix_l[j]
    results_l = [[], [], [], [], [], [], [], [], [], []]
    for imgs, result in zip(imgs_l, results_l):
        for i in range(iters):
            if i == j:
                img = imgs[100][1]
            else:
                img = random.choice(imgs)[1]
            result.append(NeuralNetwork.validate(img, matrix, gray_max=16))

    for i, result in enumerate(results_l):
        if i != j:
            color = 'gray'
            zorder = 0
            line = 1
        else:
            color = 'red'
            zorder = 1
            line = 2
        mu = np.mean(result)
        std = np.std(result)
        axes.hist(result, histtype='step', density=True,
                  color=color, linewidth=line, zorder=zorder, bins=20)
        axes.tick_params(labelsize=12)
        # axes[j].set_xlim(0, 50)
plt.savefig('./nn_meshed_z_dist_6.png')
# plt.show()

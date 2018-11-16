
from __future__ import print_function
import numpy as np
# import random
# import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_meshed import NeuralNetwork
# import mnist
import skl
import utils
# import seaborn as sns
# import sys
import random

iters = 1 * 10 ** 4

imgs_l = [skl.get_imgs_by_number(i) for i in range(10)]
strength_matrix_l = [utils.read_pickle('pkl/nn_meshed_' + str(i) + '.pkl') for i in range(10)]

fig, axes = plt.subplots(5, 2, figsize=(6, 6), sharex=True, sharey=True)
axes = axes.flatten()

for j, matrix in enumerate(strength_matrix_l):
    results_l = [[], [], [], [], [], [], [], [], [], []]
    for imgs, result in zip(imgs_l, results_l):
        for i in range(iters):
            # label, img = imgs[i % size]
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
        # sns.kdeplot(result, color=color, shade=False, shade_lowest=False, alpha=1, zorder=zorder)
        axes[j].hist(result, histtype='step', density=True, color=color, linewidth=line, zorder=zorder, bins=20)
        # axes.axvline(x=mu, linewidth=1, color=color, zorder=zorder)
        axes[j].tick_params(labelsize=8)
        # axes[j].set_xlim(0, 25)
        # axes[j].set_xlim(0, 50)
plt.savefig('./nn_meshed_z_dist.png')
# plt.show()


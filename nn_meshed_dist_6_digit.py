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
num = 6

imgs_l = [skl.get_imgs_by_number(i) for i in range(10)]
strength_matrix_l = [utils.read_pickle(
    'pkl/nn_meshed_' + str(i) + '.pkl') for i in range(10)]

fig, axes = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
# axes = axes.flatten()

imgs = imgs_l[num]
index = random.choice(range(len(imgs)))
index = 3
print('index of testing img: ', index)
img = imgs[index][1]
# skl.show(img)

# for j, matrix in enumerate(strength_matrix_l):
if True:
    results_l = [[], [], [], [], [], [], [], [], [], []]
    for matrix, result in zip(strength_matrix_l, results_l):
        for i in range(iters):
            result.append(NeuralNetwork.validate(img, matrix, gray_max=16))

    for i, result in enumerate(results_l):
        if i != num:
            color = 'gray'
            zorder = 0
            line = 1
        else:
            color = 'red'
            zorder = 1
            line = 2
        mu = np.mean(result)
        std = np.std(result)
        axes.axvline(x=mu, linewidth=line, linestyle='dotted', color=color, zorder=zorder)
        axes.hist(result, histtype='step', density=True,
                  color=color, linewidth=line, zorder=zorder, bins=20)
        axes.tick_params(labelsize=12)
        # axes[j].set_xlim(0, 50)
plt.savefig('./nn_meshed_z_dist_6_digit.png')
# plt.show()

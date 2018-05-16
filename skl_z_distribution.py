
from __future__ import print_function
import numpy as np
import random
# import argparse
# import matplotlib as mpl
# mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
# import mnist
import skl
import utils
import sys
import seaborn as sns

imgs0 = skl.get_imgs_by_number(0)
imgs1 = skl.get_imgs_by_number(0)
imgs2 = skl.get_imgs_by_number(0)
imgs3 = skl.get_imgs_by_number(0)
imgs4 = skl.get_imgs_by_number(0)
imgs5 = skl.get_imgs_by_number(0)
imgs6 = skl.get_imgs_by_number(0)
imgs2 = skl.get_imgs_by_number(3)
imgs_l = [skl.get_imgs_by_number(i) for i in range(10)]
# index_picked = [np.random.choice(len(i)) for i in imgs_l]
index_picked = [35, 65, 131, 133, 40, 65, 19, 43, 95, 27]
img_picked = [i[j][1] for i, j in zip(imgs_l, index_picked)]
print(index_picked)

strength_matrix = utils.read_pickle('pkl/nn_mnist_jellyfish_0.pkl')
results_l = [[], [], [], [], [], [], [], [], [], []]
# for label, img in imgs1:
for i in range(10000):
    for img, result in zip(img_picked, results_l):
        result.append(NeuralNetwork.validate(img, strength_matrix))
    
fig, axes = plt.subplots(1, 1, figsize=(6, 2))
for i, result in enumerate(results_l):
    # axes.hist(result, bins=50, alpha=.4)
    # sns.kdeplot(latest[1], latest[2], cmap="Blues", shade=True, shade_lowest=False, alpha=.6, zorder=2)
    if i > 0:
        color = 'gray'
        zorder = 0
    else:
        color = 'red'
        zorder = 1
    mu = np.mean(result)
    std = np.std(result)
    sns.kdeplot(result, color=color, shade=False, shade_lowest=False, alpha=1, zorder=zorder)
    axes.axvline(x=mu, linewidth=1, color=color, zorder=zorder)
    print(mu, std)
axes.tick_params(labelsize=10)
plt.savefig('./minst_z_distribution.png')
plt.show()

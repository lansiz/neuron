
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

imgs1 = skl.get_imgs_by_number(0)
imgs2 = skl.get_imgs_by_number(3)

strength_matrix = utils.read_pickle('pkl/nn_mnist_jellyfish_0.pkl')

# for label, img in imgs1:
propagated_1 = []
propagated_2 = []
r = range(np.min([len(imgs1), len(imgs2)]))
index = random.choice(r)
for i in range(10000):
    propagated_1.append(NeuralNetwork.validate(imgs1[index][1], strength_matrix))
    propagated_2.append(NeuralNetwork.validate(imgs2[index][1], strength_matrix))
propagated_1 = np.array(propagated_1)
propagated_2 = np.array(propagated_2)

fig, axes = plt.subplots(2, 1, figsize=(3, 3))
axes[0].hist(propagated_1, bins=50, alpha=.8)
axes[0].hist(propagated_2, bins=50, alpha=.8)
axes[0].tick_params(labelsize=8)
axes[1].tick_params(labelsize=8)
print(propagated_1.mean(), propagated_2.mean())
plt.savefig('./minst_z_distribution.png')
plt.show()

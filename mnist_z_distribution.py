
from __future__ import print_function
import numpy as np
import random
import argparse
import matplotlib as mpl
# mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import mnist
import utils

imgs1 = mnist.get_imgs_by_number(1)
imgs3 = mnist.get_imgs_by_number(3)

strength_matrix = utils.read_pickle('pkl/nn_mnist_jellyfish_8.pkl')

# for label, img in imgs1:
propagated_1 = []
propagated_3 = []
# for label, img in imgs3:
index = random.choice(range(4000))
for i in range(10000):
    propagated_1.append(NeuralNetwork.validate(imgs1[index][1], strength_matrix))
    propagated_3.append(NeuralNetwork.validate(imgs3[index][1], strength_matrix))
propagated_1 = np.array(propagated_1)
propagated_3 = np.array(propagated_3)

fig, axes = plt.subplots(2, 1, figsize=(3, 3))
axes[0].hist(propagated_1, bins=50, alpha=.8)
axes[0].hist(propagated_3, bins=50, alpha=.8)
axes[0].tick_params(labelsize=8)
axes[1].tick_params(labelsize=8)
print(propagated_1.mean(), propagated_3.mean())
plt.savefig('./minst_z_distribution.png')
plt.show()

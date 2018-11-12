
from __future__ import print_function
import numpy as np
import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import strengthen_functions
import mnist
import utils

pf = strengthen_functions.PF37
nn = NeuralNetwork(propagation_depth=1, strength_function=pf)

imgs = mnist.get_imgs_by_number(2)
stimulus = (imgs[100][1] / 255.).flatten()

for i in range(30000):
    probs = np.random.rand(28, 28).flatten()
    # stimulated = set(np.where(stimulus > probs)[0])
    nn.propagate_once(imgs[100][1])

strength = nn.connections_matrix.flatten()
strength_target = np.array([pf(x) for x in stimulus * strength])
diff = strength_target - strength
print(strength)
print(diff.mean(), diff.std())

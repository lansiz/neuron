
from __future__ import print_function
import numpy as np
import random
import datetime
import argparse
# import matplotlib as mpl
# mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_mnist_jellyfish import NeuralNetwork
import strengthen_functions
import mnist
import utils

nn = NeuralNetwork(propagation_depth=1)

imgs = mnist.get_imgs_by_number(2)
stimulus = (imgs[100][1] / 254.).flatten()

for i in range(30000):
    probs = np.random.rand(28, 28).flatten()
    stimulated = set(np.where(stimulus > probs)[0])
    nn.propagate_once(stimulated)

strength = nn.connections_matrix.flatten()
strength_target = np.array([strengthen_functions.PF35(x) for x in stimulus * strength])
diff = strength_target - strength
print(diff.mean(), diff.std())
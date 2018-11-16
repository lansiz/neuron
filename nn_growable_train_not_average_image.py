
from __future__ import print_function
# import numpy as np
import random
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from nn_growable import NeuralNetwork
import skl
import utils
# import sys
import strengthen_functions

parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="number")
parser.add_argument('-i', action="store", dest="iterations", default=80000, help="default: 30000")
args = parser.parse_args()
number = int(args.number)
iterations = int(args.iterations)

pf = strengthen_functions.PF81
nn = NeuralNetwork(strength_function=pf, image_scale=8, transmission_history_len=10**4)

'''
train_imgs, _ = skl.load_data()
train_imgs = train_imgs[number]
size = len(train_imgs)
'''

# average_img = skl.average_img_by_number(number)
train_imgs = skl.get_imgs_by_number(number)

plotting_strength = True
if plotting_strength: strength_stats = []
start_time = datetime.datetime.now()
for i in range(iterations):
    _, img = random.choice(train_imgs)
    nn.propagate_once(img, gray_max=16)
    if plotting_strength:
        if i % 10 == 0: strength_stats.append(nn.stats()['strength'])
end_time = datetime.datetime.now()

print('%s: ' % number, 'start time:', start_time, 'stop time: ', end_time)

if plotting_strength:
    plt.plot(strength_stats)
    plt.savefig('./nn_growable_%s.png' % number)
utils.write_pickle(nn.connections_matrix, './pkl/nn_growable_%s.pkl' % number)


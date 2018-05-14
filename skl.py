'''
Based on code from https://gist.github.com/akesling/5358964
'''

import os
from sklearn import datasets, svm, metrics
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def get_imgs_by_number(num=None):
    dataset = datasets.load_digits()
    # imgs = read(dataset)
    images_and_labels = list(zip(dataset.target, dataset.images))
    if num is None:
        imgs = images_and_labels
    else:
        imgs = [i for i in images_and_labels if i[0] == num]
    return imgs


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


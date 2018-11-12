'''
Based on code from https://gist.github.com/akesling/5358964
'''

# import os
from sklearn import datasets
# import numpy as np
import random
import utils

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def get_imgs_by_number(num=None):
    dataset = datasets.load_digits()
    images_and_labels = list(zip(dataset.target, dataset.images))
    if num is None:
        imgs = images_and_labels
    else:
        imgs = [i for i in images_and_labels if i[0] == num]
    return imgs


def build_data():
    training_data = {}
    testing_data = {}
    train_size = 120
    for i in range(10):
        imgs = get_imgs_by_number(i)
        random.shuffle(imgs)
        training_data[i] = [img[1] for img in imgs[:train_size]]
        testing_data[i] = [img[1] for img in imgs[train_size:]]
    utils.write_pickle(training_data, 'train_data.pkl')
    utils.write_pickle(testing_data, 'test_data.pkl')
    print('training and testing data is shuffled and written')


def load_data():
    return utils.read_pickle('train_data.pkl'), utils.read_pickle('test_data.pkl')


def load_test_data(num=None):
    test_imgs = utils.read_pickle('test_data.pkl')
    imgs = []
    if num:
        for img in test_imgs[num]:
            imgs.append((num, img))
        size = len(imgs)
        random.shuffle(imgs)
    else:
        for i in range(10):
            for img in test_imgs[i]:
                imgs.append((i, img))
        size = len(imgs)
        random.shuffle(imgs)
    return imgs, size


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

if __name__ == "__main__":
    build_data()


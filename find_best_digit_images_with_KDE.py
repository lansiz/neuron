from __future__ import print_function# code from skit-learn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import sys

# the number that are of interest
num = 8

# load the data
digits = load_digits()
# filter the data with number of interest
data = []
target = []
for x, y in zip(digits.data, digits.target):
    if y == num:
        data.append(x)
        target.append(y)
data_size = len(data)
print('for number %s data size is %s' % (num, data_size))
# print(data[0], target[0])

# compute the average of image
image = np.zeros(64)
for i in data:
    image += i
print(image / float(data_size))

# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=15, whiten=False)
# data_ = pca.fit_transform(data)
data_ = data

# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data_)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

# print(kde.score_samples(data))
# select the one with biggest scores

# sample 44 new points from the data
sample_size = 1000
new_data = kde.sample(sample_size, random_state=0)
# new_data = pca.inverse_transform(new_data)

best_image = np.zeros(64)
for i in new_data:
    best_image += i
best_image = best_image / float(sample_size)
print(best_image.reshape([8, 8]))

plt.figure(1, figsize=(3, 3))
plt.imshow(best_image.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

sys.exit(0)
# turn data into a 4x11 grid
data = np.array(data)
new_data = new_data.reshape((4, 11, -1))
real_data = data[:44].reshape((4, 11, -1))

# plot real digits and resampled digits
fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    ax[4, j].set_visible(False)
    for i in range(4):
        im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)
        im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)

ax[0, 5].set_title('Selection from the input data')
ax[5, 5].set_title('"New" digits drawn from the kernel density model')

plt.show()

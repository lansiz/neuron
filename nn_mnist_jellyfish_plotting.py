
from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
import utils
import csv

logs_l = list(csv.reader(open('nn_mnist_jellyfish_test.log'), delimiter=' '))
scores = [float(i[6]) for i in logs_l[1:] if int(i[4]) != int(i[5])]
# print(scores)
plt.hist(scores, bins=20)
plt.savefig('nn_mnist_jellyfish_test.png')
plt.show()


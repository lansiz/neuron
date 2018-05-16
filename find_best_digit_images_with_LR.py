from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import utils

digits = datasets.load_digits()

data = digits['data']
target = digits['target']

train_test_ratio = .9
training_size = int(len(data) * train_test_ratio)
testing_size = float(len(data) - training_size)

training_x = data[:training_size]
training_y = target[:training_size]

testing_x = data[training_size:]
testing_y = target[training_size:]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1)
softmax_reg.fit(training_x, training_y)
predicted = softmax_reg.predict(testing_x)
accuracy = (predicted == testing_y).sum() / testing_size
print(accuracy) 
positive_coefs = np.where(softmax_reg.coef_, softmax_reg.coef_, 0)

l = {}
for class_, coef in zip(softmax_reg.classes_, positive_coefs):
        l[class_] = coef

print(l)
utils.write_pickle(l, 'best_images.pkl')

# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import random
import strengthen_functions


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class NeuralNetwork(object):
    def __init__(self, strength_function=strengthen_functions.PF37, strength_mean=.5, propagation_depth=30,
        strength_std=.3, transmission_history_len=10**3, image_scale=28):
        self.sensors_number = image_scale ** 2
        self.propagation_depth = propagation_depth

        self.connections_matrix = np.random.normal(strength_mean, strength_std, (self.sensors_number, self.propagation_depth))
        self.connections_matrix = np.where(self.connections_matrix <= 1, self.connections_matrix, 1)
        self.connections_matrix = np.where(self.connections_matrix >= 0, self.connections_matrix, 0)

        self.strength_function = strength_function
        self.strengthen_rate = 1. / transmission_history_len
        # each synapse has a recording queue for its transmission in the tempral order. value 1 represents the occurance of transmission
        self.transmission_history_len = transmission_history_len
        self.transmission_history_pointer = 0
        self.transmission_history_m = self.connections_matrix.copy().astype(object)
        for i in range(self.sensors_number):
            for j in range(self.propagation_depth):
                self.transmission_history_m[i][j] = Queue_(self.transmission_history_len)
        # interations already done.
        self.I = 0

        print('#sensors: %s #tentacle: %s #PF: %s' % ( self.sensors_number, self.propagation_depth, self.strength_function))

    def propagate_once(self, mnist_img):
        self.I += 1
        mnist_img = mnist_img / 255.
        stimulated_neurons = set(np.where(mnist_img.flatten() > np.random.rand(28 ** 2))[0])

        # init the transmission history slots to 0
        for i in range(self.sensors_number):
            for j in range(self.propagation_depth):
                self.transmission_history_m[i][j].records[self.transmission_history_pointer] = 0

        # propagate the stimulus
        for i in stimulated_neurons:
            for j in range(self.propagation_depth):
                if self.connections_matrix[i][j] > np.random.rand():
                    self.transmission_history_m[i][j].records[self.transmission_history_pointer] = 1
                else:
                    break

        # strengthen the connections
        if self.I > self.transmission_history_len:
            for i in range(self.sensors_number):
                for j in range(self.propagation_depth):
                    frequency = self.transmission_history_m[i][j].records.sum() / float(self.transmission_history_len)
                    target_strength = self.strength_function(frequency)
                    current_strength = self.connections_matrix[i][j]
                    if target_strength > current_strength:
                        self.connections_matrix[i][j] = np.min((current_strength + self.strengthen_rate, 1))
                    else:
                        self.connections_matrix[i][j] = np.max((0, current_strength - self.strengthen_rate))

        # moving the pointer
        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len

    @classmethod
    def validate(cls, mnist_img, connections_matrix):
        connections_propagated = 0
        propagation_depth = connections_matrix.shape[1]
        mnist_img = mnist_img / 255
        stimulated_neurons = set(np.where(mnist_img.flatten() > np.random.rand(28 ** 2))[0])
        # propagate the stimulus
        for i in stimulated_neurons:
            for j in range(propagation_depth):
                if connections_matrix[i][j] > np.random.rand():
                    connections_propagated += 1
                else:
                    break
        return connections_propagated

    def stats(self):
        return {'strength': self.connections_matrix.mean()}


if __name__ == "__main__":
    pass


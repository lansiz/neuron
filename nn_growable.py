# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
# import random
# import strengthen_functions


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class NeuralNetwork(object):
    def __init__(self, strength_function=None, strength_mean=.5, strength_std=.3, transmission_history_len=10**4, image_scale=8, weight_sum=100):
        self.sensors_number = image_scale ** 2

        self.connections_matrix = np.random.normal(strength_mean, strength_std, self.sensors_number)
        self.connections_matrix = np.where(self.connections_matrix <= 1, self.connections_matrix, 1)
        self.connections_matrix = np.where(self.connections_matrix >= 0, self.connections_matrix, 0)

        self.strength_function = strength_function
        self.strengthen_rate = 1. / transmission_history_len
        # each synapse has a recording queue for its transmission in the tempral order. value 1 represents the occurance of transmission
        self.transmission_history_len = transmission_history_len
        self.transmission_history_pointer = 0
        self.transmission_history_m = self.connections_matrix.copy().astype(object)
        for i in range(self.sensors_number):
                self.transmission_history_m[i] = Queue_(self.transmission_history_len)
        # interations already done.
        self.I = 0

        print('#sensors: %s PF: %s' % ( self.sensors_number, self.strength_function))

    def propagate_once(self, mnist_img, gray_max=255.):
        self.I += 1
        mnist_img = mnist_img / gray_max
        stimulated_neurons = set(np.where(mnist_img.flatten() > np.random.rand(mnist_img.shape[0] ** 2))[0])

        # init the transmission history slots to 0
        for i in range(self.sensors_number):
            self.transmission_history_m[i].records[self.transmission_history_pointer] = 0

        # propagate the stimulus
        for i in stimulated_neurons:
            if self.connections_matrix[i] > np.random.rand():
                self.transmission_history_m[i].records[self.transmission_history_pointer] = 1

        # strengthen the connections
        if self.I > self.transmission_history_len:
            for i in range(self.sensors_number):
                frequency = self.transmission_history_m[i].records.sum() / float(self.transmission_history_len)
                target_strength = self.strength_function(frequency)
                current_strength = self.connections_matrix[i]
                if target_strength > current_strength:
                    self.connections_matrix[i] = np.min((current_strength + self.strengthen_rate, 1))
                else:
                    self.connections_matrix[i] = np.max((0, current_strength - self.strengthen_rate))

        # moving the pointer
        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len

    @classmethod
    def validate_1(cls, mnist_img, connections_matrix, gray_max=255.):
        connections_propagated = 0
        mnist_img = mnist_img / float(gray_max)
        stimulated_neurons = set(np.where(mnist_img.flatten() > np.random.rand(mnist_img.shape[0] ** 2))[0])
        # propagate the stimulus
        for i in stimulated_neurons:
            if connections_matrix[i] > np.random.rand():
                connections_propagated += 1
        return connections_propagated

    @classmethod
    def validate_2(cls, mnist_img, connections_matrix, gray_max=255., threshhold=.5, weight=3):
        connections_propagated = 0
        mnist_img = mnist_img / float(gray_max)
        stimulated_neurons = set(np.where(mnist_img.flatten() > np.random.rand(mnist_img.shape[0] ** 2))[0])
        # propagate the stimulus
        for i in stimulated_neurons:
            if connections_matrix[i] > threshhold:
                # connections_propagated += connections_matrix[i] * weight
                connections_propagated += 1 * weight
        return connections_propagated

    @classmethod
    def validate_3(cls, mnist_img, connections_matrix, gray_max=255., threshhold=.5, weight=3):
        connections_propagated = 0
        mnist_img = mnist_img / float(gray_max)
        stimulated_neurons = set(np.where(mnist_img.flatten() > np.random.rand(mnist_img.shape[0] ** 2))[0])
        # propagate the stimulus
        for i in stimulated_neurons:
            if connections_matrix[i] > np.random.rand():
                # connections_propagated += connections_matrix[i] * weight
                connections_propagated += 1 * weight
        return connections_propagated

    def stats(self):
        return {'strength': self.connections_matrix.mean()}


if __name__ == "__main__":
    pass


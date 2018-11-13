# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import random
import strengthen_functions


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class NeuralNetwork(object):
    def __init__(self, memory_neurons_number=1000, connections_per_memory_neuron=5, transmission_history_len=10**3, image_scale=28, connections_per_sensor=1):
        self.sensor_neurons_number = image_scale ** 2
        self.memory_neurons_number = memory_neurons_number
        self.neurons_number = self.sensor_neurons_number + self.memory_neurons_number
        self.connections_matrix = np.array([0] * (self.neurons_number ** 2), dtype=np.int8).reshape((self.neurons_number, self.neurons_number))
        # the connections matrix has four quarters
        # quarter 1: 28**2 * 1000, connections from sensor to memory neurons, 28 ** 2 * connections_per_sensor_neuron connections
        self.sensor_to_memory_connections = self.sensor_neurons_number * connections_per_sensor
        cnt = 0
        for i in range(self.sensor_neurons_number):
            js = random.sample(range(self.sensor_neurons_number, self.neurons_number), connections_per_sensor)
            for j in js:
                self.connections_matrix[i][j] = 1
        # quarter 2: 28**2 * 28**2, no connections among sensor neurons
        # quarter 3: 1000 * 28**2, no connections from memory to sensor neurons
        # quarter 4: 1000 * 1000, connections among memory neurons, their number is subject to connection_ratio
        self.memory_connections_number = self.memory_neurons_number * connections_per_memory_neuron
        cnt = 0
        while True:
            i = np.random.choice(range(self.sensor_neurons_number, self.neurons_number))
            j = np.random.choice(range(self.sensor_neurons_number, self.neurons_number))
            # get rid of self-to-self connections and mutual connections
            if self.connections_matrix[i][j] or self.connections_matrix[j][i] or (i == j):
                continue
            self.connections_matrix[i][j] = 1
            cnt += 1
            if cnt == self.memory_connections_number:
                break
        # each synapse has a recording queue for its transmission in the tempral order.
        # value 1 represents the occurance of transmission
        self.transmission_history_len = transmission_history_len
        self.transmission_history_pointer = 0
        self.transmission_history_m = self.connections_matrix.copy().astype(object)
        for i in range(self.neurons_number):
            for j in range(self.neurons_number):
                if self.connections_matrix[i][j] == 1:
                    self.transmission_history_m[i][j] = Queue_(self.transmission_history_len)
                else:
                    self.transmission_history_m[i][j] = None
        # interations already done.
        self.I = 0

        print('#neurons: %s #sensors: %s #memory: %s #connections: %s' % (
            self.neurons_number, self.sensor_neurons_number, self.memory_neurons_number,
            self.sensor_to_memory_connections + self.memory_connections_number))

    def initialize_synapses_strength(self, mean=.5, std=.3):
        '''
        initialized N ** 2 synaptic stength accroding to the gene's connections matrix
        the strength of no-exisiting synapse are set to NaN.
        this method should be reimplemented in sub-class.
        '''
        tmp_m = np.random.normal(mean, std, (self.neurons_number, self.neurons_number))
        tmp_m = np.where(tmp_m <= 1, tmp_m, 1)
        tmp_m = np.where(tmp_m >= 0, tmp_m, 0)
        self.connection_strength_m = np.where(self.connections_matrix == 1, tmp_m, -1)
        # self.connection_strength_m_origin = self.connection_strength_m.copy()

    def set_strengthen_functions(self, pf=strengthen_functions.PF34):
        '''
        initialized N ** 2 strengthen functions accroding to the gene's connections matrix
        the functions of no-exisiting synapse are set to NaN.
        this method should be reimplemented in sub-class.
        '''
        self.strengthen_functions_m = np.where(self.connections_matrix == 1, pf, '----')

    def propagate_once(self, stimulated_neurons, strengthen_rate=0.001, debug=False):
        '''
        This is the core of things.
        propagate all stimulus in a pool at a once.
        and update connections strength according to their transmission frequency.
        '''
        self.I += 1
        neurons_fired = stimulated_neurons
        neurons_newly_propagated = neurons_fired
        # init the transmission history slots to 0
        for i in range(self.neurons_number):
            for j in range(self.neurons_number):
                if self.transmission_history_m[i][j]:
                    self.transmission_history_m[i][j].records[self.transmission_history_pointer] = 0
        # propagate the stimulus
        while len(neurons_newly_propagated):
            neurons_propagated = set([])
            for i in neurons_newly_propagated:
                for j, strength in enumerate(self.connection_strength_m[i]):
                    # strength entry is either -1 (non-existence) or between [0, 1]
                    if strength < 0:
                        continue
                    r = np.random.rand()
                    if strength > r:
                        # add the newly propagated neuron
                        neurons_propagated.add(j)
                        # all propagated connections are recorded
                        self.transmission_history_m[i][j].records[self.transmission_history_pointer] = 1
            neurons_newly_propagated = neurons_propagated - neurons_fired
            neurons_fired = neurons_fired.union(neurons_propagated)
        # strengthen the connections
        if self.I > self.transmission_history_len:  # start synapse strengthening after history records are filled
            for i in range(self.neurons_number):
                for j in range(self.neurons_number):
                    # check if the connection exists according to the gene
                    if not self.connections_matrix[i][j]:
                        continue
                    # compute the to-be strength given the frequency by the transimission history
                    frequency = self.transmission_history_m[i][j].records.sum() / float(self.transmission_history_len)
                    strengthen_func = self.strengthen_functions_m[i][j]
                    strength_to_be = strengthen_func(frequency)
                    current_strength = self.connection_strength_m[i][j]

                    # if the to-be strength > crurent one, the strength should be increased by a little bit.
                    # otherwise, descrease it by a little bit.
                    if strength_to_be > current_strength:
                        tmp_strength = current_strength + strengthen_rate
                        if tmp_strength > 1:
                            self.connection_strength_m[i][j] = 1
                        else:
                            self.connection_strength_m[i][j] = tmp_strength
                    else:
                        tmp_strength = current_strength - strengthen_rate
                        if tmp_strength < 0:
                            self.connection_strength_m[i][j] = 0
                        else:
                            self.connection_strength_m[i][j] = tmp_strength

        '''
        if debug:
            print('interation %s' % self.I)
            # print(self.connection_strength_m.round(4))
        '''
        # moving the pointer
        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len

    @classmethod
    def validate(cls, img, connection_strength_m, gray_max=16.):
        '''
        test a number image in NN and collect strength of connections triggered.
        in the meantime, connections' strength stay fixed.
        '''
        count = 0
        # connections_l = []
        # connections_strength_l = []
        img = img / float(gray_max)
        neurons_fired = set(np.where(img.flatten() > np.random.rand(img.shape[0] ** 2))[0])
        neurons_newly_propagated = neurons_fired
        # propagate the stimulus
        while len(neurons_newly_propagated):
            neurons_propagated = set([])
            for i in neurons_newly_propagated:
                for j, strength in enumerate(connection_strength_m[i]):
                    # strength entry is either -1 (non-existence) or between [0, 1]
                    if strength < 0:
                        continue
                    r = np.random.rand()
                    if strength > r:
                        # add the newly propagated neuron
                        neurons_propagated.add(j)
                        # records the triggered connnections' strength
                        # connections_l.append((i, j))
                        # connections_strength_l.append(strength)
                        count += 1
            neurons_newly_propagated = neurons_propagated - neurons_fired
            neurons_fired = neurons_fired.union(neurons_propagated)
        # return connections_l, connections_strength_l
        return count

    def stats(self):
        connection_strength_m = np.where(self.connection_strength_m >= 0, self.connection_strength_m, 0)
        return {
            # 'accuracy': self.accuracy,
            # 'strength_matrix': self.connection_strength_m,
            'strength': connection_strength_m.mean()}


if __name__ == "__main__":
    pass


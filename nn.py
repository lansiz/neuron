# !/usr/bin/env python
#  -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import strengthen_functions


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class NeuralNetwork(object):
    def __init__(self, connection_matrix, transmission_history_len=10**4):
        if connection_matrix.shape[0] != connection_matrix.shape[1]:
            print('connection matrix should be squared')
            return None
        self.connection_matrix = connection_matrix
        self.N = self.connection_matrix.shape[0]
        self.connections_number = self.connection_matrix.sum()
        # print(N)
        # each synapse has a recording queue for its transmission in the tempral order.
        # value 1 represents the occurance of transmission
        self.transmission_history_len = transmission_history_len
        self.transmission_history_pointer = 0
        self.transmission_history_m = self.connection_matrix.copy().astype(object)
        for i in range(self.N):
            for j in range(self.N):
                if self.connection_matrix[i][j] == 1:
                    self.transmission_history_m[i][j] = Queue_(
                        self.transmission_history_len)
                else:
                    self.transmission_history_m[i][j] = None
        # interations already done.
        self.I = 0

    def initialize_synapses_strength(self, mean=.5, std=.3):
        '''
        initialized N ** 2 synaptic stength
        the strength of no-exisiting synapse are set to NaN.
        this method should be reimplemented in sub-class.
        '''
        tmp_m = np.random.normal(mean, std, (self.N, self.N))
        tmp_m = np.where(tmp_m <= 1, tmp_m, 1)
        tmp_m = np.where(tmp_m >= 0, tmp_m, 0)
        self.connection_strength_m = np.where(
            self.connection_matrix == 1, tmp_m, -1)
        # self.connection_strength_m_origin = self.connection_strength_m.copy()

    def initialize_synapses_strength_uniform(self):
        tmp_m = np.random.normal(0, 1, (self.N, self.N))
        self.connection_strength_m = np.where(
            self.connection_matrix == 1, tmp_m, -1)

    def set_strengthen_functions(self, pf=strengthen_functions.PF34):
        '''
        initialized N ** 2 strengthen functions
        the functions of no-exisiting synapse are set to NaN.
        this method should be reimplemented in sub-class.
        '''
        self.strengthen_functions_m = np.where(
            self.connection_matrix == 1, pf, '----')

    def show_strengthen_functions_matrix(self):
        ''' display strengthen functions in prettier matrix '''
        strengthen_functions_m = self.strengthen_functions_m.copy()
        for i in range(self.N):
            for j in range(self.N):
                val = self.strengthen_functions_m[i][j]
                if val != '----':
                    strengthen_functions_m[i][j] = val.func_name
        print(strengthen_functions_m)

    def propagate_once(self, neurons_stimulated):
        '''
        This is the core of things.
        propagate all stimulus in a pool at a once.
        and update connections strength according to their transmission frequency.
        '''
        self.I += 1
        strengthen_rate = 1. / self.transmission_history_len
        neurons_fired = neurons_stimulated
        neurons_newly_propagated = neurons_fired
        # init the transmission history slots to 0
        for i in range(self.N):
            for j in range(self.N):
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
            for i in range(self.N):
                for j in range(self.N):
                    # check if the connection exists
                    if not self.connection_matrix[i][j]:
                        continue
                    # compute the to-be strength given the frequency by the transimission history
                    frequency = self.transmission_history_m[i][j].records.sum(
                    ) / float(self.transmission_history_len)
                    target_strength = self.strengthen_functions_m[i][j](
                        frequency)
                    current_strength = self.connection_strength_m[i][j]

                    # if the to-be strength > crurent one, the strength should be increased by a little bit.
                    # otherwise, descrease it by a little bit.
                    if target_strength > current_strength:
                        self.connection_strength_m[i][j] = np.min(
                            (current_strength + strengthen_rate, 1))
                    else:
                        self.connection_strength_m[i][j] = np.max(
                            (0, current_strength - strengthen_rate))

        # forward the pointer
        self.transmission_history_pointer = (
            self.transmission_history_pointer + 1) % self.transmission_history_len

    def get_transmission_frequency(self):
        '''
        reduce the history record of connections transmission to frequency scalar.
        '''
        transmission_frequency = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.transmission_history_m[i][j]:
                    transmission_frequency[i][j] = self.transmission_history_m[i][j].records.sum(
                    ) / float(self.transmission_history_len)
        return transmission_frequency.round(4)

    def stats(self):
        connection_strength_m = np.where(
            self.connection_strength_m >= 0, self.connection_strength_m, 0)
        return {
            'strength_matrix': self.connection_strength_m,
            'strength': connection_strength_m.sum() / self.connections_number}

    def stats_square_mean(self):
        connection_strength_m = np.where(
            self.connection_strength_m >= 0, self.connection_strength_m, 0)
        return {
            'strength_matrix': self.connection_strength_m,
            'strength': np.square(connection_strength_m).sum() / self.connections_number}

    def propagate_test(self, neurons_stimulated):
        neurons_fired = neurons_stimulated
        neurons_newly_propagated = neurons_fired
        connections_propagated = 0
        while len(neurons_newly_propagated):
            neurons_propagated = set([])
            for i in neurons_newly_propagated:
                for j, strength in enumerate(self.connection_strength_m[i]):
                    if strength < 0:
                        continue
                    r = np.random.rand()
                    if strength > r:
                        neurons_propagated.add(j)
                        connections_propagated += 1
            neurons_newly_propagated = neurons_propagated - neurons_fired
            neurons_fired = neurons_fired.union(neurons_propagated)
        return connections_propagated


if __name__ == "__main__":
    pass

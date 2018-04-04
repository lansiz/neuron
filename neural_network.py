# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import strengthen_functions


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class NeuralNetwork(object):
    def __init__(self, gene, transmission_history_len=10**4, score_history_len=10**4):
        self.gene = gene
        self.N = self.gene.connections.shape[0]
        self.connections_number = self.gene.connections.sum()
        # print(N)
        # each synapse has a recording queue for its transmission in the tempral order.
        # value 1 represents the occurance of transmission
        self.transmission_history_len = transmission_history_len
        self.transmission_history_pointer = 0
        self.transmission_history_m = self.gene.connections.copy().astype(object)
        for i in range(self.N):
            for j in range(self.N):
                if self.gene.connections[i][j] == 1:
                    self.transmission_history_m[i][j] = Queue_(self.transmission_history_len)
                else:
                    self.transmission_history_m[i][j] = None
        # each NN has a recording queue regarding its accuray to do the right propagations
        # in the tempral order. value 1 represents right propagtation
        # self.score_history_len = score_history_len
        # self.score_history = Queue_(self.score_history_len)
        # self.score_history_pointer = 0
        # interations already done.
        self.I = 0
        # accuracy stats
        self.accuracy = 0

    def initialize_synapses_strength(self, mean=.5, std=.3):
        '''
        initialized N ** 2 synaptic stength accroding to the gene's connections matrix
        the strength of no-exisiting synapse are set to NaN.
        this method should be reimplemented in sub-class.
        '''
        tmp_m = np.random.normal(mean, std, (self.N, self.N))
        tmp_m = np.where(tmp_m <= 1, tmp_m, 1)
        tmp_m = np.where(tmp_m >= 0, tmp_m, 0)
        self.connection_strength_m = np.where(self.gene.connections == 1, tmp_m, -1)
        self.connection_strength_m_origin = self.connection_strength_m.copy()

    def set_strengthen_functions(self):
        '''
        initialized N ** 2 strengthen functions accroding to the gene's connections matrix
        the functions of no-exisiting synapse are set to NaN.
        this method should be reimplemented in sub-class.
        '''
        self.strengthen_functions_m = np.where(self.gene.connections == 1, strengthen_functions.PF32, '----')

    def show_strengthen_functions_matrix(self):
        ''' display strengthen functions in prettier matrix '''
        strengthen_functions_m = self.strengthen_functions_m.copy()
        for i in range(self.N):
            for j in range(self.N):
                val = self.strengthen_functions_m[i][j]
                if val != '----':
                    strengthen_functions_m[i][j] = val.func_name
        print(strengthen_functions_m)

    def propagate_once(self, stimu_pool, strengthen_rate=0.00005, debug=False):
        '''
        This is the core of things.
        propagate all stimulus in a pool at a once.
        and update connections strength according to their transmission frequency.
        '''
        self.I += 1
        neurons_fired = set([])
        for X, Y_l in stimu_pool.data:
            neurons_fired = neurons_fired.union(X)
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
                    if strength > np.random.rand():
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
                    # check if the connection exists according to the gene
                    if not self.gene.connections[i][j]:
                        continue
                    # compute the to-be strength given the frequency by the transimission history
                    frequency = \
                        self.transmission_history_m[i][j].records.sum() / float(self.transmission_history_len)
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

        if debug:
            print('interation %s' % self.I)
            print(self.connection_strength_m.round(4))
        # moving the pointer
        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len

    def evaluate_accuracy(self, stimu_pool, trys_per_stimulus=10**4):
        '''
        test each stimlus in the pool form try_per_stimulus times and compute the accuracy
        '''
        trys = .0
        scores = .0
        for X, Y_l in stimu_pool.data:
            for _ in range(trys_per_stimulus):
                trys += 1
                neurons_fired = X
                neurons_newly_propagated = X
                while len(neurons_newly_propagated):
                    neurons_propagated = set([])
                    for i in neurons_newly_propagated:
                        for j, strength in enumerate(self.connection_strength_m[i]):
                            if strength > np.random.rand():
                                neurons_propagated.add(j)
                    neurons_newly_propagated = neurons_propagated - neurons_fired
                    neurons_fired = neurons_fired.union(neurons_propagated)
                if neurons_fired in Y_l:
                    scores += 1
        self.accuracy = scores / trys 

    def get_transmission_frequency(self):
        '''
        reduce the history record of connections transmission to frequency scalar.
        '''
        transmission_frequency = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.transmission_history_m[i][j]:
                    transmission_frequency[i][j] = self.transmission_history_m[i][j].records.sum() / float(self.transmission_history_len)
        return transmission_frequency.round(4)

    def stats(self):
        # connection_strength_m = np.where(self.connection_strength_m >= 0, self.connection_strength_m, 0)
        return {
            'accuracy': self.accuracy,
            'strength_matrix': self.connection_strength_m}
            # 'strength': connection_strength_m.sum() / self.connections_number}


if __name__ == "__main__":
    pass


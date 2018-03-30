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
        # print(N)
        # each synapse has a recording queue for its transmission in the tempral order.
        # value 1 represents the occurance of transmission
        self.transmission_history_len = transmission_history_len
        self.transmission_history_m = np.array(
            [Queue_(self.transmission_history_len) for i in range(self.N ** 2)]).reshape((self.N, self.N))
        self.transmission_history_pointer = 0
        # each NN has a recording queue regarding its accuray to do the right propagations
        # in the tempral order. value 1 represents right propagtation
        self.score_history_len = score_history_len
        self.score_history = Queue_(self.score_history_len)
        self.score_history_pointer = 0

    def initialize_synapses_strength(self):
        # initialized N ** 2 synaptic stength accroding to the gene's connections matrix
        # the strength of no-exisiting synapse are set to NaN.
        # this method should be reimplemented in sub-class.
        self.connection_strength_m = np.where(self.gene.connections == 1, np.random.rand(), -1)

    def set_strengthen_functions(self):
        # initialized N ** 2 strengthen functions accroding to the gene's connections matrix
        # the functions of no-exisiting synapse are set to NaN.
        # this method should be reimplemented in sub-class.
        self.strengthen_functions_m = np.where(self.gene.connections == 1, strengthen_functions.PF29, np.NaN)

    def propagate_one_stimulus(self, stimulus, strengthen_rate=0.00005, debug=False):
        X = stimulus[0]  # set of neurons, where evn stimulation are on and propagation are start from
        Y_l = stimulus[1]  # list of set of neurons for multiple targets (propagations)
        neurons_fired = X
        neurons_newly_propagated = X
        # init the transmission history slots to 0
        for i in range(self.N):
            for j in range(self.N):
                self.transmission_history_m[i][j].records[self.transmission_history_pointer] = 1
        while len(neurons_newly_propagated):
            neurons_newly_propagated = set([])
            for i in neurons_newly_propagated:
                for j, strength in enumerate(self.connection_strength_m[i]):
                    # strength entry is either -1 (non-existence) or between [0, 1]
                    if strength > np.random.rand():
                        # add the newly propagated neuron
                        neurons_newly_propagated.add(j)
                        # all propagated connections are recorded
                        self.transmission_history_m[i][j].records[self.transmission_history_pointer] = 1
            neurons_fired = neurons_fired.union(neurons_newly_propagated)
        # record the score
        Y_hat = neurons_fired  # the 'estimated' propagation target
        if Y_hat in Y_l:
            self.score_history.records[self.score_history_pointer] = 1
        else:
            self.score_history.records[self.score_history_pointer] = 0
        # strengthen the connections
        for i in range(self.N):
            for j in range(self.N):
                if not self.gene.connections[i][j]:
                    continue
                frequency = \
                    self.transmission_history_m[i][j].records.sum() / float(self.transmission_history_len)
                strengthen_func = self.strengthen_functions_m[i][j]
                strength_to_be = strengthen_func(frequency)
                current_strength = self.connection_strength_m[i][j]

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

        # moving the pointer
        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len
        self.score_history_pointer = (self.score_history_pointer + 1) % self.score_history_len

    def stats(self):
        return {
            # 'connections_strengh': self.connection_strength_m, 'accuracy': np.random.rand(), 'other': 3}
            'accuracy': np.random.rand(),
            'strength': self.connection_strength_m.sum()}


if __name__ == "__main__":
    from gene import Gene
    from stimulus import StimuliPool
    import matplotlib.pyplot as plt
    N = 20
    S = 5
    stimu_pool = StimuliPool(N, S)
    g = Gene(N)
    nn = NeuralNetwork(g)
    nn.initialize_synapses_strength()
    nn.set_strengthen_functions()
    # print(nn.gene.connections)
    # print(nn.strengthen_functions_m)
    # print(stimu_pool.pick_one())
    stats = []
    for _ in range(5 * 10 ** 3):
        nn.propagate_one_stimulus(stimu_pool.pick_one())
        # print(nn.stats())
        stats.append(nn.stats()['accuracy'])
    print(stats)
    plt.plot(stats)

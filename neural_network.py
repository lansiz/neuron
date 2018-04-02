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
        self.transmission_history_m = np.array(
            [Queue_(self.transmission_history_len) for i in range(self.N ** 2)]).reshape((self.N, self.N))
        self.transmission_history_pointer = 0
        # each NN has a recording queue regarding its accuray to do the right propagations
        # in the tempral order. value 1 represents right propagtation
        self.score_history_len = score_history_len
        self.score_history = Queue_(self.score_history_len)
        self.score_history_pointer = 0
        # interations
        self.I = 0

    def initialize_synapses_strength(self):
        # initialized N ** 2 synaptic stength accroding to the gene's connections matrix
        # the strength of no-exisiting synapse are set to NaN.
        # this method should be reimplemented in sub-class.
        tmp_m = np.random.rand(self.N, self.N)
        self.connection_strength_m = np.where(self.gene.connections == 1, tmp_m, -1)
        self.connection_strength_m_origin = self.connection_strength_m.copy()

    def set_strengthen_functions(self):
        # initialized N ** 2 strengthen functions accroding to the gene's connections matrix
        # the functions of no-exisiting synapse are set to NaN.
        # this method should be reimplemented in sub-class.
        self.strengthen_functions_m = np.where(self.gene.connections == 1, strengthen_functions.PF31, np.NaN)

    def propagate_one_stimulus(self, stimulus, strengthen_rate=0.00005, debug=False):
        self.I += 1
        X = stimulus[0]  # set of neurons, where evn stimulation are on and propagation are start from
        Y_l = stimulus[1]  # list of set of neurons for multiple targets (propagations)
        neurons_fired = X
        neurons_newly_propagated = X
        # init the transmission history slots to 0
        for i in range(self.N):
            for j in range(self.N):
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
            neurons_fired = neurons_fired.union(neurons_propagated)
            neurons_newly_propagated = neurons_propagated
        # record the score
        Y_hat = neurons_fired  # the 'estimated' propagation target
        if Y_hat in Y_l:
            self.score_history.records[self.score_history_pointer] = 1
        else:
            self.score_history.records[self.score_history_pointer] = 0
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
        self.score_history_pointer = (self.score_history_pointer + 1) % self.score_history_len

    def stats(self):
        connection_strength_m = np.where(self.connection_strength_m >= 0, self.connection_strength_m, 0)
        return {
            # 'connections_strengh': self.connection_strength_m, 'accuracy': np.random.rand(), 'other': 3}
            'accuracy': self.score_history.records.sum() / float(self.score_history_len),
            'strength': connection_strength_m.sum() / self.connections_number}


if __name__ == "__main__":
    from gene import Gene
    from stimulus import StimuliPool
    import matplotlib as mpl
    mpl.use('Agg', warn=False)
    import matplotlib.pyplot as plt
    N = 5
    S = 5
    stimu_pool = StimuliPool(N, S)
    g = Gene(N, .8)
    print(' gene '.center(100, '-'))
    g.info()
    nn = NeuralNetwork(g)
    nn.initialize_synapses_strength()
    nn.set_strengthen_functions()
    print(' strength matrix to start '.center(100, '-'))
    print(nn.connection_strength_m_origin)
    print(' stimuli pool '.center(100, '-'))
    stimu_pool.info()
    print(' strengthen functions matrix '.center(100, '-'))
    print(nn.strengthen_functions_m)
    stats = []
    for _ in range(100000):
        nn.propagate_one_stimulus(stimu_pool.pick_one(), strengthen_rate=0.00005)
        # print(nn.stats())
        # stats.append(nn.stats()['accuracy'])
        stats.append(nn.stats()['strength'])
    # print(stats)
    print(' final strength matrix '.center(100, '-'))
    print(nn.connection_strength_m)
    plt.plot(stats)
    plt.savefig('./nn.png')

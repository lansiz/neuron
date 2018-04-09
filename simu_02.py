# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import strengthen_functions
from neural_network import NeuralNetwork

class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class NeuralNetworkLockedStrength(NeuralNetwork):
    def __init__(self, gene, connection_to_watch):
        NeuralNetwork.__init__(self, gene)
        self.connection_to_watch = connection_to_watch

    def propagate_once(self, stimu_pool, strengthen_rate=0.00005, debug=False):
        '''
        update a connections strength while others remain fixed
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
            # compute the to-be strength given the frequency by the transimission history
            i, j = self.connection_to_watch
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
            pass
        # moving the pointer
        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len

    def stats(self):
        i, j = self.connection_to_watch
        return { 'strength': self.connection_strength_m[i][j] }


if __name__ == "__main__":
    import numpy as np
    from neural_network import NeuralNetwork
    from gene import Gene
    from stimulus import StimuliPool
    import matplotlib as mpl
    mpl.use('Agg', warn=False)
    import matplotlib.pyplot as plt
    N = 7
    S = 2
    stimu_pool = StimuliPool(N, S)
    g = Gene(N, .5)
    stimu_pool.data = [[set([1]), [set([1,2])]]]
    g.connections = np.array([
            [0,0,0,0,0,0,1],
            [1,0,1,0,1,0,0],
            [1,0,0,0,0,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,0,0,1],
            [0,1,0,0,0,0,0]])
    print(' gene (connection matrix) '.center(100, '-'))
    g.info()
    nn = NeuralNetworkLockedStrength(g, (1, 4))
    nn.initialize_synapses_strength(.5, .1)
    nn.set_strengthen_functions()
    print(' strength matrix to start '.center(100, '-'))
    print(nn.connection_strength_m_origin.round(4))
    print(' stimuli pool '.center(100, '-'))
    stimu_pool.info()
    print(' strengthen functions matrix '.center(100, '-'))
    # print(nn.strengthen_functions_m)
    nn.show_strengthen_functions_matrix()
    strength_stats = []
    for _ in range(300000):
        nn.propagate_once(stimu_pool, strengthen_rate=0.00005)
        strength_stats.append(nn.stats()['strength'])
        if _ % 1000 == 0:
            nn.evaluate_accuracy(stimu_pool)
            strength_stats.append(nn.stats()['strength'])
    print(' strength matrix at fixed point '.center(100, '-'))
    print(nn.connection_strength_m.round(4))
    print(' transmission frequency at fixed point '.center(100, '-'))
    print(nn.get_transmission_frequency())
    plt.plot(strength_stats)
    plt.savefig('./simu_02.png')

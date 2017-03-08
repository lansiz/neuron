# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np


class Brain(object):
    def __init__(self, cnt):
        self.neurons_cnt = cnt
        self.connections = None
        self.stimuli = None
        self.neurons_fire_counts = np.zeros(self.neurons_cnt)

    def get_neurons_cnt(self):
        return self.neurons_cnt

    def fire(self, rounds, plasticity_begin=100000):
        if (self.connections is None) or (self.stimuli is None):
            print 'This brains has no connections or stimuli'
            return
        neurons_cnt = self.neurons_cnt
        rnd = 0
        for _ in range(int(rounds)):
            rnd += 1
            neurons_fired = np.array([False] * neurons_cnt, dtype=np.bool)  # new round begins, no neurons fired
            neurons_stimulated = np.random.rand(neurons_cnt) < self.stimuli  # decide what neurons are stimulated from environment
            if neurons_stimulated.sum():  # no neurons are stimulated, skip this round
                # propagate action potential down through the neural tree
                while True:
                    # print 'neurons_stimulated', neurons_stimulated
                    neurons_to_fire = (~ neurons_fired) & neurons_stimulated
                    # print 'neurons_to_fire', neurons_to_fire
                    if not neurons_to_fire.sum():  # no neurons are left to fire, dead end
                        break
                    self.neurons_fire_counts += neurons_to_fire
                    # build downstream stimulated neurons for next loop
                    neurons_stimulated = np.array([False] * neurons_cnt, dtype=np.bool)
                    connections_to_propagate = self.connections.propagation_decision()
                    # print connections_to_propagate
                    for i, row in enumerate(connections_to_propagate):
                        for j, propag in enumerate(row):
                            # connection failt to propagate; propagate to itself; not the stimulated
                            if (not propag) or (i == j) or (not neurons_to_fire[i]):
                                self.connections.propagate(i, j, False)
                                continue
                            # connections_transmission_cnt[i][j] += 1
                            self.connections.propagate(i, j, True)  # fire together, wired together
                            if neurons_fired[j] or neurons_to_fire[j]:  # if neuron j is already fired, skip in case recount and circle with parent neuron
                                continue
                            neurons_stimulated[j] = True  # neuron j will be stimulated if any upstream neurons propagate to it
                            # connections_transmission_cnt[i][j] += 1
                    neurons_fired |= neurons_to_fire
            # plasticity of neural connections
            # if debug: print connections_transmission_rates
            if rnd > plasticity_begin:
                self.connections.plasticity()
            self.connections.move_pointer()
            self.connections.collect_data()
        self.connections.stats()

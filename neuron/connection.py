# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import plasticity
import copy


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([False] * qlen, dtype=bool)

    def count_(self):
        return float(self.records.sum())

    @classmethod
    def create_queue(cls, cnt, qlen):
        queues = []
        for i in range(cnt):
            queues.append(Queue_(qlen))
        return queues


class Connections(object):
    def __init__(self, brain, ratio, qlen, step):
        self.neurons_cnt = neurons_cnt = brain.neurons_cnt
        self.connections_cnt = neurons_cnt ** 2
        self.brain = brain
        self.ratio = ratio
        self.step = step
        self.qlen = qlen
        self.data_trim = 20000
        queues = Queue_.create_queue(self.connections_cnt, self.qlen)
        self.queues_pointer = 0
        self.propagation_history = np.array(queues).reshape((neurons_cnt, neurons_cnt))
        self.propagation_rate = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        # all connections initialized with zero transmission probability
        self.transmission_probs_origin = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        self.transmission_probs_by_funcs = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        # all connections initialized with disconnected functions
        self.plasticity_funcs = np.array([plasticity.PF00] * self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        # sample [self.connections_cnt * ratio] connections (i -> j), excluding i -> i and j -> i
        cnt_temp = int(self.connections_cnt * self.ratio)
        self.connnections_to_sample = np.min([cnt_temp, int((self.connections_cnt - neurons_cnt) / 2)])
        i_range = j_range = range(neurons_cnt)
        connections_sampled = []
        i_ = self.connnections_to_sample
        while i_ > 0:
            i = np.random.choice(i_range)
            j = np.random.choice(j_range)
            if ((i, j) in connections_sampled) or ((j, i) in connections_sampled) or (i == j):
                continue
            connections_sampled.append((i, j))
            i_ -= 1
        # initialize data according to the sampled connections
        for i, j in connections_sampled:
            self.transmission_probs_origin[i][j] = np.random.rand()
            self.plasticity_funcs[i][j] = plasticity.funcs_sample_one()
        self.transmission_probs = self.transmission_probs_origin.copy()
        self.settings()
        # data collecting
        self.data1 = []  # propagation_rate
        self.data2 = []  # transmission_probs_by_funcs
        self.data3 = []  # transmission_probs

    def reset(self):
        neurons_cnt = self.neurons_cnt
        queues = Queue_.create_queue(self.connections_cnt, self.qlen)
        self.propagation_history = np.array(queues).reshape((neurons_cnt, neurons_cnt))
        self.queues_pointer = 0
        self.propagation_rate = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        self.transmission_probs = self.transmission_probs_origin.copy()
        self.transmission_probs_by_funcs = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        self.data1 = []
        self.data2 = []
        self.data3 = []

    def copy_to(self, brain):
        neurons_cnt = self.neurons_cnt
        if not (neurons_cnt == brain.neurons_cnt):
            print 'ERROR: connections and brain are not in the same size'
            return None
        if brain.connections is not None:
            print 'ERROR: the brain already has connections'
            return None
        new_conns = copy.copy(self)
        queues = Queue_.create_queue(self.connections_cnt, self.qlen)
        new_conns.propagation_history = np.array(queues).reshape((neurons_cnt, neurons_cnt))
        new_conns.queues_pointer = 0
        new_conns.propagation_rate = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        new_conns.transmission_probs = self.transmission_probs_origin.copy()
        new_conns.transmission_probs_by_funcs = np.zeros(self.connections_cnt).reshape((neurons_cnt, neurons_cnt))
        new_conns.data1 = []
        new_conns.data2 = []
        new_conns.data3 = []
        return new_conns

    def settings(self):
        print 'neurons: %s ratio: %s connections: %s step: %s qlen: %s' % (
            self.neurons_cnt, self.ratio, self.connnections_to_sample, self.step, self.qlen)
        print 'transmission probabilities:'
        print self.transmission_probs.round(2)
        print 'plasticity functions:'
        print plasticity.print_name(self.plasticity_funcs)

    def propagate(self, i, j, value):
        self.propagation_history[i][j].records[self.queues_pointer] = value
        rate = self.propagation_history[i][j].count_() / self.qlen
        self.propagation_rate[i][j] = rate
        plasticity_func = self.plasticity_funcs[i][j]
        prob_by_func = plasticity_func(rate)
        self.transmission_probs_by_funcs[i][j] = prob_by_func

    def move_pointer(self):
        self.queues_pointer = (self.queues_pointer + 1) % self.qlen

    def plasticity(self):
        for i, row in enumerate(self.transmission_probs):
            for j, old_prob in enumerate(row):
                if self.transmission_probs_by_funcs[i][j] > old_prob:
                    tmp_prob = old_prob + self.step
                    if tmp_prob > 1:
                        self.transmission_probs[i][j] = 1
                    else:
                        self.transmission_probs[i][j] = tmp_prob
                else:
                    tmp_prob = old_prob - self.step
                    if tmp_prob < 0:
                        self.transmission_probs[i][j] = 0
                    else:
                        self.transmission_probs[i][j] = tmp_prob

    def propagation_decision(self):
        return np.random.rand(self.connections_cnt).reshape((self.neurons_cnt, self.neurons_cnt)) < self.transmission_probs

    def collect_data(self):
        self.data1.append(self.transmission_probs.mean())
        self.data2.append(self.transmission_probs_by_funcs.mean())
        # self.data3.append(self.propagation_rate.mean())

    def trim_data(self):
        # self.data1.append(self.propagation_rate.mean())
        self.data1 = self.data1[self.data_trim:]
        self.data2 = self.data2[self.data_trim:]
        self.data3 = self.data3[self.data_trim:]

    def plot_data(self):
        alpha = .8
        plt.plot(self.data1, alpha=alpha)
        plt.plot(self.data2, alpha=alpha)
        # plt.plot(self.data3, alpha=alpha)
        plt.grid(True)

    def save_data(self):
        pass

    def stats(self):
        print 'propagation rate:'
        print self.propagation_rate
        print 'transmission probs by functions:'
        print self.transmission_probs_by_funcs.round(2)
        print 'transmission probs:'
        print self.transmission_probs.round(2)

    @classmethod
    def grow_on(cls, brain, ratio=.5, qlen=10000, step=0.00001):
        # let brain know its connections
        conns = Connections(brain, ratio, qlen, step)
        brain.connections = conns
        return conns


if __name__ == "__main__":
    import neuron
    brain = neuron.Brain(5)
    Connections.grow_on(brain)

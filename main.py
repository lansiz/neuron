# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import pickle
import plasticity

# config
rounds_to_run = 20 * 10 ** 4
debug = False
plasticity_begin = 100000
neurons_cnt = 5
prob_step = .00005
connections_percent = .8


class Queue_(object):
    pointer = 0
    history_max = 10000
    queues = []
    rounds = 1

    def __init__(self):
        self.records = np.array([False] * Queue_.history_max, dtype=bool)

    def rate(self):
        # if Queue_.rounds < Queue_.history_max:
        return float(self.records.sum()) / Queue_.history_max

    def propagate(self, value):
        self.records[Queue_.pointer] = value

    @classmethod
    def move_pointer(cls):
        cls.rounds += 1
        cls.pointer = (cls.pointer + 1) % Queue_.history_max

    @classmethod
    def create_queue(cls, cnt):
        Queue_.queues = []
        for i in range(cnt):
            Queue_.queues.append(Queue_())
        return Queue_.queues

    @classmethod
    def display(cls):
        print 'round %s:' % Queue_.rounds,
        for q in Queue_.queues:
            # print q.records.sum(),
            print q.rate(),
        print ''

    @classmethod
    def stats(cls):
        v = 0
        for q in Queue_.queues:
            # print q.records.sum(),
            v += q.rate()
        return v / len(Queue_.queues)


def read_pickle(file_name):
    try:
        f = open(file_name, 'rb')
        o = pickle.load(f)
        f.close()
        return o
    except:
        return None


def write_pickle(o, file_name):
    try:
        f = open(file_name, 'wb')
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return True
    except:
        return False


# ********* general *********
rounds = 0
data1 = []
data2 = []
# ********* neurons *********
neurons_fire_counts = np.zeros(neurons_cnt)
neurons_checked_or_not = np.array([False] * neurons_cnt, dtype=np.bool)
# connections
connections_cnt = neurons_cnt ** 2
connections_transmission_cnt = np.zeros(connections_cnt).reshape((neurons_cnt, neurons_cnt))
connections_transmission_history = np.array(Queue_.create_queue(connections_cnt)).reshape((neurons_cnt, neurons_cnt))
connections_transmission_probs = np.random.rand(connections_cnt).reshape((neurons_cnt, neurons_cnt))
# connections_transmission_probs = np.ones(connections_cnt).reshape((neurons_cnt, neurons_cnt))
for i in range(neurons_cnt):
    connections_transmission_probs[i][i] = 0
print 'initial conn transmission probs:'
print connections_transmission_probs
plasticity_funcs = plasticity.funcs_sample(int(connections_cnt * connections_percent))
plasticity_funcs = plasticity_funcs + [plasticity.PF00] * (connections_cnt - len(plasticity_funcs))
np.random.shuffle(plasticity_funcs)
connections_plasticity_funcs = np.array(plasticity_funcs).reshape((neurons_cnt, neurons_cnt))
for i in range(neurons_cnt):
    connections_plasticity_funcs[i][i] = plasticity.PF00
print connections_plasticity_funcs
# ********* stimuli *********
stimuli_probs = np.random.rand(neurons_cnt)
print 'initial stimuli probs', stimuli_probs

# ********* trainning *********
# for i in range(int(40000000)):
for _ in range(int(rounds_to_run)):
    rounds += 1
    if debug: print 'round ', rounds
    neurons_fired = np.array([False] * neurons_cnt, dtype=np.bool)  # new round begins, no neurons fired
    neurons_stimulated = np.random.rand(neurons_cnt) < stimuli_probs  # decide what neurons are stimulated from environment
    if neurons_stimulated.sum():  # no neurons are stimulated, skip this round
        # propagate action potential down through the neural tree
        while True:
            # print 'neurons_stimulated', neurons_stimulated
            neurons_to_fire = (~ neurons_fired) & neurons_stimulated
            # print 'neurons_to_fire', neurons_to_fire
            if not neurons_to_fire.sum():  # no neurons are left to fire, dead end
                break
            neurons_fire_counts += neurons_to_fire
            # build downstream stimulated neurons for next loop
            neurons_stimulated = np.array([False] * neurons_cnt, dtype=np.bool)
            connections_to_propagate = np.random.rand(connections_cnt).reshape((neurons_cnt, neurons_cnt)) < connections_transmission_probs
            # print connections_to_propagate
            for i, row in enumerate(connections_to_propagate):
                for j, propag in enumerate(row):
                    # connection failt to propagate; propagate to itself; not the stimulated
                    if (not propag) or (i == j) or (not neurons_to_fire[i]):
                        connections_transmission_history[i][j].propagate(False)
                        continue
                    # connections_transmission_cnt[i][j] += 1
                    connections_transmission_history[i][j].propagate(True)  # fire together, wired together
                    if neurons_fired[j] or neurons_to_fire[j]:  # if neuron j is already fired, skip in case recount and circle with parent neuron
                        continue
                    neurons_stimulated[j] = True  # neuron j will be stimulated if any upstream neurons propagate to it
                    # connections_transmission_cnt[i][j] += 1
            neurons_fired |= neurons_to_fire
    # plasticity of neural connections
    if debug: print connections_transmission_probs
    # connections_transmission_rates = connections_transmission_cnt / rounds
    # if debug: print connections_transmission_rates
    if rounds > plasticity_begin:
        for i, row in enumerate(connections_transmission_probs):
            for j, old_prob in enumerate(row):
                plasticity_func = connections_plasticity_funcs[i][j]
                prob_to_be = plasticity_func(connections_transmission_history[i][j].rate())
                if debug: print rounds, i, j, old_prob, prob_to_be
                # if np.abs(prob_to_be - old_prob) < 0.001:
                #    pass
                # elif prob_to_be > old_prob:
                if prob_to_be > old_prob:
                    tmp_prob = old_prob + prob_step
                    if tmp_prob > 1:
                        connections_transmission_probs[i][j] = 1
                    else:
                        connections_transmission_probs[i][j] = tmp_prob
                else:
                    tmp_prob = old_prob - prob_step
                    if tmp_prob < 0:
                        connections_transmission_probs[i][j] = 0
                    else:
                        connections_transmission_probs[i][j] = tmp_prob
    if debug: Queue_.display()
    # data.append(np.sqrt(np.divide(np.sum(np.square(connections_transmission_probs)), connections_cnt)))
    stat1 = np.mean(connections_transmission_probs)
    stat2 = Queue_.stats()
    data1.append(stat1)
    data2.append(stat2)
    # if debug: print stat
    Queue_.move_pointer()

# data = np.array(data)
# write_pickle(data, 'data.pkl')
# write_pickle(connections_transmission_probs, 'conn_probs.pkl')

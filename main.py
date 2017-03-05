# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import pickle

# config
rounds_to_run = 2 * 10 ** 5
debug = False
neurons_cnt = 2
prob_step = .00005
connections_percent = .5

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


# plasticity functions
def plasticity_func_00(x):
	return 0
def plasticity_func_01(x):
	return x

# ********* general *********
rounds = 0
data = []
# ********* neurons *********
neurons_fire_counts = np.zeros(neurons_cnt)
neurons_checked_or_not = np.array([False] * neurons_cnt, dtype=np.bool)
# connections
connections_cnt = neurons_cnt ** 2
connections_transmission_cnt = np.zeros(connections_cnt).reshape((neurons_cnt, neurons_cnt))
connections_transmission_probs = np.random.rand(connections_cnt).reshape((neurons_cnt, neurons_cnt))
for i in range(neurons_cnt):
	connections_transmission_probs[i][i] = 0
print connections_transmission_probs
plasticity_funcs = [plasticity_func_01] * int(connections_cnt * connections_percent)
plasticity_funcs = plasticity_funcs + [plasticity_func_00] * (connections_cnt - len(plasticity_funcs))
np.random.shuffle(plasticity_funcs)
connections_plasticity_funcs = np.array(plasticity_funcs).reshape((neurons_cnt, neurons_cnt))
for i in range(neurons_cnt):
	connections_plasticity_funcs[i][i] = plasticity_func_00
print connections_plasticity_funcs
# ********* stimuli *********
stimuli_probs = np.random.rand(neurons_cnt)
print 'stimuli_probs', stimuli_probs

# ********* trainning *********
# for i in range(int(40000000)):
for _ in range(int(rounds_to_run)):
	rounds += 1
	if debug: print 'round ', rounds
	neurons_fired = np.array([False] * neurons_cnt, dtype=np.bool)  # new round begins, no neurons fired
	neurons_stimulated = np.random.rand(neurons_cnt) < stimuli_probs  # decide what neurons are stimulated from environment
	if neurons_stimulated.sum(): # no neurons are stimulated, skip this round
		# propagate action potential down through the neural tree
		while True:
			# print 'neurons_stimulated', neurons_stimulated
			neurons_to_fire = (~ neurons_fired) & neurons_stimulated
			# print 'neurons_to_fire', neurons_to_fire
			if not neurons_to_fire.sum(): # no neurons are left to fire, dead end
				break
			neurons_fire_counts += neurons_to_fire
			# build downstream stimulated neurons for next loop
			neurons_stimulated = np.array([False] * neurons_cnt, dtype=np.bool)
			connections_to_propagate = np.random.rand(connections_cnt).reshape((neurons_cnt, neurons_cnt)) < connections_transmission_probs
			# print connections_to_propagate
			for i, row in enumerate(connections_to_propagate):
				for j, propag in enumerate(row):
					if not propag:  # connection from neuron i to j failed to propagate action potential
						continue
					if not neurons_to_fire[i]:  # only consider lastly stimulated neurons
						continue
					connections_transmission_cnt[i][j] += 1  # fire together, wired together
					if neurons_fired[j] or neurons_to_fire[j]:  # if neuron j is already fired, skip in case recount and circle with parent neuron
						continue
					neurons_stimulated[j] = True  # neuron j will be stimulated if any upstream neurons propagate to it
					# connections_transmission_cnt[i][j] += 1
			neurons_fired |= neurons_to_fire
	# plasticity of neural connections
	if debug: print connections_transmission_probs
	connections_transmission_rates = connections_transmission_cnt / rounds
	if debug: print connections_transmission_rates
	for i, row in enumerate(connections_transmission_probs):
			for j, old_prob in enumerate(row):
				plasticity_func = connections_plasticity_funcs[i][j]
				prob_to_be = plasticity_func(connections_transmission_rates[i][j])
				if np.abs(prob_to_be - old_prob) < 0.001:
					pass
				elif prob_to_be > old_prob:
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
	# data.append(np.sqrt(np.divide(np.sum(np.square(connections_transmission_probs)), connections_cnt)))
	stat = np.sum(connections_transmission_probs)
	data.append(stat)
	if debug: print stat

data = np.array(data)
write_pickle(data, 'data.pkl')
write_pickle(connections_transmission_probs, 'conn_probs.pkl')

# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import strengthen_functions


class Queue_(object):
    def __init__(self, qlen):
        self.records = np.array([0] * qlen, dtype=np.int8)


class Connection(object):
    def __init__(self, init_strength=np.random.rand(), pf=strengthen_functions.PF34, transmission_history_len=10**4):
        # value 1 represents the occurance of transmission
        self.strength = init_strength
        self.transmission_history_len = transmission_history_len
        self.transmission_history_pointer = 0
        self.transmission_history = Queue_(self.transmission_history_len)
        self.strengthen_function = pf
        self.strengthen_rate = float(1) / self.transmission_history_len
        self.I = 0


    def propagate_once(self, stimulus_prob, debug=False, return_target_strength=False):
        self.I += 1
        target_strength = 0
        self.transmission_history.records[self.transmission_history_pointer] = 0
        if stimulus_prob > np.random.rand():
            if self.strength > np.random.rand():
                self.transmission_history.records[self.transmission_history_pointer] = 1
        if self.I > self.transmission_history_len:
            frequency = self.transmission_history.records.sum() / float(self.transmission_history_len)
            target_strength = self.strengthen_function(frequency)
            current_strength = self.strength

            if target_strength > current_strength:
                self.strength = np.min((current_strength + self.strengthen_rate, 1))
            else:
                self.strength = np.max((0, current_strength - self.strengthen_rate))

        self.transmission_history_pointer = (self.transmission_history_pointer + 1) % self.transmission_history_len

        if return_target_strength:
            return target_strength

    def get_frequency(self):
        return self.transmission_history.records.sum() / float(self.transmission_history_len)

    def get_strength(self):
        return self.strength


if __name__ == "__main__":
    pass

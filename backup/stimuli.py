# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np


class Stimuli(object):
    def __init__(self):
        pass

    @classmethod
    def on(cls, brain):
        # let brain know its connections
        stimuli = Stimuli()
        neurons_cnt = brain.neurons_cnt
        stimuli_probs = np.random.rand(neurons_cnt)
        print 'initial stimuli probs', stimuli_probs
        brain.stimuli = stimuli_probs
        return stimuli

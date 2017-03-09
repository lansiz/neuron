# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from neuron import Brain, Connections, Stimuli

b = Brain(5)
conns = Connections.grow_on(b, step=0.00001)
Stimuli.on(b)
b.fire(40 * 10 ** 4)
conns.plot_data()
plt.savefig('fixed_points_of_synapses.png')

# !/usr/bin/env python
#  -*- coding: utf-8 -*-

from neuron import Brain, Connections, Stimuli

b = Brain(5)
conns = Connections.grow_on(b)
Stimuli.on(b)
b.fire(22 * 10 ** 4)
conns.plot_data()

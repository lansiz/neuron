# import matplotlib as mpl
# mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from connection import Connection
import strengthen_functions
import numpy as np

np.random.seed()
# ax1, ax2 = plt.subplots(1, 2)
# plt.figure(figsize=(10, 3))
pf = strengthen_functions.PF34
stimulus_prob = .8
init_strength = .1
conn = Connection(init_strength=init_strength, pf=pf, transmission_history_len=10**4)
strength = []
for _ in range(100000):
    conn.propagate_once(stimulus_prob=stimulus_prob)
    strength.append(conn.get_strength())
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(strength, strength, linewidth=2)
ax.set_ylim(0, 1)
ax.axvline(strength[-1], linestyle=':', color='gray')
ax.axhline(strength[-1], linestyle=':', color='gray')
x = np.linspace(0, 1, 100)
ax.plot(x, np.array([pf(i) for i in x]))
ax.plot(x, np.array([pf(i * stimulus_prob) for i in x]))
# ax.plot(x, x, '--')
plt.savefig('conn_07.png')
plt.show()

# import matplotlib as mpl
# mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
from connection import Connection
import strengthen_functions
import numpy as np

np.random.seed()
# ax1, ax2 = plt.subplots(1, 2)
# plt.figure(figsize=(10, 3))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
for i in range(11):
    conn = Connection(init_strength=.1 * i,
                      pf=strengthen_functions.PF13, transmission_history_len=10**4)
    strength = []
    frequency = []
    for i in range(30 * 10 ** 4):
        conn.propagate_once(stimulus_prob=.8)
        strength.append(conn.get_strength())
        # frequency.append(conn.get_frequency())
    ax.plot(strength, alpha=.8)
    # ax2.plot(frequency, alpha=.2, color='black')
    # ax1.set_xlabel('(a)')
    # ax2.set_xlabel('(b)')
    ax.set_ylim(0, 1)
    # ax2.set_ylim(0, 1)
# plt.grid(True)
plt.savefig('conn_02.png')
plt.show()

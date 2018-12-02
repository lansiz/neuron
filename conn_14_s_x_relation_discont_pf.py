import numpy as np
import strengthen_functions
from connection import Connection
import matplotlib.pyplot as plt
import multiprocessing
import matplotlib as mpl
mpl.use('Agg', warn=False)

trails = 20
x_number = 20


def seek_fp(x):
    e_l = []
    for i in range(trails):
        conn = Connection(pf=strengthen_functions.PF15)
        # the init_strength should be reset with random
        conn.strength = np.random.rand()
        for i in range(100000):
            conn.propagate_once(stimulus_prob=x)
        e_l.append(conn.strength)
    return e_l


np.random.seed()
worker_pool = multiprocessing.Pool(processes=4)
xs = np.linspace(0, 1, x_number, endpoint=False)
results_l = [worker_pool.apply_async(seek_fp, (x,)).get() for x in xs]

fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
for t in range(trails):
    y = [i[t] for i in results_l]
    ax.plot(xs, y, 'o', color='gray', alpha=.6, zorder=1)
mean_l = [np.array(i).mean() for i in results_l]
ax.plot(xs, mean_l, color='red', alpha=1, zorder=2)

# xs = np.linspace(0, 1, 100)
# y = np.array([(5. / (100 - 90 * i)) for i in xs])
# y = np.array([(1. / (1 + i)) for i in xs])
# ax.plot(xs, y, color='blue', alpha=1, zorder=3)
ax.tick_params(labelsize=10)

ax.set_ylim(0, 1)
plt.savefig('conn_10_pf15.png')
# plt.show()

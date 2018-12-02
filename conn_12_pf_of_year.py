from __future__ import print_function
from connection import Connection
import strengthen_functions
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg', warn=False)
# import multiprocessing

gs = gridspec.GridSpec(1, 8)
ax1 = plt.subplot(gs[:, :3])
ax2 = plt.subplot(gs[:, 3:8], sharey=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)

fig = ax1.get_figure()
fig.set_figwidth(10)
fig.set_figheight(4)
# gs.tight_layout(fig, rect=[0, 0, 1, 0.950])
fig.subplots_adjust(wspace=1)

pf_step = strengthen_functions.PF80
pf_linear = strengthen_functions.PF81

'''
def theta_step(x):
    a1 = 1.99
    a2 = 4.2
    a3 = 0.01
    a4 = 0.97
    return -(1 / a2 * x) * np.log((a1 / (x + a4)) - 1) + a3 / x


def theta_linear(x):
    a1 = 1.89
    a2 = 3.8
    a3 = 0.00
    a4 = 0.86
    return -(1 / a2 * x) * np.log((a1 / (x + a4)) - 1) + a3 / x
'''


figname = 'conn_12.png'


def discontinue(y):
    b_l = np.concatenate(([False], (np.abs(np.diff(y)) >= 0.1)))
    y[b_l] = np.nan
    return y


x = np.linspace(0, 1, 100)
ax1.plot(x, x, linewidth=1, linestyle=':', color='gray')
y = np.array([pf_step(i) for i in x])
ax1.plot(x, discontinue(y), linewidth=2, linestyle='-', color='green')
y = np.array([pf_linear(i) for i in x])
ax1.plot(x, discontinue(y), linewidth=2, linestyle='-', color='blue')
ax1.grid()


ax2.plot(x, x, linewidth=1, linestyle=':', color='gray')
ax2.grid()

# simulation to show s+~x relation
trails = 10
x_number = 20


def seek_fp(pf, x):
    e_l = []
    for i in range(trails):
        conn = Connection(pf=pf)
        # the init_strength should be reset with random
        conn.strength = np.random.rand()
        for i in range(100000):
            conn.propagate_once(stimulus_prob=x)
        e_l.append(conn.strength)
    return e_l


np.random.seed()
xs = np.linspace(0, 1, x_number)
# ax2.tick_params(labelsize=14)

# for step sigmoid PF step
results_l = [seek_fp(pf_step, x_) for x_ in xs]
for t in range(trails):
    y = [i[t] for i in results_l]
    ax2.plot(xs, y, 'o', color='green', alpha=.3, zorder=1)
mean_l = [np.array(i).mean() for i in results_l]
ax2.plot(xs, mean_l, color='green', alpha=1, zorder=2)

# for step sigmoid PF linear
results_l = [seek_fp(pf_linear, x_) for x_ in xs]
for t in range(trails):
    y = [i[t] for i in results_l]
    ax2.plot(xs, y, 'o', color='blue', alpha=.3, zorder=1)
mean_l = [np.array(i).mean() for i in results_l]
ax2.plot(xs, mean_l, color='blue', alpha=1, zorder=2)

plt.savefig(figname)
# plt.show()

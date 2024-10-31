from connection import Connection
import strengthen_functions
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg', warn=False)

gs = gridspec.GridSpec(1, 15)
ax1 = plt.subplot(gs[:, :5])
ax2 = plt.subplot(gs[:, 5:15])
plt.setp(ax2.get_yticklabels(), visible=False)

fig = ax1.get_figure()
fig.set_figwidth(5)
fig.set_figheight(1.5)
# gs.tight_layout(fig, rect=[0, 0, 1, 0.950])
fig.subplots_adjust(wspace=1)

pf = strengthen_functions.PF12
figname = 'conn_09_pf12.png'


def discontinue(y):
    b_l = np.concatenate(([False], (np.abs(np.diff(y)) >= 0.1)))
    y[b_l] = np.nan
    return y


stimulus_prob = .3
# stimulus_prob = .8
# stimulus_prob = .9
x = np.linspace(0, 1, 100, endpoint=False)
y = np.array([pf(i) for i in x])
y_scaled = np.array([pf(stimulus_prob * i) for i in x])
ax1.plot(x, x, linewidth=1, linestyle=':', color='gray')
ax1.plot(x, discontinue(y), linewidth=2, linestyle='-', color='black')
ax1.plot(x, discontinue(y_scaled), linewidth=2, linestyle='-', color='red')

for i in range(11):
    conn = Connection(init_strength=.1 * i, pf=pf,
                      transmission_history_len=10**4)
    strength = []
    for i in range(1 * 100000):
        conn.propagate_once(stimulus_prob=stimulus_prob)
        strength.append(conn.get_strength())
    ax2.plot(strength, alpha=.8)

ax1.tick_params(labelsize=8)
ax2.tick_params(labelsize=8)

plt.savefig(figname)
plt.show()

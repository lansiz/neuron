# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import random
import numpy as np


# plasticity functions
def PF00(x):
    '''y=0'''
    return 0


def PF01(x):
    '''1/(1+exp(-10*(x-.5)))'''
    return 1 / (1 + np.exp(-10 * (x - .5)))


def PF02(x):
    '''y=2/3'''
    return float(2) / 3


def PF03(x):
    '''y=1'''
    return 1


def PF04(x):
    '''y=2x & y=-2x+2'''
    if x < .5:
        return 2 * x
    else:
        return -2 * x + 2


def PF05(x):
    '''y=-2x+1 & y=2x-1'''
    if x < .5:
        return -2 * x + 1
    else:
        return 2 * x - 1


def PF06(x):
    '''y=-x+1 & y=x'''
    if x < .5:
        return -x + 1
    else:
        return x


def PF07(x):
    '''y=x & y=-x+1'''
    if x < .5:
        return x
    else:
        return -x + 1


def PF08(x):
    '''y=-4xx+4x'''
    return -4 * x * x + 4 * x


def PF09(x):
    '''y=4xx-4x+1'''
    return 4 * x * x - 4 * x + 1


def PF10(x):
    u'''y=.5sin(2πx)+.5'''
    return .5 * np.sin(2 * np.pi * x) + .5


def PF11(x):
    u'''y=.5sin(-2πx)+.5'''
    return .5 * np.sin(-2 * np.pi * x) + .5


def PF12(x):
    u'''y=.5sin(4πx)+.5'''
    return .5 * np.sin(4 * np.pi * x) + .5


def PF13(x):
    u'''y=.5sin(8πx)+.5'''
    return .5 * np.sin(8 * np.pi * x) + .5


def PF14(x):
    u'''0 & y=3x-1 & 1'''
    if x < (1. / 3):
        return 0
    elif x < (2. / 3):
        return 3 * x - 1
    else:
        return 1


def PF15(x):
    u'''discontinious'''
    if x < .2:
        return .2 * x
    elif x < .4:
        return .2 * x + .2
    elif x < .6:
        return .2 * x + .4
    elif x < .8:
        return .2 * x + .6
    else:
        return .2 * x + .8


def PF29(x):
    '''y=x'''
    return x


def PF30(x):
    '''y=-x+1'''
    return -x + 1


def PF31(x):
    '''y=.8x+.1'''
    return .8 * x + .1


def PF32(x):
    '''y=.9x+.05'''
    return .9 * x + .05


def PF33(x):
    u'''.5*sin(-(x+.5)π)+.5'''
    return .5 * np.sin(-1 * np.pi * (x + .5)) + .5


def PF34(x):
    '''trimmed sigmoid'''
    return (np.exp(3 * x) / (1 + np.exp(3 * x)) - .5) * 1.8 + .05


def PF35(x):
    '''trimmed sigmoid'''
    return (np.exp(5 * x) / (1 + np.exp(5 * x)) - .5) * 1.8 + .05


def funcs_pool_all():
    globals_ = globals().copy()
    globals_ = [(k, globals_[k]) for k in sorted(globals_.keys())]
    plasticity_funcs = []
    for name, obj in globals_:
        if 'PF' in name:
            plasticity_funcs.append(obj)
    return plasticity_funcs


def funcs_sample(cnt):
    plasticity_funcs = funcs_pool_all()
    funcs_sampled = []
    for i in range(cnt):
        funcs_sampled.append(random.choice(plasticity_funcs))
    return funcs_sampled


def funcs_sample_one():
    plasticity_funcs = funcs_pool_all()
    return random.choice(plasticity_funcs)


def print_name(arr):
    arr = arr.copy()
    for i, row in enumerate(arr):
        for j, func in enumerate(row):
            arr[i][j] = func.__name__.lower()
    return arr


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('Agg', warn=False)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plasticity_funcs = funcs_pool_all()
    print('plasticity functions cnt: %s' % len(plasticity_funcs))
    # plot the functions
    funcs_count = len(plasticity_funcs)
    col_cnt = 6
    row_cnt = 5
    fig = plt.figure(figsize=(12, 1.6 * row_cnt + 1))
    gs = gridspec.GridSpec(row_cnt, col_cnt)
    ax_list = [fig.add_subplot(s) for s in gs]
    x = np.linspace(0, 1, 100)
    ticks = [0, .5, 1]
    xlabels = ylabels = ticks
    for func, ax in zip(plasticity_funcs, ax_list[:funcs_count]):
        y = [func(i) for i in x]
        ax.plot(x, y, linewidth=2, color='red')
        # for tick in ax.xaxis.get_major_ticks(): tick.set_visible(False)
        # for tick in ax.yaxis.get_major_ticks(): tick.set_visible(False)
        ax.set_xticks(ticks)
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ylabels, rotation=0)
        ax.set_title(func.__doc__, fontsize=10)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        ax.grid(True)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.975])
    # fig.suptitle('plasticity functions', fontsize=12)
    fig.savefig('plasticity_functions.png')

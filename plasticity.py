# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import random

# plasticity functions
def plasticity_func_00(x):
    '''y=0'''
    return 0


def plasticity_func_01(x):
    '''y=1/3'''
    return float(1) / 3


def plasticity_func_02(x):
    '''y=2/3'''
    return float(2) / 3


def plasticity_func_03(x):
    '''y=1'''
    return 1


def plasticity_func_04(x):
    '''y=2x & y=-2x+2'''
    if x < .5:
        return 2 * x
    else:
        return -2 * x + 2


def plasticity_func_05(x):
    '''y=-2x+1 & y=2x-1'''
    if x < .5:
        return -2 * x + 1
    else:
        return 2 * x - 1


def plasticity_func_06(x):
    '''y=-x+1 & y=x'''
    if x < .5:
        return -x + 1
    else:
        return x


def plasticity_func_07(x):
    '''y=x & y=-x+1'''
    if x < .5:
        return x
    else:
        return -x + 1


def plasticity_func_08(x):
    '''y=-4xx+4x'''
    return -4 * x * x + 4 * x


def plasticity_func_09(x):
    '''y=4xx-4x+1'''
    return 4 * x * x - 4 * x + 1


def plasticity_func_10(x):
    u'''y=.5sin2πx+.5'''
    return .5 * np.sin(2 * np.pi * x) + .5


def plasticity_func_11(x):
    u'''y=.5sin-2πx+.5'''
    return .5 * np.sin(-2 * np.pi * x) + .5


def plasticity_func_29(x):
    '''y=x'''
    return x


def plasticity_func_30(x):
    '''y=-x+1'''
    return -x + 1


def funcs_sample(cnt):
    globals_ = globals().copy()
    globals_ = [(k, globals_[k]) for k in sorted(globals_.keys())]
    plasticity_funcs = []
    for name, obj in globals_:
        if 'plasticity_func' in name:
            plasticity_funcs.append(obj)
    funcs_sampled = []
    for i in range(cnt):
        funcs_sampled.append(random.choice(plasticity_funcs))
    return funcs_sampled


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    globals_ = globals().copy()
    globals_ = [(k, globals_[k]) for k in sorted(globals_.keys())]
    plasticity_funcs = []
    for name, obj in globals_:
        if 'plasticity_func' in name:
            plasticity_funcs.append(obj)
    print 'plasticity_funcs cnt: %s' % len(plasticity_funcs)
    # plot the functions
    funcs_count = len(plasticity_funcs)
    col_cnt = 6
    row_cnt = 5
    fig = plt.figure('plasticity functions', figsize=(12, 1.6 * row_cnt + 1))
    gs = gridspec.GridSpec(row_cnt, col_cnt)
    ax_list = [fig.add_subplot(s) for s in gs]
    x = np.linspace(0, 1, 100)
    ticks = [0, .5, 1]
    xlabels = ylabels = ticks
    for func, ax in zip(plasticity_funcs, ax_list[:funcs_count]):
        y = [func(i) for i in x]
        ax.plot(x, y)
        ax.set_xticks(ticks)
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ylabels, rotation=0)
        ax.set_title(func.__doc__, fontsize=10)
        ax.grid(True)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.975])
    fig.suptitle('plasticity functions', fontsize=12)
    fig.savefig('plasticity_functions.png')

# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import multiprocessing


def seek_fixed_point(gene, stimu_pool, NN_class, I, strengthen_rate):
    nn = NN_class(gene)
    nn.initialize_synapses_strength()
    nn.set_strengthen_functions()
    for i in range(I):
        # pick one stimulus from pool
        stimu = stimu_pool.pick_one()
        nn.propagate_one_stimulus(stimu, strengthen_rate)
    return nn.stats()
    # {'connections_strengh': [1, 2, 3], 'accuracy': .3, 'other': 3}


class Environment(object):
    def __init__(self, worker_number=3, name=''):
        self.worker_pool = multiprocessing.Pool(processes=worker_number)

    def evaluate_fitness(self, gene_pool, stimu_pool, NN_class, I, strengthen_rate):
        ''' Method to evluate genes' fittness (accuracy) after through I iterations of stimuli,
        Paramenter NN_class should be NeuralNetork class or its subclasses. '''
        return [self.worker_pool.apply_async(
            seek_fixed_point, (gene, stimu_pool, NN_class, I, strengthen_rate)).get() for gene in gene_pool]

    def select_mating_pool(self, nn_pool, stats_d):
        # return mating_pool
        return nn_pool

    def reproduce_nn_pool(self, mating_pool, P):
        # return nn_pool
        return mating_pool

    def output_stats(self, nn_pool):
        pass

    def store_stats(self, nn_pool):
        pass

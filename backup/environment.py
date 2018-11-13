# !/usr/bin/env python
#  -*- coding: utf-8 -*-
import multiprocessing
import numpy as np
import random
from gene import Gene


def seek_fixed_point(gene, stimu_pool, NN_class, I, strengthen_rate):
    nn = NN_class(gene)
    nn.initialize_synapses_strength()
    nn.set_strengthen_functions()
    for i in range(I):
        nn.propagate_once(stimu_pool, strengthen_rate)
    nn.evaluate_accuracy(stimu_pool)
    return nn.stats()


class Environment(object):
    def __init__(self, worker_number=40, name=''):
        self.worker_pool = multiprocessing.Pool(processes=worker_number)

    def evaluate_fitness(self, gene_pool, stimu_pool, NN_class, I, strengthen_rate):
        '''
        Method to evluate genes' fittness (accuracy) after through I iterations of stimuli,
        Paramenter NN_class should be NeuralNetork class or its subclasses.
        '''
        return [self.worker_pool.apply_async(
            seek_fixed_point, (gene, stimu_pool, NN_class, I, strengthen_rate)).get() for gene in gene_pool]

    def build_new_gene_pool(self, P, nn_pool, stats_l, mating_pool_size=1000, strength_threshhold=.0):
        # combine genes and its fittness stats
        fit_data = []
        for gene, stat in zip(nn_pool, stats_l):
            stat['gene'] = gene
            fit_data.append(stat)

        # perish those with zero accuracy
        '''
        fit_data = filter(lambda x: x['accuracy'] > 0, fit_data)
        # if no gene could scores, just end the evolution
        if not len(fit_data):
            print('all genes extincted')
            return None
        '''

        # statistical ouput before doing anything
        # accuracy stats
        accuracy_a = np.array([d['accuracy'] for d in fit_data])
        print('accurcy stats - count: %s max: %s min: %s mean: %s std: %s' % (
            accuracy_a.shape[0], accuracy_a.max().round(4), accuracy_a.min().round(4),
            accuracy_a.mean().round(4), accuracy_a.var().round(4)))
        # strength matrix stats
        strength_a = np.array([])
        for d in fit_data:
            strength_a = np.concatenate((strength_a, d['strength_matrix'].flatten()))
        strength_a = strength_a[strength_a >= 0]
        if not strength_a.shape[0]:  # no connections, quit evolution
            print('all synaptic connections extincted')
            return None
        print('strength stats - count: %s max: %s min: %s mean: %s std: %s' % (
            strength_a.shape[0], strength_a.max().round(4), strength_a.min().round(4),
            strength_a.mean().round(4), strength_a.var().round(4)))

        return self.selection(fit_data, strength_threshhold)

    def selection(self, fit_data, strength_threshhold):
        # for the gene with best accuracy, find out its strongest and weakest connections.
        fit_data = sorted(fit_data, key=lambda x: x['accuracy'], reverse=True)
        the_best = fit_data[0]
        # gene = the_best['gene'] 
        strength_matrix = the_best['strength_matrix']
        strength_max = strength_matrix.max()
        location_max = np.where(strength_matrix == strength_max)
        max_i = location_max[0][0]
        max_j = location_max[1][0]
        '''
        strength_min = strength_matrix.min()
        location_min = np.where(strength_matrix == strength_min)
        min_i = location_min[0][0]
        min_j = location_min[1][0]
        '''

        # prevail the strongest connection and perish the weakest connection
        for d in fit_data:
            gene = d['gene']
            gene.connections[max_i][max_j] = 1
            gene.connections[max_j][max_i] = 0
            # gene.connections[min_i][min_j] = 0

        # perish connections with weak strength from gene
        for i in fit_data:
            temp = i['strength_matrix'] >= strength_threshhold
            i['gene'].connections &= temp
            i['gene'].connections_number = i['gene'].connections.sum()

        return [i['gene'] for i in fit_data]

    def selection_1(self, fit_data, strength_threshhold):
        # normalize accuracy for pool selection for i in fit_data:
        i['accuracy'] /= accuracy_a.max()

        # pool selection 
        mating_pool = []
        while len(mating_pool) < mating_pool_size:
            choice = random.choice(fit_data)
            r = np.random.rand()
            if choice['accuracy'] > r:
                mating_pool.append(choice)

        # perish connections with weak strength from gene
        for i in mating_pool:
            temp = i['strength_matrix'] >= strength_threshhold
            i['gene'].connections &= temp
            i['gene'].connections_number = i['gene'].connections.sum()

        # mating pool for reproduction
        mating_pool = [i['gene'] for i in mating_pool]

        # Sample two parents from mating pool, reproduce children and impose muatation on the newborn.
        gene_pool = []
        for _ in range(P):
            new_gene = Gene.crossover(random.sample(mating_pool, 2)) 
            gene_pool.append(new_gene)

        # mutation

        return gene_pool

    def output_stats(self, nn_pool):
        pass

    def store_stats(self, nn_pool):
        pass


if __name__ == "__main__":
    env = Environment()
    gene_pool = Gene.creations(1, 7)
    stats_l = []
    for gene in gene_pool:
        tmp_m = np.random.rand(7, 7)
        connection_strength_m = np.where(gene.connections == 1,tmp_m, -1)
        stat = {}
        stat['strength_matrix'] = connection_strength_m
        stat['accuracy'] = np.random.rand()
        stats_l.append(stat)
    mating_pool = env.select_mating_pool(gene_pool, stats_l, mating_pool_size=10, strength_threshhold=.1)
    print(mating_pool)


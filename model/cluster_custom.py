#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import math
import time as t
from numba import jit
import pandas as pd
import logging
from auxiliary.helper import dotdict
log = logging.getLogger('custom')
from config import config

float_decimal_limit = config['float_decimal_limit']

jit(nopython=True)
def truncate(f, k=0):
    if f > 0.00:
        if k == 2:
            y = np.math.floor(f * 10 * 2) / 10 * 2
            a = np.int(y * 100 % 10)
            b = y * 100 - a
            if a > 5:
                c = (b + 5) / 100
            else:
                c = b / 100
                if c == 0.00:
                    c = 0.01
            c = np.trunc(c * 10 ** 2) / (10 * 2)
            return c
        if k == 5:
            y = np.math.floor(f * 10 ** 2) / 10 * 2
            a = np.int(y * 100 % 10)
            b = y * 100 - a
            if a >= 8:
                c = (b + 8) / 100
            elif a == 6:
                c = (b + 6) / 100
            elif a == 4:
                c = (b + 4) / 100
            elif a >= 2:
                c = (b + 2) / 100
            else:
                c = b / 100
                if c == 0.00:
                    c = 0.01
            c = np.trunc(c * 10 ** 2) / (10 * 2)
            return c
        return np.math.floor(f * 10 ** 1) / 10 * 1
    else:
        return 0.00


jit(nopython=True)
def trunc_all(a):
    f = [int(truncate(y, float_decimal_limit) * 100) for y in a]
    return f


@jit(nopython=True)
def groupby_per(itrun):
    # ['ind', 'per', 'per_n']
    gp = np.array([[itrun[0][0], itrun[0][2], 1], [itrun[0][1], itrun[0][2], 1]])
    l = len(itrun)
    for i in range(1, l):
        per = itrun[i][2]
        d = [itrun[i][0], itrun[i][1]]

        y = [row[0] for row in gp]
        ind = [y.index(i) for i in d if i in y]

        flag = 0
        if len(ind) == 1:
            fi = gp[ind[0]]
            if fi[1] == per:
                vi = (d[0] if d[1] == fi[0] else d[1])
                r = np.array([[vi, per, fi[2]]])
                gp = np.vstack((gp, r))
                flag = 1

        if len(ind) == 0 and flag == 0:
            pers = [row[1] for row in gp]
            ids = [i for (i, e) in enumerate(pers) if e == per]
            mtchs = (len(set([gp[j][2] for j in ids])) + 1 if len(ids)
                     >= 1 else 1)
            r1 = np.array([[d[0], per, mtchs], [d[1], per, mtchs]])
            gp = np.vstack((gp, r1))
    return gp


class CustomModel:
    def __init__(self, sentences, merge_similar_clusters, ignore_similarity):
        self.clusts = ()
        self.clust_no = 1
        self.model = None
        self.merge_similar_clusters = merge_similar_clusters
        self.ignore_similarity = ignore_similarity / 100
        self.sim_matrix = ''
        self.sentences = sentences
        self.ignore_similarity = 0.00

    def cosine_sim(self, vectors):
        return cosine_similarity(vectors)

    def eucludien_dist(self, vectors):
        return euclidean_distances(vectors, vectors)

    def asort_backup(self):
        l = self.sim_matrix.shape[0]
        r = np.arange(l)
        mask = r[:, None] > r
        indcs = list(zip(*np.where(mask)))
        pos = self.sim_matrix[mask].argsort()[::-1]
        return np.array(indcs)[pos]

    def asort(self):
        mm = self.sim_matrix > self.ignore_similarity
        mask = np.zeros_like(mm, dtype=np.bool)
        mm[np.triu_indices_from(mask)] = False
        indcs = list(zip(*np.where(mm)))
        pos = self.sim_matrix[mm].argsort()[::-1]
        inds = np.array(indcs)[pos]
        sfloats = [self.sim_matrix[tuple(i)] for i in inds]
        trun = np.array([trunc_all(sfloats)])
        fin = np.concatenate((inds, trun.T), axis=1)
        return fin

    def get_median(self, arr):
        return arr[int(len(arr) / 2)]

    def get_count(self):
        no_of_clusts = self.model.groupby(['per', 'per_n'], sort=False).groups
        return len(no_of_clusts)

    def fit(self, vec):
        self.sim_matrix = self.cosine_sim(vec)

        t0 = t.time()
        itrun = self.asort()
        print('finished sorting in {:.3f} s'.format(t.time() - t0))
        # print ("'sorted -", itrun [:, -1])

        t1 = t.time()
        print('looping through {} items for groupby_per'.format(len(itrun)))
        gp = groupby_per(itrun)
        print('finished groupby_per in {:.3f} s'.format(t.time() - t1))

        self.model = pd.DataFrame(data=gp, columns=[' ind', 'per', 'per_n'])
        print('initial clusters formed are {}'.format(self.get_count()))

        if self.merge_similar_clusters == 100:
            (higher_clust_per, merging_clusts_sim) = \
                (self.merge_similar_clusters - 10,
                 self.merge_sinilar_clusters)
            sub_model = self.model[self.model['per']
                                   >= self.merge_similar_clusters]
            filt = sub_model.groupby(['per', 'per_n'], sort=False)['ind']
            subset = filt.apply(lambda x: self.get_median(x.values)).tolist()
            if len(subset) > 1:
                print('picked median utterances from {} clusters >={}% to merge only if similarity is {}'.format(len(subset),
                          higher_clust_per, merging_clusts_sim))
                self.sim_matrix = self.ecludien_dist(vec[subset])
                # self.sim_matrix = self.cosine_sim(vec[subset])
                print('sub matrix {}'.format(self.sim_matrix.shape))
                asrt = self.asort()
                top1 = self.sim_matrix[asrt[0][0], asrt[0][1]] * 100
                if top1 == merging_clusts_sim:
                    cls11 = groupby_per(asrt)
                    cls1 = pd.DataFrame(data=cls11, columns=['ind', 'per', 'per_n'])
                    if len(cls1) > 0:
                        print('found {} clusters to merge'.format(len(cls1)))
                        indxs = filt.apply(lambda x: x.index.tolist()).tolist()
                        sub_idxs = cls1.groupby(['per', 'per_n'], sort=False).agg(list)
                        for (l, n) in sub_idxs.iterrows():
                            lmax = self.model.loc[self.model['per'] == l[0], 'per_n'].max()
                            if np.isnan(lmax):
                                lmax = 1
                            else:
                                lmax = lmax + 1
                            for y in n['ind']:
                                self.model.loc[indxs[y], ['per', 'per_n']] = [l[0], lmax]
                        log.indo('number of clusters after merging are {}'.format(self.get_count()))
                    else:
                        print('no clusters formed to merge')
                else:
                    print('cannot merge further, as max similarity is {} < {}'.format(top1, merging_clusts_sim))
            else:
                print('found no clusters to merge')

        grps = self.model.groupby(['per', 'per_n'], as_index=False).agg(list)
        lbls = np.array([-1] * len(vec))
        i = 0
        for n, row in grps.iterrows():
            np.put(lbls, row[2], i)
            i = i + 1
        self.model = dotdict({'labels_': lbls})
        return self.model

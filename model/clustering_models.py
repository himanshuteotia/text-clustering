import sklearn.cluster as cluster
from joblib import Memory
import numpy as np
import time as t
import json
from config import config
from model.cluster_custom import CustomModel

try:
    import hdbscan
except:
    print("error while importing hdbscan, please check if the package is installed.")

class ClusteringModels:
    def __init__(self, sentences, vec, k=None):
        self.sentences = sentences
        self.vec = vec
        self.k = k
        self.model = None

    def cluster(self, alg_name, kwarg=None):
        alg = self.model_specs(alg_name)
        _kwarg = alg['specs']
        if kwarg is not None:
            for (k, v) in kwarg.items():
                if k in _kwarg.keys():
                    _kwarg[k] = v
        if alg_name != 'custom':
            print('{} params are {}'.format(alg_name, _kwarg))
        t1 = t.time()
        self.model = alg['model'](**_kwarg).fit(self.vec)
        print('{} clustered in {:.2f}s'.format(alg_name, t.time() - t1))
        return self.model

    def group_by(self, labels, min_samples, output_sentences):
        clusts = {}
        c = 1
        for l in set(labels):
            if l > -1:
                pos = np.where(labels == l)[0].tolist()
                clus = list(np.take(output_sentences, pos))
                if len(clus) >= min_samples:
                    clusts['C' + str(c) + str(len(clus))] = clus
                    c = c + 1
        return clusts

    def model_specs(self, algorithm):
        default_k = 10
        n_jobs = 4  # number of parallel jobs to run using processors. None means using 1 processor & -1 means all
        switcher = {
            'custom': {
                'model': CustomModel,
                'specs': {'sentences': self.sentences, 'merge_similar_clusters': 90, 'ignore_similarity': 30}
            },
            "ahc": {
                'model': cluster.AgglomerativeClustering,
                'specs': {'n_clusters': default_k, 'linkage': 'warn', 'affinity': 'euclidean', 'memory': None,
                          'connectivity': None, 'compute_full_tree': 'auto', 'distance_threshold': None}
            },
            'kmeans':{
                'model': cluster.KMeans,
                'specs': {'n_clusters': default_k, 'max_iter': 600, 'algorithm': 'auto', 'init': 'k-means++',
                          'n_init':10, 'tol': 1e-4, 'verbose': 0, 'random_state': None, 'copy_x': True}
            },
            'hdbscan':{
                'model': hdbscan.HDBSCAN,
                'specs': {'algorithm': 'best', 'approx_min_span_tree': True, 'gen_min_span_tree': True,
                          'memory': Memory(cachedir=None), 'metric': 'euclidean', 'min_cluster_size': 5,
                          'min_samples': 5, 'p': None}
            }
        }
        return switcher.get(algorithm, 'kmeans')


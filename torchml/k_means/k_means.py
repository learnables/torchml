from __future__ import annotations
import math
import torch
import random
#from torchmetrics.functional import pairwise_euclidean_distance
import torchml as ml


class KMeans(ml.Model):

    """
    ## Description

    Implementation of K Means Clustering Algorithm for pytorch.

    ## References


    ## Arguments

    * `metric` (str or callable, default="euclidean"):
        Metric to use for distance computation. Only Euclidiean metric is supported for now.

    * `shrink_threshold` (float, default=None):
        Threshold for shrinking centroids to remove features. Not supported for now.

    ## Example



    """
    def __init__(self, 
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        self.n_clusters=n_clusters,
        self.init=init,
        self.n_init=n_init,
        self.max_iter=max_iter,
        self.tol=tol,
        self.verbose=verbose,
        self.random_state=random_state,
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.label = []


    def check_params(self, X): 
        super().check_params(X)
        if self.init != "k-means++":
            raise ValueError(
                "Only k-means is supported for now, "
                f"got {self.alinitgorithm} instead."
            )
        if self.algorithm != "lloyd":
            raise ValueError(
                "Only Lloyd is supported for now, "
                f"got {self.algorithm} instead."
            )
    
    # finding the idx of nearast centroid of specific sample, return the idx of the centroid and the distance
    def near_cent(sample:torch.tensor, centroids:list)->tuple[int,float]: 
        _min_dist = torch.nn.PairwiseDistance(sample,centroids[0])
        _min_idx = -1
        for i in range(len(centroids)):
            _curr_dist = torch.nn.PairwiseDistance(sample,centroids[i])
            if _curr_dist < _min_dist:
                _min_dist = _curr_dist
                _min_idx = i
        return _min_idx, _min_dist
    
    def init_centroids(self, X:torch.tensor, x_squared_norms=None, init=None, random_state=None, init_size=None, n_centroids=None)->tuple(list,list):
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        #KMEANS++:
        
        #number of iteration used to find K centroids
        _iter = self.n_clusters
        #current centroid idx
        _curr_idx = random.randint(0,n_samples)
        #current centroid var
        _curr_cent = X[_curr_idx]
        #centroid list
        _cents = [_curr_cent]   
        for _ in range(1,_iter):
            _curr_cent = X[_curr_idx]
            _max_distance = 0
            _max_idx = -1
            _total_dist = 0
            #for every x in X:
            p = []
            sum_p = []
            #_dist = [0 for _ in n_samples]
            
            #looping through samples to find the 
            for i in range(n_samples):
                _near_idx, _near_dist = self.near_cent(X[i],_cents)
                _total_dist += _near_dist
                #_dist[i]=_near_idx

            _rand = random.random()
            
            for i in range(n_samples):
                _near_idx, _near_dist = self.near_cent(X[i],_cents)
                p[i] = _near_dist / _total_dist
                if i == 0:
                    sum_p[0] = p[i] 
                else:
                    sum_p[i] = p[i-1]+p[i]
                    if (sum_p[i]>=_rand):
                        _curr_idx = i
                        break
            
            #_rand = random.random()*_total_dist

            #_curr_idx = _max_idx
            _cents.append(X[_curr_idx])
            '''
            for i in range(n_samples):
                _temp_dist = torch.nn.PairwiseDistance(X[i],_curr_cent)
                if(_temp_dist>_max_distance):
                    _max_distance = _temp_dist
                    _max_idx = i
            _curr_idx = _max_idx
            _cents.append(X[_max_idx])
            

        
        for i in range(1,_iter):
            _random_idx = random.randint(0,n_samples)
            while(X[_random_idx]in _cent):
                _random_idx=random.randint(0,n_samples)
            _cent.append(X[_random_idx])
        '''
        _cluster = self.find_clusters(X,_cents)

        #dist: vec[idx of near centroid] = [idices of samples]
        return _cents,_cluster

        #_random_idx = random.randint(0,n_samples)

    def find_clusters(self, X:torch.tensor, centroids:list)->list:

        n_samples, n_features = X.shape
        _cluster = [[] for _ in range(n_samples)]

        for i in range(n_samples):
            _near_idx, _near_dist = self.near_cent(X[i],centroids)
            _cluster[_near_idx].append(i)
        
        return _cluster


    
    def fit(self, X:torch.tensor, y:torch.tensor=None , sample_weight:torch.tensor=None):
        
        #X_mean = torch.mean(X, axis=0)
        #X -= X_mean
        n_samples, n_features = X.shape
        _cents = []
        _label = []

        _cents,_cluster = self.init_centroids(X)
        # for up to max iteration:
        for _iter in range(1,self.max_iter):

            #_cents,_cluster = self.init_centroids(X)
            
            # calculate the new centers with mean of each centroid type
            _new_cents = torch.zeros(len(_cents),n_features)
            for idx,samps in enumerate(_cluster):
                _new_cents[idx] = torch.mean(samps,dim=0)

            #update cluster with new centroids
            _cluster = self.find_clusters(X,_new_cents)


            # if different in centroids doesn't change, converged
            diff_cents = _new_cents - _cents;
            if (diff_cents.sum()<self.tol):
                break
            
        for i in range(n_samples):
            for cent in _cluster:
                if X[i] in cent:
                    _label[i] = cent

        self._label = _label

        return 

def print_label(self):
    print(self._label)
    return




    
def predict(self, X:torch.tensor)->torch.tensor:
    ret = torch.tensor()



    return ret

    
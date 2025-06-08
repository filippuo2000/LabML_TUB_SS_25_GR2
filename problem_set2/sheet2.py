""" ps2_implementation.py

PUT YOUR NAME HERE:
Gustavo Eggert Martinez


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram


def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (n x d) data matrix with each datapoint in one row
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (k x d) matrix with each cluster center in one column
    r: assignment vector
    loss: the loss function value
    """

    n, d = X.shape
    r = np.full(n, -1)                                  # shape: n
    randIndc = np.random.choice(n, k, replace=False)    # shape: k
    mu = X[randIndc]                                    # shape: k x d

    for i in range(max_iter):
        Dist = cdist(X, mu, 'euclidean')                # shape: n x k
        rNew = np.argmin(Dist, axis=1)                  # shape: n

        changes = (r != rNew).sum()         
        if (changes == 0):
            break

        for j in range(k):
            C = X[rNew == j]                            # shape: |C_j| x d
            if (C.size > 0):
                mu[j] = C.mean(axis=0)

        loss = kmeans_crit(X, rNew, k, d)

        print('Number of iterations: ' + str(i+1))
        print('Number of changed cluster memberships : ' + str(changes))
        print('Loss function value : ' + str(loss))

        r = rNew

    print()
    return mu, r, loss


def kmeans_crit(X, r, k, d):
        """ Computes k-means criterion

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector
        k: number of clusters
        d: number of dimensions

        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """

        c = list(set(r))
        k = len(c)
        loss = 0

        mu = np.full((k, d), np.inf)        # shape: k x d
        for j in range(k):
            C = X[r == c[j]]                   # shape: |C_j| x d
            if (C.size > 0):
                mu[j] = C.mean(axis=0)
                loss += np.sum((C - mu[j])**2, axis=1).sum()

        return loss


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    X: (n x d) data matrix with each datapoint in one row
    r: (n) assignment vector

    Output:
    R: ((k-1) x n) matrix that contains cluster memberships before each step
    kmloss: (k) vector with loss after each step
    mergeidx: ((k-1) x 2) matrix that contains merge idx for each step
    """

    n, d = X.shape
    k = np.max(r) + 1
    R = np.full((k-1, n), -1)
    R[0] = r
    kmloss = np.full(k, np.inf)
    kmloss[0] = kmeans_crit(X, R[0], k, d)
    mergeidx = np.full((k-1, 2), -1)

    plot_clusters(X, r, k)

    for i in range(k - 1):
        clusters = list(set(R[i]))
        kmloss[i+1] = np.inf

        kNew = len(clusters)
        for j in range(kNew):
            c1 = clusters[j]
            for l in range(j+1, len(clusters)):
                c2 = clusters[l]
                rNew = [c2 if idx == c1 else idx for idx in R[i]]
                loss = kmeans_crit(X, rNew, k, d)
                if (loss < kmloss[i+1]):
                    if i < k - 2:
                        R[i+1] = rNew
                    kmloss[i+1] = loss
                    mergeidx[i] = [c1, c2]

    return R, kmloss, mergeidx


def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: (n) vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    k = kmloss.size
    maxLoss = kmloss[-1]
    c = np.arange(k)                # array for new indices of clusters after merging
    Z = np.zeros((k-1, 4))

    for i in range(k-1):
        Z[i, 0] = c[mergeidx[i, 0]]
        Z[i, 1] = c[mergeidx[i, 1]]
        c[mergeidx[i, 0]] = k + i
        c[mergeidx[i, 1]] = k + i
        Z[i, 2] = kmloss[i] / maxLoss
    
    plt.figure()
    dn = dendrogram(Z)
    plt.title("Dendrogramm of hierarch. agglom. Kmeans clustering for k = " + str(k))
    plt.xlabel('Cluster index')
    plt.ylabel('Normalized increase in loss function')
    plt.show()

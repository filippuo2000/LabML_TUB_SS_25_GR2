""" ps2_implementation.py


PUT YOUR NAME HERE:
Michał

Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution
"""
from scipy.io import loadmat
import os
from turtle import distance  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from matplotlib.patches import Ellipse


# Function by Gustavo. Needed for assignment 4,5,6
def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (n x d) data matrix with each datapoint in one row
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (k x d) matrix with each cluster center in one row
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

        #print('Number of iterations: ' + str(i+1))
        #print('Number of changed cluster memberships : ' + str(changes))
        #print('Loss function value : ' + str(loss))

        r = rNew

    #print()
    return mu, r, loss

# Helping function
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


### ASSIGNMENT 4 ###
def norm_pdf(X, mu, C):
    """ 
    Computes the multivariate Gaussian PDF at each row of X.

    Input:
      X  : (n × d) data matrix, each row is a datapoint
      mu : (d,)     mean vector
      C  : (d × d)  covariance matrix

    Output:
      pdf_vals : (n,)  PDF value for each data point
    """

    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    C = np.asarray(C, dtype=float)
    n, d = X.shape

    X_centered = X - mu   # (n, d)

    # 1) Ensure C is positive‐definite by attempting Cholesky.
    #    If it fails, add a scaled ridge until it succeeds.
    min_ridge = 1e-6
    max_tries = 10
    delta = min_ridge * (np.trace(C) / float(d) + 1e-16)
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(C)
            break
        except np.linalg.LinAlgError:
            C += delta * np.eye(d)
            delta *= 10.0
    else:
        raise np.linalg.LinAlgError(
            "Could not make C positive‐definite after adding ridge."
        )

    # 2) Solve L @ Y = X_centered.T  →  Y = L^{-1} @ (X_centered).T
    #    Then (x_i − mu)^T C^{-1} (x_i − mu) = sum(Y[:, i]^2).
    y = np.linalg.solve(L, X_centered.T)   # (d, n)
    quad = np.sum(y**2, axis=0)            # (n,) array of Mahalanobis terms

    # 3) Compute log of normalization constant:
    #    det(C) = det(L)^2 because C = L L^T and L is triangular.
    #    Hence, sqrt(det(C)) = det(L) = ∏ diag(L),
    #    so log(sqrt(det(C))) = sum(log(diag(L))).
    log_det_C_half = np.sum(np.log(np.diag(L)))
    log_norm_const = (d / 2.0) * np.log(2.0 * np.pi) + log_det_C_half

    # 4) Compute log‐PDF for each point:
    #    log_pdf[i] = −0.5 * quad[i] − log_norm_const
    log_pdf = -0.5 * quad - log_norm_const

    # 5) Clamp log_pdf to prevent exp(...) overflow.
    #    In high dimensions, log_pdf could be very large.
    MAX_LOG = 700.0
    log_pdf = np.minimum(log_pdf, MAX_LOG)

    # 6) Exponentiate to obtain the PDF values.
    pdf_vals = np.exp(log_pdf)

    return pdf_vals


### ASSIGNMENT 5 ###
def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ 
    Implements EM for Gaussian Mixture Models

    Input:
      X          : (n × d) data matrix with each datapoint in one row
      k          : number of clusters
      max_iter   : maximum number of iterations
      init_kmeans: whether k-means should be used for initialization
      eps        : when log-likelihood change < eps, terminate loop

    Output:
      pi     : (k,) array of mixture weights
      mu     : (k × d) array of cluster centers (one per row)
      sigma  : list of k covariance matrices (each d × d)
      loglik : log-likelihood after the final iteration
    """

    X = np.asarray(X, dtype=float)  # (n, d)
    n, d = X.shape

    # Initialization
    if init_kmeans:
        # Run kmeans from multiple random starts to find the best result
        iterations = 100
        k = 10
        best_loss = np.inf
        best_params = None
        for it in range(iterations):
            #print(it)
            mu, r, loss = kmeans(X, k)
            #print(f"loss: {loss}")
            if loss < best_loss:
                best_loss = loss
                best_params = (mu, r)
        mu_init = best_params[0]
        r = best_params[1]

        pi = np.bincount(r, minlength=k) / n
        sigma = []
        for j in range(k):
            pts = X[r == j]
            if pts.shape[0] > 1:
                Cj = np.cov(pts, rowvar=False)
            else:
                Cj = np.eye(d)
            Cj += 1e-6 * np.eye(d)  # add small ridge to ensure positive-definiteness
            sigma.append(Cj)
        mu = mu_init.copy()  # (k, d)
    else:
        pi = np.ones(k) / k
        idx = np.random.choice(n, size=k, replace=False)
        mu = X[idx, :].copy()    # pick k random points as initial means
        sigma = [np.eye(d) for _ in range(k)]

    loglik_old = -np.inf

    # EM loop
    for iteration in range(1, max_iter + 1):
        # E-step: compute p_ij = pi_j * N(x_i | mu_j, Sigma_j)
        pdfs = np.zeros((n, k), dtype=float)
        for j in range(k):
            pdf_j = norm_pdf(X, mu[j], sigma[j])  # (n,)
            pdfs[:, j] = pi[j] * pdf_j

        # Compute responsibilities
        denominator = np.sum(pdfs, axis=1, keepdims=True)  # (n, 1)
        gamma = pdfs / denominator                         # (n, k)

        # Compute log-likelihood
        loglik = np.sum(np.log(denominator))
        #print(f"Iteration {iteration:3d}, log-likelihood: {loglik:.6f}")

        # Check for convergence
        if np.abs(loglik - loglik_old) < eps:
            break
        loglik_old = loglik

        # M-step: update parameters
        n_k = gamma.sum(axis=0)           # (k,) effective number of points per cluster

        # Update mixture weights
        pi = n_k / float(n)

        # Update means: mu_j = (1 / n_k[j]) ∑_i gamma[i,j] * x_i
        mu = (gamma.T @ X) / n_k[:, None]  # (k, d)

        # Update covariances
        sigma = []
        for j in range(k):
            Xc = X - mu[j]  # (n, d)
            # Compute Σ_j = (1 / n_k[j]) ∑_i gamma[i,j] * (x_i - mu_j)(x_i - mu_j)^T
            Cj = (Xc.T * gamma[:, j]) @ Xc / n_k[j]  # (d, d)
            eps_j = 1e-3 * (np.trace(Cj) / d)
            Cj += eps_j * np.eye(d)  # add trace-based ridge to ensure positive-definiteness
            sigma.append(Cj)

    return pi, mu, sigma, float(loglik)


### ASSIGNMENT 6 ###
def plot_gmm_solution(X, mu, sigma):
    """
    Plots covariance ellipses for a GMM solution.

    Input:
      X     : (n × d) data matrix, each row is a datapoint
      mu    : (k × d) matrix of cluster centers (one per row)
      sigma : list of k covariance matrices (each d × d)
    """
    fig, ax = plt.subplots()

    # Scatter the data points
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)

    # Plot the cluster means
    ax.scatter(
        mu[:, 0], mu[:, 1],
        c='red', marker='x', s=100, lw=2, label='means'
    )

    # Plot one ellipse per Gaussian component
    for (m, cov) in zip(mu, sigma):
        # Eigen‐decomposition of covariance matrix
        vals, vecs = np.linalg.eigh(cov)

        # Sort eigenvalues in descending order
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Compute angle of the principal eigenvector (in degrees)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Width and height of ellipse: 2 * n_std * sqrt(eigenvalue)
        # Here n_std = 2 for roughly 95% confidence region
        width, height = 2 * 2 * np.sqrt(vals[:2])

        ell = Ellipse(
            xy=m,
            width=width, height=height,
            angle=angle,
            edgecolor='black',
            facecolor='none',
            lw=1.5
        )
        ax.add_patch(ell)

    ax.set_aspect('equal', 'datalim')
    ax.legend()
    plt.show()


def load_usps_data():
    """
    Load the 'usps.mat' file
    """
    base_dir = os.path.dirname(__file__)
    mat_path = os.path.join(base_dir, 'data', 'usps.mat')
    return loadmat(mat_path)


if __name__ == "__main__":
    data_2 = np.load("data/2_gaussians.npy", allow_pickle=True)
    data_5 = np.load("data/5_gaussians.npy", allow_pickle=True)
    usps_data = load_usps_data()
    X = data_5.T

    # Run EM from multiple random starts to find the best result
    R = 10
    k = 5
    best_loglik = -np.inf
    best_params = None
    for _ in range(R):
        pi_r, mu_r, sigma_r, loglik_r = em_gmm(X, k, init_kmeans=False)
        if loglik_r > best_loglik:
            best_loglik = loglik_r
            best_params = (pi_r, mu_r, sigma_r)
    
    plot_gmm_solution(X, best_params[1], best_params[2])

    

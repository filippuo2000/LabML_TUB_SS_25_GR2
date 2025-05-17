import numpy as np

def lle(X, m, tol, n_rule, k=None, epsilon=None):
    if not isinstance(X, np.ndarray):
        raise ValueError('For LLE an invalid input for parameter X was given')
    n, d = X.shape
    NBHs = np.ndarray

    if n_rule == 'knn':
        if k == None:
            raise ValueError('For LLE the method knn was chosen but no parameter k was given')
        NBHs = kNN(X, k)
    elif n_rule == 'eps-ball':
        if epsilon == None:
            raise ValueError('For LLE the method epsilon ball was chosen but no parameter epsilon was given')
        NBHs = epsBall(X, epsilon)
    else:
        raise ValueError('For LLE an invalid input for the method specification was given')
    
    if not is_graph_connected(NBHs, n):
        raise ValueError('Neighborhood graph not connected')
    
    return lleComp(NBHs, X, n, m, tol)

def kNN(X, k):
    Dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    SortIndc = np.argsort(Dist, axis=1)[:, 1:k+1]
    return SortIndc

def epsBall(X, epsilon):
    Dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    MaskIndc = (0 < Dist) & (Dist <= epsilon)
    indcList = [np.where(row)[0] for row in MaskIndc]
    return indcList

def is_graph_connected(NBHs, vertexCount):
    visited = np.full(vertexCount, False)
    toVisit = [0]
    detected = np.full(vertexCount, False)
    detected[0] = True

    while len(toVisit) != 0:
        v = toVisit.pop(0)
        if not visited[v]:
            visited[v] = True
            for n in NBHs[v]:
                if not detected[n]:
                    detected[n] = True
                    toVisit.append(n)

    return np.all(visited) 

def lleComp(NBHs, X, n, m, tol):
    W = np.zeros((n, n))

    for i in range(n):
        X_NB_shifted = X[NBHs[i]] - X[i]
        C = X_NB_shifted @ X_NB_shifted.T
        k = NBHs[i].size

        w = np.linalg.solve(C + np.eye(k) * tol, np.ones(k))
        w /= w.sum()
        W[i, NBHs[i]] = w

    M = np.eye(n) - W
    M = M.T @ M

    L, V = np.linalg.eigh(M)
    Z = V[:, 1:m+1]

    return Z

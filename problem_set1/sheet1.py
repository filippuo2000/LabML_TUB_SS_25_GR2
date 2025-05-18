import numpy as np
import matplotlib.pyplot as plt


####### Assignment 1: PCA #######
class PCA():
    def __init__(self, Xtrain):
       """
        Perform PCA on the training data Xtrain.

        Parameters
        ----------
        Xtrain : array-like of shape (n_samples, n_features)
            The training data.

        Attributes
        ----------
        C : ndarray of shape (n_features,)
            Data mean (center).
        U : ndarray of shape (n_features, n_features)
            Principal directions as columns, sorted by explained variance.
        D : ndarray of shape (n_features,)
            Principal values (variances) sorted in descending order.
        """
       # d-dimensional center of data
       self.C = np.mean(Xtrain, axis=0)
       # Center the data
       Xc = Xtrain - self.C
       # Compute Covariance matrix of the cetered data
       cov = np.cov(Xc.T)
       # Eigen-decomposition
       eigenvalues, eigenvectors = np.linalg.eig(cov)
       # Sort eigenvalues and eigenvectors in descending order
       order_of_importance = np.argsort(eigenvalues)[::-1] 
       # d × d matrix, which contains the principal directions
       self.U =  eigenvectors[:,order_of_importance]
       # vector of length d, which contains the principal values sorted in descending order
       self.D = eigenvalues[order_of_importance]

    def project(self, Xtest, m):
        """
        Project test data onto the first m principal components.

        Parameters
        ----------
        Xtest : array-like of shape (n_samples, n_features)
            New data to project.
        m : int
            Number of principal components to retain.

        Returns
        -------
        Z : ndarray of shape (n_samples, m)
            Projected data in principal component space.
        """
        # Validate dimensions
        if Xtest.shape[1] != self.C.shape[0]:
            raise ValueError("Xtest must have the same number of features as Xtrain.")
        if m < 1 or m > self.U.shape[1]:
            raise ValueError(f"m must be between 1 and {self.U.shape[1]}.")
        # Center the test data
        Xc = Xtest - self.C
        # Project onto first m principal directions
        Z = Xc @ self.U[:, :m]

        return Z
        
    def denoise(self, Xtest, m):
        """
        Denoise (reconstruct) test data using the first m principal components.

        Parameters
        ----------
        Xtest : array-like of shape (n_samples, n_features)
            New data to denoise.
        m : int
            Number of principal components to use for reconstruction.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_features)
            Reconstructed (denoised) data.
        """
        # First project the data
        Z = self.project(Xtest, m)
        # Reconstrut from the m principal components
        Y = Z @ self.U[:, :m].T + self.C
        return Y
    
####### Assignment 2: Gamma-Index #######
def gammaidx(X: np.ndarray, k: int):
    if k<=0:
        raise ValueError("k cannot be less or equal than 0, got k={k}")
    #1 calc the distances from a point to the all other points
    #2 choose top k closest points
    #3 calc the gamma idx for those chosen points
    # repeat for all the points in the dataset

    y = np.empty((X.shape[0]))
    for idx, point in enumerate(X):
        distances = np.sqrt(np.sum((X - point)**2, axis=1))
        top_idxs = list(np.argsort(distances)[:k+1])
        if idx in top_idxs:
            top_idxs.remove(idx)

        y[idx] = np.mean(distances[top_idxs])

    return y

####### Assignment 3: AUC #######
def auc(y_true: np.ndarray, y_pred:np.ndarray, plot=True) -> int:
    #1 sort all prediction scores
    #2 choose a prediction score as a threshold
    #3 calculate the TPR and FPR rates for this threshold
    #4 plot that point on the graph
    #5 repeat for all predictions

    # TPR = TP/(TP+FN), FPR=FP/(FP+TN)

    y_pred_idxs = np.argsort(y_pred)[::-1]

    FPRS = []
    TPRS = []

    positives = np.sum(y_true>0)
    negatives = len(y_true) - positives
    
    for pred_idx in y_pred_idxs:
        threshold = y_pred[pred_idx]
        preds = np.where(y_pred>=threshold, 1, -1)
        TP = np.sum((preds==y_true) & (y_true==1))
        FP = np.sum((preds!=y_true) & (y_true==-1))

        TPR = TP/positives
        FPR = FP/negatives

        FPRS.append(FPR)
        TPRS.append(TPR)

    if plot:
        plt.plot(FPRS, TPRS)
        plt.show()

    auc_score = np.trapezoid(TPRS, FPRS)

    return auc_score

####### Assignment 4: LLE #######
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
    AdjList = [v.tolist() for v in NBHs]
    for i in range(vertexCount):
        for j in AdjList[i]:
            AdjList[j].append(i)

    visited = np.full(vertexCount, False)
    toVisit = [0]
    detected = np.full(vertexCount, False)
    detected[0] = True

    while len(toVisit) != 0:
        v = toVisit.pop(0)
        if not visited[v]:
            visited[v] = True
            for n in AdjList[v]:
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
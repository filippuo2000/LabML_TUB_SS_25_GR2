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

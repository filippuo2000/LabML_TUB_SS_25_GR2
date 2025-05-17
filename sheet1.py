import numpy as np


#################################
##### PART 1: Implementation ####
#################################

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
    
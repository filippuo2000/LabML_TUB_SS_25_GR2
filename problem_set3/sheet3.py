""" ps3_implementation.py

PUT YOUR NAME HERE:
Filip Matysik


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import random
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from itertools import product

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    y_true (array-like): Array of true target values.
    y_pred (array-like): Array of predicted target values.

    Returns:
    float: The mean absolute error between the true and predicted values.

    Raises:
    ValueError: If the size of y_true and y_pred do not match.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if np.size(y_true) != np.size(y_pred):
        raise ValueError(f"y_true and y_pred matrices are not of an equal size, \
                         got y_pred: {np.size(y_pred), y_true: {np.size(y_true)}}")
    return np.mean(np.abs(y_true-y_pred))

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    """
    Perform cross-validation to find the best hyperparameters for a given model.
    Parameters:
    X : array-like
        Feature dataset.
    y : array-like
        Target variable.
    method : callable
        A callable that returns a model instance when called with parameters.
    params : dict
        A dictionary where keys are parameter names and values are lists of parameter values to try.
    loss_function : callable, optional
        A function to compute the loss (default is mean_absolute_error).
    nfolds : int, optional
        Number of folds for cross-validation (default is 10).
    nrepetitions : int, optional
        Number of times to repeat the cross-validation (default is 5).

    Returns:
    method : object
        The fitted model with the best hyperparameters and the associated cross-validated loss.
    """
    param_list = list(product(*list(params.values())))
    best_loss = np.inf
    best_params = param_list[0]

    for param_set in param_list:
        algo = method(*param_set)
        total_loss = 0

        for n in range(nrepetitions):
            # shuffle and split the data
            x_data, y_data = np.copy(X), np.copy(y)
            random_indices = list(range(len(y_data)))
            random.shuffle(random_indices) # in place operation
            x_data, y_data = x_data[random_indices], y_data[random_indices] 
            x_data, y_data = np.array_split(x_data, nfolds), np.array_split(y_data, nfolds)

            for k in range(nfolds):
                Xtest, Ytest = x_data[k], y_data[k]
                Xtrain = np.concatenate((*x_data[:k], *x_data[k+1:]), axis=0)
                Ytrain = np.concatenate((*y_data[:k], *y_data[k+1:]), axis=0)

                algo.fit(Xtrain, Ytrain)
                preds = algo.predict(Xtest)

                loss = loss_function(y_true=Ytest, y_pred=preds)
                total_loss += loss

        avg_loss = total_loss / (nrepetitions*nfolds)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = param_set

    best_classifier = method(*best_params)
    method = best_classifier.fit(X, y)
    method.cvloss = best_loss
        
    return method
  
class krr:
    """
    Kernel Ridge Regression (KRR) class for fitting and predicting using different kernels.

    Attributes:
        kernel (str): The type of kernel to use ('linear', 'polynomial', 'gaussian').
        kernelparameter (float): The parameter for the kernel (e.g., degree for polynomial, sigma for gaussian).
        regularization (float): The regularization constant C.
        X (np.ndarray): The training data.
        K (np.ndarray): The kernel matrix.
        alpha (np.ndarray): The coefficients for the predictions.
    """
class krr:
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
            """
            Fit the model to the training data.

            Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, n_features).
            y (numpy.ndarray): The target values of shape (n_samples,).
            kernel (str, optional): The kernel type to be used ('linear', 'polynomial', 'gaussian').
            kernelparameter (float, optional): The parameter for the kernel function (used for polynomial and gaussian).
            regularization (float, optional): The regularization constant C.

            Returns:
            self: Returns an instance of the model fitted on the full training set.
            """
            if kernel is not False:
                self.kernel = kernel
            if kernelparameter is not False:
                self.kernelparameter = kernelparameter
            if regularization is not False:
                self.regularization = regularization

            n, d = X.shape
            self.X = X

            if self.kernel == 'linear':
                self.K = self._linear(X, X) 
            elif self.kernel == 'polynomial':
                self.K = self._polynomial(X, X, self.kernelparameter)
            elif self.kernel == 'gaussian':
                self.K = self._gaussian(X, X, self.kernelparameter)

            if self.regularization == 0:
                self.regularization, _ = self._LOOCV(self.K, y)

            while True:
                try:
                    alpha = np.linalg.inv(self.K + self.regularization*np.identity(n))
                    break
                except np.linalg.LinAlgError:
                    print("Matrix still singular, adding more regularization.")
                    self.K += 1e-6 * np.identity(n)
                
            self.alpha = alpha @ y

            return self

    def predict(self, X):
            """
            Predict the output for the given input data using the specified kernel.

            Parameters:
            X (array-like): Input data for which predictions are to be made.

            Returns:
            array-like: Predicted output based on the input data and the kernel method.
            """
            if self.kernel == 'linear':
                kernel = self._linear(X, self.X) 
            elif self.kernel == 'polynomial':
                kernel = self._polynomial(X, self.X, self.kernelparameter)
            elif self.kernel == 'gaussian':
                kernel = self._gaussian(X, self.X, self.kernelparameter)

            return kernel @ self.alpha
    
    def _LOOCV(self, K: np.ndarray, y: np.ndarray):
            """
            Perform Leave-One-Out Cross-Validation (LOOCV) to find the best regularization parameter.

            This method computes the optimal regularization parameter by evaluating the mean squared error
            of predictions made using the kernel matrix K and the target values y. It iterates over a range
            of candidate values for the regularization parameter and selects the one that minimizes the error.

            Parameters:
            K (np.ndarray): The kernel matrix used for the model.
            y (np.ndarray): The target values corresponding to the data points.

            Returns:
            tuple: A tuple containing the best regularization parameter and the corresponding error.
            """
            eigvals, eigvecs = np.linalg.eigh(K)
            U = eigvecs
            L = np.diag(eigvals)
            I = np.identity(U.shape[0])

            # maybe they should be spaced in a circle, not in a line only
            cands = np.mean(L) * np.logspace(-3, 2, num=10)
            lowest_error = np.inf
            best_C = cands[0]

            for c in cands:
                S = ((U @ L) @ np.linalg.inv(L + c*I)) @ U.T
                S_y = S @ y
                err = np.mean(((y-S_y) / (1-np.diagonal(S)))**2)
                if err < lowest_error:
                    lowest_error = err
                    best_C = c
            return best_C, err

    def _linear(self, X, X_prime):
        return X @ X_prime.T

    def _polynomial(self, X, X_prime, d):
        return ((X @ X_prime.T) + 1)**d

    def _gaussian(self, X, X_prime, sigma):
        return np.exp(-(np.sum(X**2, axis=1)[:, None] -2*(X @ X_prime.T) + np.sum(X_prime**2, axis=1)[None, :]) / 2*(sigma**2))
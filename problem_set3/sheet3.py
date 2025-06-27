""" ps3_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


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
import tqdm
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import pickle



def zero_one_loss(y_true, y_pred):
    # type: (np.ndarray, np.ndarray) -> float
    ''' 
    Input:
        y_true: (n) vector of true targets
        y_pred: (n) vector of predicted targets 
    Output:
        zero-one-loss
    '''

    # print(y_true)
    # print(y_pred)
    mask = y_true != np.sign(y_pred)
    # print(mask)
    loss = np.mean(mask)
    # print(loss)
    return loss


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    # type: (np.ndarray, np.ndarray) -> float
    ''' 
    Input:
        y_true: (n) vector of true targets
        y_pred: (n) vector of predicted targets 
    Output:
        mean absolute error (MAE)
    '''

    n = y_true.size
    mae = np.absolute(y_pred - y_true).mean()
    return mae


class method_class:
    cvloss: float
    y_pred: float
    kernel: str
    kernelparameter: float | int
    regularization: float
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0): ...
    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False): ...
    def predict(self, X) -> np.ndarray: ...

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    # type: (np.ndarray, np.ndarray, type[method_class], dict, function[[np.ndarray, np.ndarray], float], int, int) -> type[method_class]
    ''' 
    Input:
        X: (n x d) matrix of data
        y: (n) vector of labels/targets
        method: class with functions fit(X, y) and predict(X)
        parameters: dictionary with parameter names as keys
        loss_function(y_true, y_pred): function handle to the loss function to be used, e.g. zero_one_loss
        nfolds: number of partitions
        nrepetitions: number of repetitions
    Output:
        method: ...
    '''
    # print('\n\n')

    if not all(hasattr(method, func) for func in ['fit', 'predict']):
        raise TypeError("'method' must be a class including functions 'fit' and 'predict'")

    n, d = X.shape
    part_size = n / nfolds
    param_names = list(params.keys())
    # print(params)
    # vals = [v.tolist() if isinstance(v, np.ndarray) else v for v in params.values()]
    vals = [list(v) for v in params.values()]
    # print(vals)
    param_val_combs = list(it.product(*vals))
    # print(param_val_combs)

    best_param_comb = param_val_combs[0]
    best_avg_loss = np.infty

    for param_values in param_val_combs:
        args = dict(zip(param_names, param_values))
        classifier = method(**args)
        avg_loss = 0.0

        for i in range(nrepetitions):
            for j in range(nfolds):
                part_start = int(np.ceil(part_size * j))
                part_end = int(np.ceil(part_size * (j + 1)))
                # print('s:e', part_start, part_end)

                X_test = X[part_start:part_end]
                y_test = y[part_start:part_end]
                X_train = np.delete(X, slice(part_start, part_end),axis=0)
                y_train = np.delete(y, slice(part_start, part_end),axis=0)

                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                
                l = loss_function(y_test, y_pred)
                avg_loss += l
                # print('l: ', l)

        avg_loss /= nrepetitions * nfolds 
        # print('avg_loss: ', avg_loss)
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_param_comb = param_values

    args = dict(zip(param_names, best_param_comb))
    classifier = method(**args)
    classifier.fit(X_train, y_train)
    classifier.cvloss = best_avg_loss

    # print(best_param_comb)
    # print('\n TODO: zero_one_loss + 1 (c) \n')
    return classifier


def polynomial_kernel(X1, X2, degree):
    # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    K = X1 @ X2.T
    if degree > -1:
        K = (K + 1.0)**degree
    return K


def gaussian_kernel(X1, X2, width):
    # type: (np.ndarray, np.ndarray, float) -> np.ndarray
    Dist = X1[:,None,:] - X2[None,:,:]
    Dist = np.sum(Dist**2, axis=2)
    K = np.exp(- Dist / (2 * width**2))
    return K


def LOOCV(K, y, ncandidates=500, log_var=np.exp(3)):
    # type: (np.ndarray, np.ndarray, int, float) -> float
    l, U = la.eigh(K)
    UTy = U.T @ y
    mean_l = l.mean()
    log_mean_l = np.log(mean_l)
    candidates = np.logspace(log_mean_l - log_var, log_mean_l + log_var, ncandidates)

    best_C = candidates[0]
    lowest_error = np.inf

    for C in candidates:
        llC = l / (l + C)
        LLCI = np.diag(llC)
        Sy = U @ LLCI @ UTy
        S_diag = (U**2 @ llC)

        error = (((y - Sy) / (1 - S_diag))**2).mean()
        if (error < lowest_error):
            lowest_error = error
            best_C = C

    return best_C

  
class krr():
    ''' your header here!
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        # type: (str, int | float, float) -> krr
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        # type: (np.ndarray, np.ndarray, str, int | float, float) -> krr
        ''' your header here!
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        if self.kernel=='linear':
            self.kernelparameter = -1
        
        if self.kernel=='linear' or self.kernel=='polynomial':
            self.kernel_function = polynomial_kernel
        elif self.kernel=='gaussian':
            self.kernel_function = gaussian_kernel
        else:
            raise TypeError("'kernel' argument has an unvalid value")
        
        K = self.kernel_function(X, X, self.kernelparameter)
        self.X_train = X

        C = float(self.regularization)
        if C==0:
            C = LOOCV(K, y)
            self.regularization = C
        
        n, d = X.shape
        # a = la.inv(K + C * np.eye(n)) @ y
        a = la.solve(K + C * np.eye(n), y)
        self.learn_func_coeffs = np.reshape(a, (a.size, 1))

        return self

    def predict(self, X):
        # type: (np.ndarray) -> np.ndarray
        ''' your header here!
        '''

        K = self.kernel_function(X, self.X_train, self.kernelparameter)
        # print(K.shape, self.learn_func_coeffs.shape)
        self.y_pred = K @ self.learn_func_coeffs

        return self.y_pred





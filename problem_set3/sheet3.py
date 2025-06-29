"""
Author: MICHAL SZCZEPANSKI
Date: 20.06.2025
"""

import numpy as np
import scipy.linalg as la
import itertools as it
from tqdm import tqdm
from scipy.linalg import eigh

# Import test data function
from ps3_tests import noisysincfunction


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error between true and predicted values.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Mean absolute error
        
    Raises:
        ValueError: If input arrays have different lengths
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must match.")
    return np.mean(np.abs(y_true - y_pred))


def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    """
    Perform k-fold cross-validation with multiple repetitions for hyperparameter tuning.
    
    This function evaluates different parameter combinations using cross-validation
    and returns the best performing model instance.
    
    Args:
        X (np.ndarray): Input features, shape (n_samples, n_features)
        y (np.ndarray): Target values, shape (n_samples,)
        method (class): Machine learning method class (e.g., krr)
        params (dict): Dictionary of parameter names and their possible values
        loss_function (callable): Loss function to evaluate model performance
        nfolds (int): Number of folds for cross-validation
        nrepetitions (int): Number of repetitions of cross-validation
        
    Returns:
        object: Best model instance with lowest cross-validation loss
    """
    n_samples = X.shape[0]
    fold_size = n_samples // nfolds
    all_combos = list(it.product(*params.values()))

    # Handle single parameter combination case
    if len(all_combos) == 1:
        combo = all_combos[0]
        inst = method(**dict(zip(params.keys(), combo)))
        avg_error_rate = 0.0
        
        # Perform cross-validation for single parameter set
        for _ in tqdm(range(nrepetitions), desc='Repetitions'):
            for j in tqdm(range(nfolds), desc='Folds', leave=False):
                # Calculate fold boundaries
                start = j * fold_size
                end = start + fold_size if j < nfolds - 1 else n_samples
                
                # Split data into training and validation sets
                Xtrain = np.concatenate((X[:start], X[end:]), axis=0)
                ytrain = np.concatenate((y[:start], y[end:]), axis=0)
                Xval = X[start:end]
                yval = y[start:end]
                
                # Train and evaluate model
                inst.fit(Xtrain, ytrain)
                ypred = inst.predict(Xval)
                avg_error_rate += loss_function(yval, ypred)
        
        # Store average cross-validation loss
        avg_error_rate /= (nfolds * nrepetitions)
        
        # Create final model with best parameters and train on full dataset
        final_model = method(**dict(zip(params.keys(), combo)))
        final_model.fit(X, y)
        final_model.cvloss = avg_error_rate
        print(f"\nAverage CV loss: {final_model.cvloss:.4f}")
        return final_model

    # Handle multiple parameter combinations case
    best_error_rate = float('inf')
    best_params = None
    total_iterations = len(all_combos) * nrepetitions * nfolds
    
    # Grid search with progress tracking
    with tqdm(total=total_iterations, desc='Cross-validation progress') as pbar:
        for p in all_combos:
            avg_error_rate = 0.0
            inst = method(**dict(zip(params.keys(), p)))
            
            # Perform cross-validation for current parameter combination
            for _ in range(nrepetitions):
                for j in range(nfolds):
                    # Calculate fold boundaries
                    start = j * fold_size
                    end = start + fold_size if j < nfolds - 1 else n_samples
                    
                    # Split data into training and validation sets
                    Xtrain = np.concatenate((X[:start], X[end:]), axis=0)
                    ytrain = np.concatenate((y[:start], y[end:]), axis=0)
                    Xval = X[start:end]
                    yval = y[start:end]
                    
                    # Train and evaluate model
                    inst.fit(Xtrain, ytrain)
                    ypred = inst.predict(Xval)
                    avg_error_rate += loss_function(yval, ypred)
                    
                    # Update progress bar with current parameters
                    param_str = ', '.join([f'{k}={v:.2e}' if isinstance(v, float) else f'{k}={v}'
                                         for k, v in zip(params.keys(), p)])
                    pbar.set_description(f'CV Progress ({param_str})')
                    pbar.update(1)
            
            # Check if current combination is best so far
            avg_error_rate /= (nfolds * nrepetitions)
            if avg_error_rate < best_error_rate:
                best_error_rate = avg_error_rate
                best_params = dict(zip(params.keys(), p))

    # Create a new instance with the best parameters and train it on full dataset
    final_model = method(**best_params)
    final_model.fit(X, y)
    final_model.cvloss = best_error_rate
    
    return final_model


class krr:
    """
    Kernel Ridge Regression (KRR) implementation.
    
    Supports multiple kernel types and automatic regularization parameter selection
    using efficient leave-one-out cross-validation.
    
    Attributes:
        kernel (str): Type of kernel ('linear', 'gaussian', 'polynomial')
        kernelparameter (float): Kernel-specific parameter (sigma for Gaussian, degree for polynomial)
        regularization (float): Regularization parameter (lambda). If 0, uses LOO-CV to find optimal value
        best_C (float): Optimal regularization parameter found by LOO-CV
        eigenvalues (np.ndarray): Eigenvalues of kernel matrix (for efficient LOO-CV)
        eigenvectors (np.ndarray): Eigenvectors of kernel matrix (for efficient LOO-CV)
        X_train (np.ndarray): Training input data
        y_train (np.ndarray): Training target data
        weights (np.ndarray): Learned weights for predictions
    """
    
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        """
        Initialize KRR model.
        
        Args:
            kernel (str): Kernel type - 'linear', 'gaussian', or 'polynomial'
            kernelparameter (float): Kernel parameter (sigma for Gaussian, degree for polynomial)
            regularization (float): Regularization parameter. If 0, will use LOO-CV to find optimal value
        """
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.best_C = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.X_train = None
        self.y_train = None
        self.weights = None

    def _compute_kernel_matrix(self, X1, X2=None):
        """
        Compute kernel matrix between X1 and X2.
        
        Args:
            X1 (np.ndarray): First set of data points, shape (n1, d)
            X2 (np.ndarray, optional): Second set of data points, shape (n2, d).
                                     If None, computes kernel matrix of X1 with itself
                                     
        Returns:
            np.ndarray: Kernel matrix of shape (n1, n2) or (n1, n1) if X2 is None
            
        Raises:
            ValueError: If kernel type is not supported
        """
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            # Linear kernel: K(x,y) = x^T y
            K = np.dot(X1, X2.T)
            
        elif self.kernel == 'gaussian':
            # Gaussian (RBF) kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))
            sigma = self.kernelparameter
            # Efficient computation using broadcasting
            K = np.exp(-np.linalg.norm(X1[:, None] - X2, axis=2) ** 2 / (2 * sigma ** 2))
            
        elif self.kernel == 'polynomial':
            # Polynomial kernel: K(x,y) = (x^T y + 1)^d
            degree = self.kernelparameter
            K = (np.dot(X1, X2.T) + 1) ** degree
            
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")
            
        return K

    def _find_optimal_C(self, K, y):
        """
        Find optimal regularization parameter using efficient leave-one-out cross-validation.
        
        This method searches over logarithmically spaced candidates around the mean
        of the eigenvalues of the kernel matrix.
        
        Args:
            K (np.ndarray): Kernel matrix, shape (n, n)
            y (np.ndarray): Target values, shape (n,)
            
        Returns:
            tuple: (best_C, best_error) - optimal regularization parameter and corresponding LOO-CV error
        """
        # Compute eigendecomposition of kernel matrix once
        self.eigenvalues, self.eigenvectors = eigh(K)
        
        # Handle numerical issues - ensure eigenvalues are non-negative
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-12)
        
        # Create regularization parameter candidates based on eigenvalue spectrum
        # Use more robust approach for polynomial kernels
        valid_eigenvals = self.eigenvalues[self.eigenvalues > 1e-10]
        
        if len(valid_eigenvals) == 0:
            # Fallback if no valid eigenvalues
            mean_eigenval = 1.0
        else:
            # Use median instead of mean for more robustness against outliers
            mean_eigenval = np.median(valid_eigenvals)
        
        # Ensure mean_eigenval is reasonable
        mean_eigenval = np.clip(mean_eigenval, 1e-6, 1e6)
        
        # Create more conservative search range for polynomial kernels
        if self.kernel == 'polynomial':
            # Polynomial kernels often need stronger regularization
            log_min = np.log10(mean_eigenval * 1e-2)
            log_max = np.log10(mean_eigenval * 1e4)
            # Use more candidates for better search
            C_candidates = np.logspace(log_min, log_max, 15)
        else:
            # Original range for other kernels
            log_min = np.log10(mean_eigenval * 1e-4)
            log_max = np.log10(mean_eigenval * 1e2)
            C_candidates = np.logspace(log_min, log_max, 10)
        
        # Find best regularization parameter
        best_error = np.inf
        best_C = C_candidates[0]
        
        for C in C_candidates:
            error = self._efficient_loocv_error(K, y, C)
            if np.isfinite(error) and error < best_error:
                best_error = error
                best_C = C
        
        # If no valid solution found, use a reasonable default
        if not np.isfinite(best_error):
            best_C = mean_eigenval * 0.1
            best_error = self._efficient_loocv_error(K, y, best_C)
            
        return best_C, best_error

    def _efficient_loocv_error(self, K, y, C):
        """
        Compute leave-one-out cross-validation error efficiently using eigendecomposition.
        
        This method uses the analytical formula for LOO-CV error:
        ε = (1/n) * Σ((y_i - [Sy]_i) / (1 - S_ii))^2
        where S = K(K + CI)^(-1) is the hat matrix.
        
        The computation is made efficient by using eigendecomposition:
        K = ULU^T, so K(K + CI)^(-1) = UL(L + CI)^(-1)U^T
        
        Args:
            K (np.ndarray): Kernel matrix, shape (n, n)
            y (np.ndarray): Target values, shape (n,)
            C (float): Regularization parameter
            
        Returns:
            float: LOO-CV error for the given regularization parameter
        """
        n = len(y)
        
        # Use precomputed eigendecomposition
        L = self.eigenvalues
        U = self.eigenvectors
        
        # Add numerical stability check
        if C <= 0:
            C = 1e-10
        
        # Compute L(L + CI)^(-1) efficiently with numerical stability
        L_reg_inv = L / (L + C)
        
        # Check for numerical issues
        if not np.all(np.isfinite(L_reg_inv)):
            return np.inf

        # Compute Sy = UL(L + CI)^(-1)U^T y
        Uty = np.dot(U.T, y)
        Sy = np.dot(U, L_reg_inv * Uty)  
        
        # Compute diagonal elements of hat matrix S = UL(L + CI)^(-1)U^T
        S_diag = np.sum(U**2 * L_reg_inv, axis=1)
        
        # More robust handling of numerical issues
        # Clip S_diag to prevent numerical issues
        S_diag = np.clip(S_diag, 0, 0.999)  # Ensure S_ii < 1
        
        # Compute LOO-CV error using the efficient formula
        numerator = (y - Sy)**2
        denominator = (1 - S_diag)**2
        
        # Handle potential division by zero more robustly
        valid_indices = denominator > 1e-10
        if not np.any(valid_indices):
            return np.inf
        
        # Use only valid indices for computation
        if np.all(valid_indices):
            loocv_error = np.mean(numerator / denominator)
        else:
            # Weighted average giving more weight to reliable predictions
            weights = denominator[valid_indices]
            loocv_error = np.average(numerator[valid_indices] / denominator[valid_indices], 
                                    weights=weights)
        
        # Additional check for numerical stability
        if not np.isfinite(loocv_error) or loocv_error < 0:
            return np.inf
            
        return loocv_error

    def fit(self, X, y):
        """
        Fit the KRR model to training data.
        
        If regularization parameter is 0, automatically finds optimal value using
        efficient leave-one-out cross-validation.
        
        Args:
            X (np.ndarray): Training input features, shape (n_samples, n_features)
            y (np.ndarray): Training target values, shape (n_samples,)
        """
        # Store training data
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix for training data
        K = self._compute_kernel_matrix(X)
        
        # Determine regularization parameter to use
        if self.regularization == 0:
            # Use efficient LOO-CV to find optimal regularization parameter
            self.best_C, best_error = self._find_optimal_C(K, y)
            regularization_to_use = self.best_C
        else:
            # Use provided regularization parameter
            regularization_to_use = self.regularization
        
        # Update regularization parameter for consistency
        self.regularization = regularization_to_use
        
        # Solve for weights: (K + λI)α = y
        self.weights = la.solve(K + regularization_to_use * np.eye(len(X)), y)

    def predict(self, X):
        """
        Make predictions on new data points.
        
        Args:
            X (np.ndarray): Test input features, shape (n_test, n_features)
            
        Returns:
            np.ndarray: Predicted values, shape (n_test,)
            
        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if self.X_train is None or self.weights is None:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Compute kernel matrix between test and training points
        kernel_matrix = self._compute_kernel_matrix(X, self.X_train)
        
        # Return predictions: f(x) = Σ α_i K(x, x_i)
        return np.dot(kernel_matrix, self.weights)
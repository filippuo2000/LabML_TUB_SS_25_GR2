import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from torch.nn import Module, Parameter, ParameterList
from torch.optim import SGD
from tqdm import trange



class neural_network(Module):
    def __init__(self, layers=[2,100,2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = ParameterList([
            Parameter(scale*torch.randn(m, n)) 
            for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([
            Parameter(scale*torch.randn(n))
            for n in layers[1:]])

        self.p = p
        self.lr = lr
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):
        """
        Parameters:
            X: input tensor (batch_size, input_dim)
            W: weight tensor (input_dim, output_dim)
            b: bias tensor (output_dim,)
        Returns:
            A: activated output tensor (batch_size, output_dim)
        """
        Z = X @ W + b
        if self.train and self.p is not None:
            mask = torch.bernoulli(
                torch.full((1, Z.shape[1]), 1.0 - self.p, device = Z.device, dtype=Z.dtype)
            )
            A = torch.clamp(Z, min=0.0)
            A = A * mask
        else:
            scale = (1 - self.p) if self.p is not None else 1.0
            A = torch.clamp(scale * Z, min=0.0)
        return A

    def softmax(self, X, W, b):
        """
        Parameters:
            X: input tensor (batch_size, input_dim)
            W: weight tensor (input_dim, output_dim)
            b: bias tensor (output_dim,)
        Returns:
            Y: softmax activated output tensor (batch_size, output_dim)
        """
        Z = X @ W + b
        # We should subtract the max for numerical stability
        Z = Z - Z.max(dim=1, keepdim=True).values
        expZ = torch.exp(Z)
        Y = expZ / expZ.sum(dim=1, keepdim=True)
        return Y

    def forward(self, X):
        """
        Parameters:
            X: input tensor (batch_size, input_dim)
        Returns:
            Y: output tensor after passing through the network (batch_size, output_dim)
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float, device=self.weights[0].device)
        else:
            X = X.float()
        Z = X.float()
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = self.relu(Z, W, b)
        Y = self.softmax(Z, self.weights[-1], self.biases[-1])
        return Y

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        """
        Parameters:
            ypred: predicted output tensor (batch_size, output_dim)
            ytrue: true output tensor (batch_size, output_dim)
        Returns:
            loss: computed loss value (scalar)
        """
        n = ypred.shape[0]
        # cross-entropy loss L(y_pred, y_true)
        cross_entropy = - (ytrue * torch.log(ypred + 1e-8)).sum() / n   # adding small regularization to avoid log(0)
        # weight decay regularization
        decay = 0.0
        for W in self.weights:
            decay += torch.sum(W * W)     
        return cross_entropy + self.lam * decay

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):  # Fixed: default plot=False
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.floor(.9 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in trange(nsteps, desc="Training", unit="it"):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(Ltrain, label='Training loss')
            plt.plot(Lval, label='Validation loss')
            plt.plot(Aval, label='Validation acc')
            plt.legend()
            plt.title('Training Progress')
            plt.xlabel('Steps')
            plt.ylabel('Loss/Accuracy')
            plt.show()


class svm_qp():
    """ Support Vector Machines via Quadratic Programming """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
    
    def _kernel_function(self, x, y):
        if self.kernel == 'linear':
            return np.dot(x, y)
        elif self.kernel == 'rbf' or self.kernel == 'gaussian':
            # gamma = kernelparameter
            diff = x - y
            return np.exp(-self.kernelparameter * np.dot(diff, diff))
        elif self.kernel == 'poly':
            # degree = kernelparameter
            return (1.0 + np.dot(x, y)) ** self.kernelparameter
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
           
    def fit(self, X, Y):
        """
        Solve the SVM dual via QP
            minimize (1/2) alpha^T P alpha - 1^T alpha
            subject to: 0 <= alpha_i <= C, sum_i alpha_i y_i = 0
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).flatten()
        n_samples = X.shape[0]        

        # compute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        
        # Setup QP matrices
        P = np.outer(Y, Y) * K
        q = -np.ones(n_samples)

        # constraints 0 <= alpha_i <= C | Gx <= h
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples)))

        # equality constraint sum_i alpha_i y_i = 0 | Ax = b
        A = Y.reshape(1, -1)
        b = np.zeros(1)

        # Solve QP problem
        try:
            # Add small regularization to P for numerical stability
            P_reg = P + 1e-12 * np.eye(n_samples)
            sol = qp(cvxmatrix(P_reg, tc='d'),
                     cvxmatrix(q, tc='d'),
                     cvxmatrix(G, tc='d'),
                     cvxmatrix(h, tc='d'),
                     cvxmatrix(A, tc='d'),
                     cvxmatrix(b, tc='d'))
            if sol['status'] != 'optimal':
                print(f"Warning: QP solver status: {sol['status']}")
            alpha = np.array(sol['x']).flatten()
        except Exception as e:
            print(f"QP solver failed: {e}")
            return

        # Clean up numerical errors
        alpha = np.maximum(alpha, 0)  # Ensure non-negative
        alpha[alpha > self.C] = self.C  # Ensure <= C
        
        # Use adaptive tolerance based on the range of alpha values
        max_alpha = np.max(alpha)
        tol = max(1e-6, max_alpha * 1e-6)  # Adaptive tolerance
        
        # Support vectors have non-zero alpha
        sv_mask = alpha > tol
        
        # Store all support vectors
        self.alpha_sv = alpha[sv_mask]
        self.X_sv = X[sv_mask]
        self.Y_sv = Y[sv_mask]
        
        # Store all alphas for debugging
        self.alpha_all = alpha

        # Count different types of support vectors
        margin_sv_mask = (alpha > tol) & (alpha < self.C - tol)
        self.margin_sv_mask = margin_sv_mask

        # Compute bias using margin support vectors
        if np.any(margin_sv_mask):
            # Use margin support vectors for bias calculation
            margin_indices = np.where(margin_sv_mask)[0]
            bs = []
            for idx in margin_indices:
                s = 0.0
                for j in range(len(self.alpha_sv)):
                    s += self.alpha_sv[j] * self.Y_sv[j] * self._kernel_function(self.X_sv[j], X[idx])
                bs.append(Y[idx] - s)
            self.b = np.mean(bs)
        else:
            self.b = 0.0

    def predict(self, X):
        if self.alpha_sv is None:
            raise ValueError("Model not fitted yet")
            
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        # Compute decision function for each sample
        for i in range(n_samples):
            s = 0.0
            for j in range(len(self.alpha_sv)):
                s += self.alpha_sv[j] * self.Y_sv[j] * self._kernel_function(self.X_sv[j], X[i])
            y_pred[i] = s + self.b
        return y_pred


class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'

        # Handle different kernel parameters to match custom implementation
        if kernel == 'rbf':
            gamma = kernelparameter
        elif kernel == 'poly':
            degree = int(kernelparameter)
            gamma = 'scale'
        else:
            gamma = 'scale'
            degree = 3

        self.clf = sklearn.svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma if kernel == 'rbf' else 'scale',
            degree=degree if kernel == 'poly' else 3,
            coef0=1.0 if kernel == 'poly' else 0.0,
            tol=1e-4  # Match tolerance with custom implementation
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        # Store support vectors correctly
        self.X_sv = self.clf.support_vectors_
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, y, model, title="Decision Boundary"):
    """
    Plot decision boundary for 2D classification problems
    
    Parameters:
        X: input data (N x 2)
        y: labels 
        model: fitted classifier with predict() method
        title: plot title
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    assert X.shape[1] == 2, "X must be Nx2"

    # 1) data points
    plt.figure(figsize=(8,6))
    face = ['white','black', 'lightgray'][:len(classes)]
    for i, cls in enumerate(classes):
        plt.scatter(
            X[y == cls,0], X[y == cls,1],
            edgecolor='k', facecolor=face[i], s=60,
            label=f'Class {cls}', alpha=0.8
        )

    # 2) grid
    margin = 0.05
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin

    xx = np.linspace(x_min-dx, x_max+dx, 300)
    yy = np.linspace(y_min-dy, y_max+dy, 300)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    # 3) decision function
    vals = model.predict(grid)
    if vals.ndim>1 and vals.shape[1]==2:
        # multiclass-style output
        Z = (vals[:,1] - vals[:,0]).reshape(XX.shape)
    else:
        Z = vals.reshape(XX.shape)

    # 4) draw background and boundary
    plt.contourf(XX, YY, Z, levels=[-1e9,0,1e9],
                 colors=('lightcoral','lightskyblue'), alpha=0.3)
    plt.contour(XX, YY, Z, levels=[0], colors='k', linewidths=2)

    # 5) only if model has .X_sv attribute, mark support vectors
    if hasattr(model, 'X_sv') and model.X_sv is not None:
        sv = np.asarray(model.X_sv)
        if sv.ndim==2 and sv.shape[1]==2:
            plt.scatter(
                sv[:,0], sv[:,1],
                marker='x', s=150, color='red',
                linewidths=3, label='Support Vectors'
            )

    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    # 1) Generate moons data
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    y = np.where(y == 0, -1, 1)   # convert labels {0,1} → {-1,+1}

    # 2) Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3) Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # 4) Fit custom QP SVM
    svm_custom = svm_qp(kernel='rbf', kernelparameter=1.0, C=1.0)
    svm_custom.fit(X_train, y_train)
    acc_custom = np.mean(np.sign(svm_custom.predict(X_test)) == y_test)
    print(f"Custom SVM test accuracy: {acc_custom:.3f}")

    # 5) Fit scikit-learn SVM
    svm_sk = svm_sklearn(kernel='rbf', kernelparameter=1.0, C=1.0)
    svm_sk.fit(X_train, y_train)
    acc_sk = np.mean(np.sign(svm_sk.predict(X_test)) == y_test)
    print(f"Sklearn SVM test accuracy: {acc_sk:.3f}")

    plot_boundary_2d(X_train, y_train, svm_custom, title="Custom QP SVM on Moons")


    plot_boundary_2d(X_train, y_train, svm_sk, title="Sklearn SVM on Moons")


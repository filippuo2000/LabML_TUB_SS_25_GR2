import numpy as np
import matplotlib.pyplot as plt

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
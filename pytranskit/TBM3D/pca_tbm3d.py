import numpy as np
from sklearn.metrics import balanced_accuracy_score

class SimplePCA:
    def __init__(self, n_components=None):
        """
        n_components: n_components
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, S, _ = np.linalg.svd(X_centered @ X_centered.T, full_matrices=False)
        # print(S)
        V = (X_centered.T @ U) / np.sqrt(S + 1e-10)
        Vt = V.T
        explained_variance = (S) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = explained_variance / explained_variance.sum()
        
        if self.n_components < 1:
            cumulative = np.cumsum(self.explained_variance_ratio_)
            self.k_ = np.searchsorted(cumulative, self.n_components) + 1
        else:
            self.k_ = self.n_components
        
        self.components_ = Vt[:self.k_]
        
        return X_centered @ self.components_.T
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components_ + self.mean_

def predict_score(testscore, valscore, vallabel):
    '''
    find_best_threshold
    and predict_score
    '''

    best_thresh = None
    best_bacc = -1
    best_direction = None  # 'greater' or 'less'

    thresholds = np.unique(valscore)
    for t in thresholds:
        pred_greater = (valscore >= t).astype(int)
        bacc_greater = balanced_accuracy_score(vallabel, pred_greater)

        if bacc_greater > best_bacc:
            best_bacc = bacc_greater
            best_thresh = t
            best_direction = 'greater'

        pred_less = (valscore <= t).astype(int)
        bacc_less = balanced_accuracy_score(vallabel, pred_less)

        if bacc_less > best_bacc:
            best_bacc = bacc_less
            best_thresh = t
            best_direction = 'less'

    if best_direction == 'greater':
        test_pred = (testscore >= best_thresh).astype(int)
    else:
        test_pred = (testscore <= best_thresh).astype(int)

    return best_thresh, best_direction, best_bacc, test_pred
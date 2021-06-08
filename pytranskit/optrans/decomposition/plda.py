import warnings
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

from ..utils import check_array


class PLDA(BaseEstimator):
    """
    Penalized Linear Discriminant Analysis.

    This is both a dimensionality reduction method and a linear classifier.

    Parameters
    ----------
    alpha : scalar (default=1.)
        Parameter that controls the proportion of LDA vs PCA.
        If alpha=0, PLDA functions like LDA. If alpha is large, PLDA functions
        more like PCA.
    n_components : int or None (default=None)
        Number of components to keep. If n_components is not set, all
        components are kept: n_components == min(n_samples, n_features).

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Axes in the feature space. The components are sorted by the explained
        variance.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : array, shape(n_components,)
        Proportion of variance explained by each of the selected components.
        If n_components is not set then all components are stored and the sum
        of explained variance ratios is equal to 1.0.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        The number of components.
    coef_ : array, shape (n_features,) or (n_classes, n_features)
        Weight vector(s).
    intercept_ : array, shape (n_features,)
        Intercept term.
    class_means_ : array, shape (n_classes, n_features)
        Class means, estimated from the training set.
    classes_ : array, shape (n_classes,)
        Unique class labels.

    References
    ----------
    W. Wang et al. Penalized Fisher Discriminant Analysis and its Application
    to Image-Based Morphometry. Pattern Recognit. Lett., 32(15):2128-35, 2011
    """

    def __init__(self, alpha=1., n_components=None):
        self.is_fitted = False
        self.alpha = alpha
        self.n_components_ = n_components


    def _check_is_fitted(self):
        if not self.is_fitted:
            raise AssertionError("The fit function has not been "
                                 "called yet. Call 'fit' before using "
                                 "this method".format(type(self).__name__))


    def _class_means(self, X, y):
        """
        Compute class means.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.
        y : array, shape (n_samples,)
            Target values.

        Returns
        -------
        means : array, shape (n_features,)
            Class means.
        """
        means = []
        classes = np.unique(y)
        for group in classes:
            Xg = X[y==group,:]
            means.append(Xg.mean(0))
        return np.asarray(means)


    def _solve_eigen(self, X, y):
        """
        Eigenvalue solver.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.
        """
        # Data dimensions
        n_samp, n_feat = X.shape

        # Initialize between-class and within-class scatter matrices
        S_B = np.zeros((n_feat,n_feat))
        S_W = np.zeros((n_feat,n_feat))

        # Loop over each class label
        for i,yi in enumerate(self.classes_):
            # Get indices of data in this class
            ind = (y==yi)
            ni = ind.sum()

            # Class mean and centered class data
            Xi_cent = X[ind] - np.tile(self.class_means_[i], (ni,1))

            # Get difference of means as column vector (so transpose will work)
            diff = (self.class_means_[i]-self.mean_).reshape((n_feat,1))

            # Update scatter matrices
            S_W += Xi_cent.T.dot(Xi_cent)
            S_B += float(ni) * diff.dot(diff.T)

        # PLDA solution is given by generalized eigenvalue decomposition:
        # S_T w = lambda*(S_W + alpha*I) w, where S_T = S_B + S_W
        alpha_I = self.alpha*np.identity(n_feat)
        evals, evecs = eigh(S_B+S_W, b=S_W+alpha_I)

        # Sort the eigenvalues (eigenvectors) in descending order
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals)
                                                 )[::-1][:self.n_components_]
        evecs = evecs[:,np.argsort(evals)[::-1]]
        evecs /= np.linalg.norm(evecs, axis=0)

        # Set the components for transforming data, and coefficients and
        # intercept for classification
        self.components_ = evecs[:,:self.n_components_].T
        self.coef_ = np.dot(self.class_means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(self.class_means_.dot(self.coef_.T))
        return


    def fit(self, X, y):
        """
        Fit PLDA model according to the given training data and parameters.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.
        """
        # Check input arrays
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)
        y = check_array(y.astype(int), ndim=1, dtype='numeric',
                        force_all_finite=True)

        # Check input arrays are same length
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must be the same: "
                             "{} vs {}".format(X.shape[0], y.size))

        # Check that n_components does not exceed maximum possible
        max_components = min(X.shape)
        if self.n_components_ is None:
            self.n_components_ = max_components
        elif self.n_components_ > max_components:
            self.n_components_ = max_components
            warnings.warn("n_components exceeds maximum possible components. "
                          "Setting n_components = {}".format(max_components))

        # Set useful data attributes
        self.classes_ = np.unique(y)
        self.mean_ = X.mean(axis=0)
        self.class_means_ = self._class_means(X, y)

        self._solve_eigen(X, y)

        # Adjust coefficients and intercept for binary classification problems
        if self.classes_.size == 2:
            self.coef_ = np.array(self.coef_[1,:] - self.coef_[0,:], ndmin=2)
            self.intercept_ = np.array(self.intercept_[1]-self.intercept_[0],
                                       ndmin=1)

        # Transform data so we can get the explained variance
        self.explained_variance_ = np.dot(X, self.components_.T).var(axis=0)

        self.is_fitted = True
        return


    def transform(self, X):
        """
        Transform data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        self._check_is_fitted()

        # Check input arrays
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)

        # Check number of features is correct
        n_feat = self.components_.shape[1]
        if X.shape[1] != n_feat:
            raise ValueError("X has {} features per sample."
                             "Expecting {}".format(X.shape[1], n_feat))

        # Transform data
        X_new = np.dot(X, self.components_.T)

        return X_new[:,:self.n_components_]


    def fit_transform(self, X, y):
        """
        Fit the model with X and transform X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)


    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        Note: If n_components is less than the maximum, information will be
        lost, so reconstructed data will not exactly match the original data.

        Parameters
        ----------
        X : array shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Data transformed back into original space.
        """
        self._check_is_fitted()

        X = check_array(X, ndim=2)

        # Check data dimensions
        if X.shape[1] != self.n_components_:
            raise ValueError("X has {} features per sample."
                             "Expecting {}".format(X.shape[1],
                             self.n_components_))

        # Inverse transform
        X_original = self.mean_ + np.dot(X, self.components_)

        return X_original


    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        scores : array, shape=(n_samples,) if n_classes == 2
                 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        self._check_is_fitted()

        # Check input array
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)

        # Check number of features is correct
        n_feat = self.components_.shape[1]
        if X.shape[1] != n_feat:
            raise ValueError("X has {} features per sample."
                             "Expecting {}".format(X.shape[1], n_feat))

        scores = np.dot(X, self.coef_.T) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores


    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples,)
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            ind = (scores > 0).astype(np.int)
        else:
            ind = scores.argmax(axis=1)

        return self.classes_[ind]


    def predict_proba(self, X):
        """
        Estimate probability.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(self.classes_) == 2:
            # Binary case
            return np.column_stack([1-prob, prob]).T
        else:
            # One-vs-rest normalization
            prob /= prob.sum(axis=1).reshape((prob.shape[0],-1))
            return prob


    def predict_log_proba(self, X):
        """
        Estimate log probability.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))


    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Test samples.
        y : array, shape (n_samples,)
            True labels for X.
        sample_weight : array, shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        # Check input arrays
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)
        y = check_array(y.astype(int), ndim=1, dtype='numeric',
                        force_all_finite=True)

        # Check input arrays are same length
        if X.shape[0] != y.size:
            raise ValueError("Number of samples in X and y must be the same: "
                             "{} vs {}".format(X.shape[0], y.size))

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


    def predict_transformed(self, X_trans):
        """
        Predict class labels for data that have already been transformed by
        self.transform(X).

        This is useful for plotting classification boundaries.
        Note: Due to arithemtic discrepancies, this may return slightly
        different class labels to self.predict(X).

        Parameters
        ----------
        X_trans : array, shape (n_samples, n_components)
            Test samples that have already been transformed into PLDA space.

        Returns
        -------
        y : array, shape (n_samples,)
            Predicted class labels for X_trans.
        """
        self._check_is_fitted()

        # Check input array
        X_trans = check_array(X_trans, ndim=2)

        # Make sure this is a set of transformed data
        if X_trans.shape[1] != self.n_components_:
            raise ValueError("Number of features in X_trans must match "
                             "n_components: {}".format(self.n_components_))

        # Transform class means into PLDA space
        mean_trans = self.transform(self.class_means_)

        # Initialize useful values
        n_samples = X_trans.shape[0]
        n_classes = mean_trans.shape[0]
        dists = np.zeros((n_samples,n_classes))

        # Compute the distance between each data sample and each class mean
        for i,mean in enumerate(mean_trans):
            dists[:,i] = np.linalg.norm(X_trans-np.tile(mean,(n_samples,1)),
                                        axis=1)

        # Classification based on shortest distance to class mean
        ind = dists.argmin(axis=1)

        return self.classes_[ind]

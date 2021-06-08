import numpy as np
from sklearn.cross_decomposition import CCA as CanonCorr

from ..utils import check_array


class CCA():
    """
    Canonical Correlation Analysis.

    This is a wrapper for scikit-learn's CCA class, which allows it to be used
    in a similar manner to PLDA and PCA.

    Parameters
    ----------
    n_components : int (default=1)
        Number of components to keep.
    scale : bool (default=True)
        Whether to scale the data?
    max_iter : int (default=500)
        The maximum number of iterations of the NIPALS inner loop.
    tol : float (default=1e-6)
        The tolerance used in the iterative algorithm.
    copy : bool (default=True)
        Whether the deflation be done on a copy. Let the default value to True
        unless you don’t care about side effects.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        X block weights vectors.
    components_y_ : array, shape (n_components, n_targets)
        Y block weights vectors.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected weights for
        the X data.
    explained_variance_y_ : array, shape (n_components,)
        The amount of variance explained by each of the selected weights for
        the Y data.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean of X, estimated from the training set.
    mean_y_ : array, shape (n_targets,)
        Per-feature empirical mean of Y, estimated from the training set.
    n_components_ : int
        The number of components.

    References
    ----------
    [scikit-learn's documentation on CCA]
    (http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html)
    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.
    """
    def __init__(self, n_components=1, scale=True, max_iter=500, tol=1e-6,
                 copy=True):
        self.is_fitted = False
        self.n_components_ = n_components
        self.cca = CanonCorr(n_components=n_components, scale=scale,
                             max_iter=max_iter, tol=tol, copy=copy)
        return


    def _check_is_fitted(self):
        if not self.is_fitted:
            raise AssertionError("The fit function has not been "
                                 "called yet. Call 'fit' before using "
                                 "this method".format(type(self).__name__))
        return


    def fit(self, X, Y):
        """
        Fit model to data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        Y : array, shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        """
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)
        Y = check_array(Y, ndim=2, dtype='numeric', force_all_finite=True)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must be the same: "
                             "{} vs {}".format(X.shape[0], Y.shape[0]))

        if self.n_components_ > X.shape[1]:
            raise ValueError("n_components exceeds number of features in X: "
                             "{} > {}".format(self.n_components_, X.shape[1]))

        if self.n_components_ > Y.shape[1]:
            raise ValueError("n_components exceeds number of targets in Y: "
                             "{} > {}".format(self.n_components_, Y.shape[1]))

        self.cca.fit(X, Y)

        self.components_ = self.cca.x_weights_.T
        self.components_y_ = self.cca.y_weights_.T
        self.mean_ = self.cca.x_mean_
        self.mean_y_ = self.cca.y_mean_

        # Get the explained variance of the transformed data
        self.explained_variance_ = self.cca.x_scores_.var(axis=0)
        self.explained_variance_y_ = self.cca.y_scores_.var(axis=0)

        self.is_fitted = True
        return


    def transform(self, X, Y=None):
        """
        Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input X data.
        Y : array, shape (n_samples, n_targets) or None (default=None)
            Input Y data. If Y=None, then only the transformed X data are
            returned.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed X data.
        Y_new : array, shape (n_samples, n_components)
            Transformed Y data. If Y=None, only X_new is returned.
        """
        self._check_is_fitted()

        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)

        if Y is None:
            return self.cca.transform(X, Y=None, copy=True)
        else:
            Y = check_array(Y, ndim=2, dtype='numeric', force_all_finite=True)
            X_new, Y_new = self.cca.transform(X, Y=Y, copy=True)

            # If n_components=1, reshape Y_new so it is 2D
            if self.n_components_ == 1:
                n_samples = Y_new.shape[0]
                Y_new = Y_new.reshape((n_samples,1))
            return X_new, Y_new


    def fit_transform(self, X, Y):
        """
        Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        Y : array, shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed X data.
        Y_new : array, shape (n_samples, n_components)
            Transformed Y data.
        """
        self.fit(X, Y)
        return self.transform(X, Y=Y)


    def score(self, X, Y):
        """
        Return Pearson product-moment correlation coefficients for each
        component.

        The values of R are between -1 and 1, inclusive.

        Note: This is different from sklearn.cross_decomposition.CCA.score(),
        which returns the coefficient of determination of the prediction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input X data.
        Y : array, shape (n_samples, n_targets) or None (default=None)
            Input Y data.

        Returns
        -------
        score : float or array, shape (n_components,)
            Pearson product-moment correlation coefficients. If n_components=1,
            a single value is returned, else an array of correlation
            coefficients is returned.
        """
        x_trans, y_trans = self.transform(X, Y)

        score = np.zeros(self.n_components_)
        for i in range(self.n_components_):
            score[i] = np.corrcoef(x_trans[:,i], y_trans[:,i])[0,1]

        if self.n_components_ == 1:
            return score[i]
        else:
            return score


    def inverse_transform(self, X, Y=None):
        """
        Transform data back to its original space.

        Note: This is not exact!

        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            Transformed X data.
        Y : array, shape (n_samples, n_components) or None (default=None)
            Transformed Y data. If Y=None, only the X data are transformed back
            to the original space.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            X data transformed back into original space.
        Y_original : array, shape (n_samples, n_targets)
            Y data transformed back into original space. If Y=None, only
            X_original is returned.
        """
        self._check_is_fitted()

        # Check X is in transformed space
        X = check_array(X, ndim=2, dtype='numeric', force_all_finite=True)
        if X.shape[1] != self.n_components_:
            raise ValueError("X has {} features per sample."
                             "Expecting {}".format(X.shape[1],
                             self.n_components_))

        # Invert X into original space
        X_original = np.dot(X, self.components_) + self.mean_

        if Y is None:
            return X_original
        else:
            # Check Y is in transformed space
            Y = check_array(Y, ndim=2, dtype='numeric', force_all_finite=True)
            if Y.shape[1] != self.n_components_:
                raise ValueError("Y has {} features per sample."
                                 "Expecting {}".format(Y.shape[1],
                                 self.n_components_))

            # Invert Y into original space
            Y_original = np.dot(Y, self.components_y_) + self.mean_y_

            return X_original, Y_original

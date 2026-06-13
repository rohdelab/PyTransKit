import numpy as np
from scipy.optimize import linprog


class DiscreteLOT:
    """
    Discrete Linear Optimal Transport (LOT) transform.

    This class computes the LOT embedding of a target measure (X1, a1)
    with respect to a reference measure (X0, a0).
    
    Authors
    -------
    Mohammad Shifat-E-Rabbi 
    Adapted for pytranskit and research workflow.

    Original framework inspired by:
    Kolouri, Soheil et al. (Optimal Transport methods)

    Date
    ----
    Created: June 2026

    Parameters
    ----------
    normalize : bool, optional (default=True)
        Whether to normalize input weights a0 and a1.
    solver : str, optional (default="highs")
        Linear programming solver to use.
    """

    def __init__(self, normalize=True, solver="highs"):
        self.normalize = normalize
        self.solver = solver

        # Will be set after fit
        self.X0 = None
        self.a0 = None
        self.fitted_ = False

    # -------------------------
    # Internal utilities
    # -------------------------
    def _prepare_inputs(self, X, a):
        X = np.asarray(X)
        a = np.asarray(a)

        if X.ndim == 1:
            X = X[:, None]

        if self.normalize:
            a = a / np.sum(a)

        return X, a

    def _compute_cost(self, X0, X1):
        return ((X0[:, None, :] - X1[None, :, :]) ** 2).sum(axis=-1)

    def _solve_ot(self, C, a0, a1):
        N0 = len(a0)
        N1 = len(a1)

        A_eq = np.zeros((N0 + N1, N0 * N1))

        # Row constraints (source)
        for i in range(N0):
            A_eq[i, i * N1:(i + 1) * N1] = 1

        # Column constraints (target)
        for j in range(N1):
            A_eq[N0 + j, j::N1] = 1

        b_eq = np.concatenate([a0, a1])

        res = linprog(
            C.flatten(),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * (N0 * N1),
            method=self.solver
        )

        if not res.success:
            raise ValueError(f"OT solver failed: {res.message}")

        return res.x.reshape(N0, N1)

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X0, a0):
        """
        Store the reference measure.

        Parameters
        ----------
        X0 : array (N0, d)
        a0 : array (N0,)
        """
        X0, a0 = self._prepare_inputs(X0, a0)

        self.X0 = X0
        self.a0 = a0
        self.N0, self.d = X0.shape
        self.fitted_ = True

        return self

    def transform(self, X1, a1):
        """
        Compute LOT embedding of target measure.

        Parameters
        ----------
        X1 : array (N1, d)
        a1 : array (N1,)

        Returns
        -------
        s1_hat : array (N0, d)
        a1_hat : array (N0,)
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        X1, a1 = self._prepare_inputs(X1, a1)

        # Cost matrix
        C = self._compute_cost(self.X0, X1)

        # Optimal transport plan
        Gamma = self._solve_ot(C, self.a0, a1)

        # LOT map
        a0_inv = np.diag(1.0 / self.a0)
        s1_hat = a0_inv @ (Gamma @ X1)

        # Mass is preserved on reference
        a1_hat = self.a0.copy()

        return s1_hat, a1_hat

    def fit_transform(self, X0, a0, X1, a1):
        """
        Convenience method: fit reference and transform target.
        """
        return self.fit(X0, a0).transform(X1, a1)
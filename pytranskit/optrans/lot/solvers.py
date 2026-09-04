"""
Authors
-------
Naqib Sad Pathan 
Adapted for pytranskit and research workflow.

Inspired by
------------
1. "Wang W, Slepčev D, Basu S, Ozolek JA, Rohde GK. A linear optimal transportation 
framework for quantifying and visualizing variations in sets of images. 
International journal of computer vision. 2013 Jan;101(2):254-69."
2. "Flamary R, Courty N, Gramfort A, Alaya MZ, Boisbunon A, Chambon S, Chapel L, 
Corenflos A, Fatras K, Fournier N, Gautheron L. Pot: Python optimal transport. 
Journal of Machine Learning Research. 2021;22(78):1-8."

Date
-----
Created: September 2026


Descriptions
------------
Discrete optimal-transport solvers with a target-by-reference convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TransportResult:
    coupling: FloatArray
    cost: float
    converged: bool
    solver: str


def _weights(count: int, supplied: ArrayLike | None) -> FloatArray:
    values = np.full(count, 1.0 / count) if supplied is None else np.asarray(supplied, dtype=float)
    if values.shape != (count,) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Transport weights must be non-negative vectors with positive mass")
    return values / values.sum()


def _linprog_transport(cost: FloatArray, a: FloatArray, b: FloatArray) -> tuple[FloatArray, bool]:
    n_target, n_reference = cost.shape
    target_rows = csr_matrix(
        (
            np.ones(n_target * n_reference),
            (
                np.repeat(np.arange(n_target), n_reference),
                np.arange(n_target * n_reference),
            ),
        ),
        shape=(n_target, n_target * n_reference),
    )
    reference_rows = csr_matrix(
        (
            np.ones(n_target * n_reference),
            (
                np.tile(np.arange(n_reference), n_target),
                np.arange(n_target * n_reference),
            ),
        ),
        shape=(n_reference, n_target * n_reference),
    )
    constraints = vstack([target_rows, reference_rows[:-1]])
    masses = np.concatenate([a, b[:-1]])
    result = linprog(cost.ravel(), A_eq=constraints, b_eq=masses, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Linear-program OT failed: {result.message}")
    return result.x.reshape(cost.shape), bool(result.success)


def solve_transport(
    target: ArrayLike,
    reference: ArrayLike,
    solver: str = "sinkhorn",
    target_weights: ArrayLike | None = None,
    reference_weights: ArrayLike | None = None,
    reg: float = 0.01,
    max_iter: int = 10_000,
) -> TransportResult:
    """
    Solve discrete optimal transport between target and reference cell matrices.

    Parameters
    ----------
    target : ArrayLike
        Target cell matrix (N_target_cells, D_features).
    reference : ArrayLike
        Reference cell matrix (N_reference_cells, D_features).
    solver : str, default="sinkhorn"
        Algorithm name ('sinkhorn', 'hungarian', 'emd', 'linprog').
    target_weights : ArrayLike, optional
        Weights for target cells. Defaults to uniform (1/N_target).
    reference_weights : ArrayLike, optional
        Weights for reference cells. Defaults to uniform (1/N_reference).
    reg : float, default=0.01
        Entropic regularization factor for Sinkhorn solver.
    max_iter : int, default=10000
        Maximum solver iterations.

    Returns
    -------
    TransportResult
        Data class containing coupling matrix, total transport cost, and convergence state.
    """
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if target.ndim != 2 or reference.ndim != 2 or target.shape[1] != reference.shape[1]:
        raise ValueError("target and reference must be 2D with equal marker/feature dimensions")

    a = _weights(len(target), target_weights)
    b = _weights(len(reference), reference_weights)
    cost = cdist(target, reference, metric="sqeuclidean")

    name = solver.lower()
    converged = True

    if name == "hungarian":
        if len(target) != len(reference):
            raise ValueError("Hungarian matching requires equal target and reference cell counts")
        row, column = linear_sum_assignment(cost)
        coupling = np.zeros_like(cost)
        coupling[row, column] = 1.0 / len(target)

    elif name == "linprog":
        coupling, converged = _linprog_transport(cost, a, b)

    elif name in {"emd", "emd2", "plot"}:
        try:
            import ot
        except ImportError as error:
            raise ImportError("EMD solver requires the 'POT' dependency (`pip install POT`)") from error
        coupling = np.asarray(ot.emd(a, b, cost, numItermax=max_iter))

    elif name == "sinkhorn":
        if reg <= 0:
            raise ValueError("Sinkhorn regularization factor `reg` must be positive")
        try:
            import ot
        except ImportError as error:
            raise ImportError("Sinkhorn solver requires the 'POT' dependency (`pip install POT`)") from error

        scale = max(float(cost.max()), np.finfo(float).eps)
        coupling, log = ot.sinkhorn(
            a, b, cost / scale, reg=reg, numItermax=max_iter, stopThr=1e-7, log=True
        )
        coupling = np.asarray(coupling)
        converged = bool(log.get("niter", max_iter) < max_iter - 1)

    else:
        raise ValueError(f"Unknown transport solver: {solver}")

    return TransportResult(
        coupling=coupling,
        cost=float(np.sum(coupling * cost)),
        converged=converged,
        solver=name,
    )

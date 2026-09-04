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
PyTransKit Linear Optimal Transport (LOT) Module
Provides object-oriented interface for batch LOT computation and reconstruction.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .solvers import TransportResult, solve_transport


@dataclass(frozen=True)
class LOTResult:
    reference: NDArray[np.float64]
    transported: NDArray[np.float64]
    displacement: NDArray[np.float64]
    embedding: NDArray[np.float64]
    transport: TransportResult


class LinearOptimalTransport:
    """
    Linear Optimal Transport Transformer.

    Parameters
    ----------
    reference : ArrayLike
        Reference matrix of shape (N_cells, D_features).
    solver : str, default="sinkhorn"
        OT solver name ('sinkhorn', 'hungarian', 'emd', 'linprog').
    representation : str, default="displacement"
        Embedding representation type ('displacement' or 'map').
    solver_kwargs : dict, optional
        Additional solver parameters.
    """

    def __init__(
        self,
        reference: ArrayLike,
        solver: str = "sinkhorn",
        representation: str = "displacement",
        solver_kwargs: dict[str, Any] | None = None,
    ):
        self.reference = np.asarray(reference, dtype=np.float64)
        self.solver = solver
        self.representation = representation
        self.solver_kwargs = solver_kwargs or {}

        # Matrix dimensions: N_cells = rows, D_features = columns
        self.n_cells, self.d_features = self.reference.shape

    def transform_sample(self, target: ArrayLike) -> LOTResult:
        """Compute LOT result for a single target matrix (N_target_cells, D_features)."""
        tgt_arr = np.asarray(target, dtype=np.float64)

        transport = solve_transport(
            tgt_arr, self.reference, solver=self.solver, **self.solver_kwargs
        )
        ref_weights = transport.coupling.sum(axis=0)

        transported = self.reference.copy()
        active = ref_weights > np.finfo(float).eps
        transported[active] = (
            transport.coupling[:, active].T @ tgt_arr
        ) / ref_weights[active, None]

        displacement = (transported - self.reference) * np.sqrt(ref_weights[:, None])

        if self.representation == "displacement":
            matrix = displacement
        elif self.representation in {"map", "legacy_map"}:
            matrix = transported
        else:
            raise ValueError("representation must be 'displacement' or 'map'")

        # Reshape to 1D vector using Fortran order ('F') to group features
        embedding = matrix.reshape(-1, order="F")
        return LOTResult(self.reference, transported, displacement, embedding, transport)

    def transform_batch(
        self,
        targets: Sequence[ArrayLike] | dict[str | int, ArrayLike],
        n_jobs: int = -1,
    ) -> dict[str | int, LOTResult]:
        """Compute LOT embeddings for multiple samples in parallel using CPU cores."""
        import os

        if isinstance(targets, dict):
            target_dict = {k: np.asarray(v, dtype=np.float64) for k, v in targets.items()}
        else:
            target_dict = {i: np.asarray(v, dtype=np.float64) for i, v in enumerate(targets)}

        num_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)

        worker_fn = partial(
            _parallel_worker_transform,
            transformer=self,
        )

        results: dict[str | int, LOTResult] = {}

        if num_workers > 1 and len(target_dict) > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker_fn, key, target)
                    for key, target in target_dict.items()
                ]
                for future in futures:
                    key, res = future.result()
                    results[key] = res
        else:
            for key, target in target_dict.items():
                _, res = worker_fn(key, target)
                results[key] = res

        return results

    def lot_embedding_to_cell_matrix(
        self,
        embedding: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Reconstruct the 2D cell matrix (N_cells, D_features) from a 1D LOT embedding.

        Parameters
        ----------
        embedding : NDArray
            1D vector of length (N_cells * D_features).
        weights : NDArray, optional
            Reference cell weights (N_cells,). If None, uniform weights (1/N_cells) are used.

        Returns
        -------
        NDArray
            Reconstructed cell feature matrix of shape (N_cells, D_features).
        """
        # Step 1: Unflatten 1D array back to 2D matrix using Fortran order ('F')
        matrix = embedding.reshape((self.n_cells, self.d_features), order="F")

        # Step 2: Convert back to target cell coordinates if representation is 'displacement'
        if self.representation == "displacement":
            if weights is None:
                weights = np.full(self.n_cells, 1.0 / self.n_cells, dtype=np.float64)

            # Revert scaling factor: displacement / sqrt(weights) + reference
            scale = np.sqrt(weights[:, None])
            cell_matrix = np.where(scale > 0, matrix / scale, 0.0) + self.reference
            return cell_matrix

        # If representation is 'map', matrix is already the transported cell positions
        return matrix


def _parallel_worker_transform(
    key: str | int,
    target: np.ndarray,
    transformer: LinearOptimalTransport,
) -> tuple[str | int, LOTResult]:
    """Top-level helper function for ProcessPoolExecutor serialization."""
    res = transformer.transform_sample(target)
    return key, res

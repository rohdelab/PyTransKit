"""
Authors
-------
Naqib Sad Pathan 
Adapted for pytranskit and research workflow.

Inspired by
------------
"Wang W, Slepčev D, Basu S, Ozolek JA, Rohde GK. A linear optimal transportation 
framework for quantifying and visualizing variations in sets of images. 
International journal of computer vision. 2013 Jan;101(2):254-69."

Date
-----
Created: September 2026


Descriptions
------------
How to use.
"""


import numpy as np
from .engine import LinearOptimalTransport

# From outside this folder, run from the repository root with:
# python -m pytranskit.optrans.lot.tutorial

def main() -> None:
    # 1. Define mock reference (1000 cells, 8 markers), target samples and custom solver parameters
    reference = np.random.randn(1000, 8)
    targets = {
        "patient_A": np.random.randn(1200, 8),
        "patient_B": np.random.randn(1000, 8),
        "patient_C": np.random.randn(900, 8),
    }
    custom_solver_args = {
        "reg": 0.05,        # Entropic regularization (e.g., 0.05 instead of default 0.01)
        "max_iter": 20000,   # Increase max iterations for convergence
    }

    # 2. Pass solver_kwargs to the model
    lot_model = LinearOptimalTransport(
        reference=reference,
        solver="emd2",
        representation="displacement",
        solver_kwargs=custom_solver_args,
    )

    # Parallel transform
    results = lot_model.transform_batch(targets, n_jobs=-1)

    # Reconstruct 2D cell matrix from 1D embedding
    sample_embedding = results["patient_A"].embedding
    reconstructed_matrix = lot_model.lot_embedding_to_cell_matrix(sample_embedding)

    print("Embedding shape:", sample_embedding.shape)           # (8000,)
    print("Reconstructed shape:", reconstructed_matrix.shape)   # (1000, 8)


if __name__ == "__main__":
    main()

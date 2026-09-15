import numpy as np
import pytest

from pytranskit.TBM3D.multVOT import multVOT


def gaussian_3d(shape=(8, 8, 8), center=None, sigma=1.5):
    """Create a small normalized 3D Gaussian."""
    if center is None:
        center = tuple((s - 1) / 2.0 for s in shape)

    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    image = np.exp(
        -(
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
        )
        / (2.0 * sigma**2)
    )

    image = image.astype(np.float64)

    # Unit-mass normalization.
    image /= image.sum()

    # Same small positive offset used by the TBM3D pipeline.
    image += 1e-5

    return image / image.sum()


def test_tbm3d_identity_transport():
    """
    End-to-end smoke test for 3D TBM.

    Identical source and target images should produce a valid
    transport map close to the identity map.
    """

    shape = (8, 8, 8)

    image = gaussian_3d(shape)

    results = multVOT(
        image,
        image,
        numScales=0,
        dc=0.1,
    )

    # ---------------------------------------------------------
    # Check result structure
    # ---------------------------------------------------------

    required_keys = {
        "f1",
        "f2",
        "f3",
        "I0",
        "I1",
        "I0_recon",
        "MSE1",
        "MSE2",
        "MSE3",
        "mass",
        "curl",
        "iter",
    }

    assert required_keys.issubset(results.keys())

    # ---------------------------------------------------------
    # Check transport map
    # ---------------------------------------------------------

    f1 = results["f1"]
    f2 = results["f2"]
    f3 = results["f3"]

    assert f1.shape == shape
    assert f2.shape == shape
    assert f3.shape == shape

    assert np.isfinite(f1).all()
    assert np.isfinite(f2).all()
    assert np.isfinite(f3).all()

    # ---------------------------------------------------------
    # Identity transport should be close to x -> x
    # ---------------------------------------------------------

    X, Y, Z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    displacement = np.sqrt(
        (f1 - X) ** 2
        + (f2 - Y) ** 2
        + (f3 - Z) ** 2
    )

    mean_displacement = np.mean(displacement)

    assert np.isfinite(mean_displacement)
    assert mean_displacement < 1.0

    # ---------------------------------------------------------
    # Check reconstructed image
    # ---------------------------------------------------------

    reconstruction = results["I0_recon"]

    assert reconstruction.shape == shape
    assert np.isfinite(reconstruction).all()

    # MSE should be finite.
    mse = np.asarray(results["MSE3"])

    assert mse.ndim == 1
    assert len(mse) >= 1
    assert np.isfinite(mse).all()


def test_tbm3d_small_translation():
    """
    End-to-end test with two slightly shifted 3D Gaussian volumes.

    The algorithm should produce a valid, non-trivial transport map.
    """

    shape = (8, 8, 8)

    source = gaussian_3d(
        shape,
        center=(3.5, 3.5, 3.5),
    )

    target = gaussian_3d(
        shape,
        center=(4.0, 3.5, 3.5),
    )

    results = multVOT(
        source,
        target,
        numScales=0,
        dc=0.1,
    )

    # ---------------------------------------------------------
    # Basic output validation
    # ---------------------------------------------------------

    for key in ("f1", "f2", "f3", "I0_recon"):
        assert key in results
        assert results[key].shape == shape
        assert np.isfinite(results[key]).all()

    assert results["iter"] >= 1

    # ---------------------------------------------------------
    # The transport map should remain finite and valid.
    # ---------------------------------------------------------

    f1 = results["f1"]
    f2 = results["f2"]
    f3 = results["f3"]

    X, Y, Z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    displacement = np.sqrt(
        (f1 - X) ** 2
        + (f2 - Y) ** 2
        + (f3 - Z) ** 2
    )

    assert np.isfinite(displacement).all()

    # A translated image should not result in NaN/Inf
    # or an unrealistically large deformation.
    assert np.mean(displacement) < 2.0

    # ---------------------------------------------------------
    # Check optimization diagnostics.
    # ---------------------------------------------------------

    for key in ("MSE1", "MSE2", "MSE3", "mass", "curl"):
        values = np.asarray(results[key])

        assert values.ndim == 1
        assert np.isfinite(values).all()
        assert len(values) >= results["iter"]
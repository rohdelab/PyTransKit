import numpy as np

from pytranskit.TBM3D.gradient import (
    compVOTGradients,
    projection_metric,
)


def identity_map(shape):
    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    return x.astype(float), y.astype(float), z.astype(float)


def test_comp_vot_gradients_identity_map():
    shape = (7, 8, 6)
    f1, f2, f3 = identity_map(shape)

    rng = np.random.default_rng(10)
    image = rng.random(shape) + 0.1

    f1t, f2t, f3t, reconstructed, error, flag = compVOTGradients(
        f1, f2, f3, image, image, lambd=50, gamma=0.1
    )

    assert flag == 0
    assert reconstructed.shape == shape
    assert error.shape == shape

    for array in (f1t, f2t, f3t, reconstructed, error):
        assert np.isfinite(array).all()

    np.testing.assert_allclose(reconstructed, image, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(error, 0.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(f1t, 0.0, atol=1e-8)
    np.testing.assert_allclose(f2t, 0.0, atol=1e-8)
    np.testing.assert_allclose(f3t, 0.0, atol=1e-8)


def test_comp_vot_gradients_detects_orientation_reversal():
    shape = (6, 6, 6)
    x, y, z = identity_map(shape)

    # Reverse the first coordinate. The Jacobian determinant is negative.
    f1 = (shape[0] - 1) - x
    image = np.ones(shape, dtype=float)

    *_, flag = compVOTGradients(
        f1, y, z, image, image, lambd=0, gamma=0
    )

    assert flag == 1


def test_projection_metric_identical_subspaces():
    rng = np.random.default_rng(123)
    A = rng.normal(size=(12, 3))

    d = projection_metric(A, A)

    np.testing.assert_allclose(d, 0.0, atol=1e-10)


def test_projection_metric_is_symmetric():
    rng = np.random.default_rng(321)
    A = rng.normal(size=(15, 3))
    B = rng.normal(size=(15, 3))

    d_ab = projection_metric(A, B)
    d_ba = projection_metric(B, A)

    np.testing.assert_allclose(d_ab, d_ba, atol=1e-10)
    assert d_ab >= 0

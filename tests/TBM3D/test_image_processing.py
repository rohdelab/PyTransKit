import numpy as np

from pytranskit.TBM3D.image_processing import (
    GP_ker,
    GPReduce,
    GPExpand,
    gaussian_bf,
    gen_pdf,
)


def test_gp_kernel_is_normalized():
    kernel = GP_ker()
    assert kernel.shape == (5, 5, 5)
    assert np.all(kernel >= 0)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-12)


def test_gpreduce_halves_3d_volume():
    image = np.arange(8 * 10 * 12, dtype=float).reshape(8, 10, 12)
    reduced = GPReduce(image)

    assert reduced.shape == (4, 5, 6)
    assert np.isfinite(reduced).all()


def test_gpreduce_preserves_constant_image():
    image = np.ones((8, 8, 8), dtype=float)
    reduced = GPReduce(image)
    np.testing.assert_allclose(reduced, 1.0, atol=1e-12)


def test_gpexpand_returns_requested_shape():
    image = np.ones((4, 5, 3), dtype=float)
    expanded = GPExpand(image, (8, 9, 6))

    assert expanded.shape == (8, 9, 6)
    assert np.isfinite(expanded).all()


def test_gpexpand_preserves_constant_image_interior():
    image = np.ones((4, 4, 4), dtype=float)
    expanded = GPExpand(image, (7, 7, 7))

    assert expanded.shape == (7, 7, 7)
    np.testing.assert_allclose(expanded, 1.0, atol=1e-12)


def test_gaussian_bf_is_normalized():
    r = np.arange(-2, 3)
    x, y, z = np.meshgrid(r, r, r, indexing="ij")
    kernel = gaussian_bf(x, y, z, sigma=1.0)

    assert kernel.shape == (5, 5, 5)
    assert np.isfinite(kernel).all()
    assert np.all(kernel >= 0)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-12)


def test_gen_pdf_is_a_probability_density(reference_volume):
    pdf = gen_pdf(reference_volume, sigma=0, dc=0.1)

    assert pdf.shape == reference_volume.shape
    assert np.isfinite(pdf).all()
    assert np.all(pdf > 0)
    np.testing.assert_allclose(pdf.sum(), 1.0, atol=1e-12)

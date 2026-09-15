import numpy as np
import matplotlib

matplotlib.use("Agg")

from pytranskit.TBM3D.visualization import (
    inpaint_nans3,
    pad_to_square,
    inverse_image,
    visual_roc,
)


def test_inpaint_nans3_removes_nan_values():
    volume = np.ones((6, 6, 6), dtype=float)
    volume[2, 2, 2] = np.nan

    result = inpaint_nans3(volume)

    assert result.shape == volume.shape
    assert np.isfinite(result).all()


def test_pad_to_square():
    image = np.ones((3, 5), dtype=float)
    result = pad_to_square(image, pad_value=-1)

    assert result.shape == (5, 5)
    np.testing.assert_array_equal(result[1:4, :], image)
    assert np.all(result[[0, 4], :] == -1)


def test_inverse_image_identity_transport():
    shape = (4, 5, 3)
    reference = np.ones(shape, dtype=float)
    reference[1:3, 1:4, 1] = 2.0

    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    # Zero transport-domain displacement => identity map.
    zero_features = np.zeros((5, 3 * np.prod(shape)), dtype=float)

    images = inverse_image(zero_features, reference)

    assert images.shape == (5,) + shape
    assert np.isfinite(images).all()
    assert np.min(images) >= 0 - 1e-12
    assert np.max(images) <= np.max(reference) + 1e-12


def test_visual_roc_returns_auc():
    train_labels = np.array([0, 0, 1, 1])
    test_labels = np.array([0, 0, 1, 1])
    train_scores = np.array([0.0, 0.2, 0.8, 1.0])
    test_scores = np.array([0.1, 0.3, 0.7, 0.9])

    auc = visual_roc(
        train_scores,
        train_labels,
        test_scores,
        test_labels,
    )

    np.testing.assert_allclose(auc, 1.0)

import numpy as np
import pytest


def gaussian_3d(shape=(8, 9, 7), center=None, sigma=1.5):
    """Create a small, deterministic, unit-mass 3D Gaussian."""
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
    return image / image.sum()


@pytest.fixture
def volume_shape():
    return (8, 9, 7)


@pytest.fixture
def reference_volume(volume_shape):
    return gaussian_3d(volume_shape)


@pytest.fixture
def shifted_volume(volume_shape):
    center = (
        (volume_shape[0] - 1) / 2.0 + 0.8,
        (volume_shape[1] - 1) / 2.0 - 0.6,
        (volume_shape[2] - 1) / 2.0 + 0.5,
    )
    return gaussian_3d(volume_shape, center=center)


@pytest.fixture
def training_volumes(volume_shape):
    centers = [
        ((volume_shape[0] - 1) / 2.0 - 1.0,
         (volume_shape[1] - 1) / 2.0,
         (volume_shape[2] - 1) / 2.0),
        ((volume_shape[0] - 1) / 2.0,
         (volume_shape[1] - 1) / 2.0,
         (volume_shape[2] - 1) / 2.0),
        ((volume_shape[0] - 1) / 2.0 + 1.0,
         (volume_shape[1] - 1) / 2.0,
         (volume_shape[2] - 1) / 2.0),
        ((volume_shape[0] - 1) / 2.0,
         (volume_shape[1] - 1) / 2.0 + 1.0,
         (volume_shape[2] - 1) / 2.0),
    ]
    return np.stack(
        [gaussian_3d(volume_shape, center=c) for c in centers],
        axis=0,
    )

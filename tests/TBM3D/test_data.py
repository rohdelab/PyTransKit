import numpy as np
import pandas as pd

from pytranskit.TBM3D.image_processing import load_data


def test_load_data_train_without_reference(tmp_path):
    image0 = np.arange(27, dtype=float).reshape(3, 3, 3) + 1.0
    image1 = np.arange(27, dtype=float).reshape(3, 3, 3) + 2.0

    p0 = tmp_path / "image0.npy"
    p1 = tmp_path / "image1.npy"
    np.save(p0, image0)
    np.save(p1, image1)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {"image_path": [str(p0), str(p1)], "label": [0, 1]}
    ).to_csv(csv_path, index=False)

    images, labels, reference = load_data(
        str(csv_path),
        normalized=False,
        mode="train",
        epsilon=1e-5,
    )

    assert len(images) == 2
    assert labels == [0, 1]
    assert images[0].shape == (3, 3, 3)
    assert reference.shape == (3, 3, 3)
    assert np.isfinite(reference).all()

    # The implementation normalizes each image and then adds epsilon.
    expected = image0 / image0.sum() + 1e-5
    np.testing.assert_allclose(images[0], expected)


def test_load_data_test_mode_does_not_return_reference(tmp_path):
    image = np.ones((3, 3, 3), dtype=float)
    image_path = tmp_path / "image.npy"
    np.save(image_path, image)

    csv_path = tmp_path / "test.csv"
    pd.DataFrame(
        {"image_path": [str(image_path)], "label": [1]}
    ).to_csv(csv_path, index=False)

    images, labels = load_data(
        str(csv_path),
        normalized=True,
        mode="test",
    )

    assert len(images) == 1
    assert labels == [1]
    np.testing.assert_array_equal(images[0], image)

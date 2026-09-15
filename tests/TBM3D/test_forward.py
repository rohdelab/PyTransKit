import pickle
import numpy as np

import pytranskit.TBM3D.TBM_forward as forward


def test_process_features():
    shape = (3, 4, 5)
    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    reference = np.ones(shape, dtype=float)
    reference[1, 1, 1] = 2.0

    result0 = {
        "f1": x.copy(),
        "f2": y.copy(),
        "f3": z.copy(),
        "label": 0,
    }
    result1 = {
        "f1": x + 0.5,
        "f2": y,
        "f3": z,
        "label": 1,
    }

    features, labels = forward.process_features(
        [result0, result1], reference
    )

    assert features.shape == (2, 3 * np.prod(shape))
    assert labels.shape == (2,)
    np.testing.assert_array_equal(labels, np.array([0, 1]))
    assert np.isfinite(features).all()

    # Identity maps produce zero displacement features.
    np.testing.assert_allclose(features[0], 0.0, atol=1e-12)


def test_run_tbm_adds_label_and_saves_result(tmp_path, monkeypatch):
    expected = {
        "f1": np.zeros((2, 2, 2)),
        "f2": np.zeros((2, 2, 2)),
        "f3": np.zeros((2, 2, 2)),
        "iter": 1,
    }

    def fake_mult_vot(reference, image, numScales=4, dc=0.1):
        return dict(expected)

    monkeypatch.setattr(forward, "multVOT", fake_mult_vot)

    output_path = tmp_path / "result.pkl"
    result = forward.run_tbm(
        np.ones((2, 2, 2)),
        7,
        np.ones((2, 2, 2)),
        output_path=str(output_path),
        numScales=1,
        dc=0.2,
    )

    assert result["label"] == 7
    assert output_path.exists()

    with output_path.open("rb") as f:
        saved = pickle.load(f)

    assert saved["label"] == 7
    np.testing.assert_array_equal(saved["f1"], expected["f1"])


def test_tbm3d_forward_serial_orchestration(monkeypatch):
    calls = []

    def fake_run_tbm(
        image, label, reference, output_path=None, numScales=4, dc=0.1
    ):
        calls.append((image.shape, label, reference.shape, numScales, dc))
        return {"f1": image, "f2": image, "f3": image, "label": label}

    monkeypatch.setattr(forward, "run_tbm", fake_run_tbm)

    images = [
        np.zeros((2, 2, 2)),
        np.ones((2, 2, 2)),
    ]
    labels = [0, 1]
    reference = np.ones((2, 2, 2))

    results = forward.TBM3D_forward(
        images,
        labels,
        reference,
        output_folder=None,
        parallel=False,
        numScales=1,
        dc=0.1,
    )

    assert len(results) == 2
    assert [r["label"] for r in results] == [0, 1]
    assert len(calls) == 2

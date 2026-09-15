import numpy as np

from pytranskit.TBM3D.pca_tbm3d import SimplePCA, predict_score


def test_simple_pca_transform_and_inverse():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(8, 6))

    pca = SimplePCA(n_components=3)
    Z = pca.fit_transform(X)

    assert Z.shape == (8, 3)
    assert pca.components_.shape == (3, 6)
    assert pca.mean_.shape == (6,)
    assert np.isfinite(Z).all()
    assert np.isfinite(pca.components_).all()

    Z_again = pca.transform(X)
    np.testing.assert_allclose(Z_again, Z, rtol=1e-10, atol=1e-10)

    X_reconstructed = pca.inverse_transform(Z)
    assert X_reconstructed.shape == X.shape
    assert np.isfinite(X_reconstructed).all()


def test_simple_pca_fractional_components():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, 5))

    pca = SimplePCA(n_components=0.9)
    Z = pca.fit_transform(X)

    assert 1 <= pca.k_ <= 5
    assert Z.shape == (10, pca.k_)
    assert pca.explained_variance_ratio_.shape[0] == 10
    np.testing.assert_allclose(
        pca.explained_variance_ratio_.sum(),
        1.0,
        atol=1e-10,
    )


def test_predict_score_selects_validation_threshold():
    valscore = np.array([-2.0, -1.0, 1.0, 2.0])
    vallabel = np.array([0, 0, 1, 1])
    testscore = np.array([-3.0, -1.5, 1.5, 3.0])

    threshold, direction, bacc, prediction = predict_score(
        testscore, valscore, vallabel
    )

    assert direction in {"greater", "less"}
    assert np.isfinite(threshold)
    assert 0.0 <= bacc <= 1.0
    np.testing.assert_array_equal(prediction, np.array([0, 0, 1, 1]))

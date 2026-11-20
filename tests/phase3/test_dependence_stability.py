import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.run_dependence_stability import correlated, ks_alarm


def test_intervention_preserves_each_gaussian_marginal_and_only_changes_selected_pair():
    rng = np.random.default_rng(3301)
    z = rng.normal(size=(100_000, 8))
    x = correlated(z, (2, 5), .6)
    untouched = [j for j in range(8) if j != 5]
    assert np.array_equal(x[:, untouched], z[:, untouched])
    assert abs(float(np.corrcoef(x[:, 2], x[:, 5])[0, 1]) - .6) < .01
    assert abs(float(x[:, 5].mean())) < .02
    assert abs(float(x[:, 5].std()) - 1) < .02


def test_stationary_intervention_is_identity_and_invalid_protocol_inputs_fail():
    rng = np.random.default_rng(3302)
    z = rng.normal(size=(500, 8))
    assert np.array_equal(correlated(z, None, 0), z)
    with pytest.raises(ValueError):
        correlated(z, (5, 2), .6)
    with pytest.raises(ValueError):
        correlated(z, (2, 5), 1.0)


def test_ks_comparator_uses_frozen_training_pca_and_does_not_alarm_identical_rows():
    rng = np.random.default_rng(3303)
    train = rng.normal(size=(2000, 8))
    ref = rng.normal(size=(1000, 8))
    scaler = StandardScaler().fit(train)
    train_z, ref_z = scaler.transform(train), scaler.transform(ref)
    pca = PCA(n_components=3, svd_solver="full").fit(train_z)
    ref_pca = pca.transform(ref_z)
    assert not ks_alarm(ref_z, ref_pca, ref_z.copy(), pca)

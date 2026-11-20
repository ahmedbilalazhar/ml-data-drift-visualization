import numpy as np

from scripts.run_elec2_temporal_validation import load_ele2


def test_elec2_source_checksum_order_and_feature_admission():
    x, y, date, period, ids, digest = load_ele2()
    assert x.shape == (45_312, 6)
    assert set(np.unique(y)) == {0, 1}
    assert len(np.unique(ids)) == 45_312
    assert np.all(np.diff(date) >= 0)
    same_group = (date[1:] == date[:-1]) & (x[1:, 0] == x[:-1, 0])
    assert np.all(np.diff(period)[same_group] >= 0)
    assert np.isfinite(x).all()
    assert digest == '2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2'
    assert np.all((period >= 0) & (period <= 1))

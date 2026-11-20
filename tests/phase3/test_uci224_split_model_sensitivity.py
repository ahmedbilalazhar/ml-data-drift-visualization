import numpy as np

from scripts.run_uci224_split_model_sensitivity import MODELS, SEEDS, WINDOWS, partitions


def test_locked_matrix_sizes_and_future_cohorts():
    assert SEEDS == tuple(range(420001, 420007))
    assert MODELS == ('logistic', 'tree_depth6')
    assert WINDOWS == {'batch1': (1,), 'batch1_2': (1, 2)}
    assert sum(9 if key == 'batch1' else 8 for key in WINDOWS) == 17


def test_split_partition_is_disjoint_and_exhaustive():
    y = np.repeat(np.arange(1, 7), 80)
    tr, ca, rf = partitions(y, 420001)
    assert len(tr) + len(ca) + len(rf) == len(y)
    assert len(set(tr) & set(ca)) == 0
    assert len(set(tr) & set(rf)) == 0
    assert len(set(ca) & set(rf)) == 0
    assert set(np.concatenate((tr, ca, rf))) == set(range(len(y)))

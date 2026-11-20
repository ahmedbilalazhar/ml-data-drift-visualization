"""Meaningful Phase 3 scientific invariants, independent of legacy E0 tests."""
from fractions import Fraction
from itertools import product
import numpy as np
import pytest
from scipy.stats import binom
from drift_monitoring.phase3.core import (poisson_binomial, impact, harm_kmax, is_harm,
    holm, innovations, baseline_x, monitoring_partition, historical_partition,
    label_probability, fit_artifacts, FeatureBatch, score_batch, atc_threshold)
from drift_monitoring.phase3.replay import LabelBroker, replay, as_of, release_feedback


@pytest.mark.parametrize('q,expected', [([0]*5, [1,0,0,0,0,0]), ([1]*5, [0,0,0,0,0,1])])
def test_poisson_binomial_degenerate(q, expected):
    np.testing.assert_allclose(poisson_binomial(q), expected)


def test_poisson_binomial_matches_binomial_and_bruteforce():
    np.testing.assert_allclose(poisson_binomial([.7]*40), binom.pmf(np.arange(41), 40, .7), atol=1e-14)
    q = [.1,.35,.8,.99]
    truth = np.zeros(5)
    for state in product([0,1], repeat=4):
        truth[sum(state)] += np.prod([p if z else 1-p for p,z in zip(q,state)])
    np.testing.assert_allclose(poisson_binomial(q), truth, atol=1e-14)


@pytest.mark.parametrize('q', [[-1], [1.1], [np.nan], [np.inf]])
def test_invalid_probabilities(q):
    with pytest.raises(ValueError):
        poisson_binomial(q)


@pytest.mark.parametrize('reference', [0, 1, 1400, 1500, 1501, 2000])
def test_strict_harm_boundary_exact(reference):
    kmax = harm_kmax(500, reference, 2000)
    for k in range(501):
        truth = Fraction(reference,2000)-Fraction(k,500) > Fraction(1,20)
        assert (k <= kmax) == truth
        assert is_harm(k,500,reference,2000) == truth
    result, pmf = impact(np.full(500,.7), reference,2000)
    np.testing.assert_allclose(result['concern'], pmf[:max(0,min(501,kmax+1))].sum(), atol=1e-13)


def test_holm_known_case():
    np.testing.assert_allclose(holm([.01,.04,.03]), [.03,.06,.06])
    np.testing.assert_allclose(holm([.8,.8]), [1.,1.])


def test_c6_analytic_covariance_and_empirical_marginals():
    # Linear transform of independent N(0,1), not the invalid rotation of correlated X.
    for rho in [.3,.9]:
        transform = np.array([[1,0],[rho,np.sqrt(1-rho*rho)]])
        np.testing.assert_allclose(np.diag(transform@transform.T), [1,1])
        np.testing.assert_allclose((transform@transform.T)[0,1], rho)
    z,_ = innovations(90001, 3, 200000)
    for rho in [.3,.9]:
        x = baseline_x(z,rho)
        assert abs(x[:,1].var()-1) < .015
        assert abs(np.corrcoef(x[:,:2].T)[0,1]-rho) < .01


def test_generation_pairing_regimes_and_labels_follow_modified_inputs():
    base = monitoring_partition(90001,'C0')
    z,u = innovations(90001,3,12000)
    for c in range(1,7):
        p = monitoring_partition(90001,f'C{c}')
        np.testing.assert_array_equal(p.x[:4000],base.x[:4000])
        np.testing.assert_array_equal(p.y[:4000],base.y[:4000])
    for c in ['C3','C4','C5']:
        p = monitoring_partition(90001,c)
        np.testing.assert_array_equal(p.x,base.x)
    recurring = monitoring_partition(90001,'C5')
    np.testing.assert_array_equal(recurring.y[8000:],base.y[8000:])
    c2 = monitoring_partition(90001,'C2')
    assert c2.x[4000,0]-base.x[4000,0] == pytest.approx(1.5/2000)
    np.testing.assert_allclose(c2.x[5999:,0]-base.x[5999:,0],1.5)
    for c in ['C1','C2','C6']:
        p = monitoring_partition(90001,c)
        np.testing.assert_array_equal(p.y,(u<label_probability(p.x)).astype(np.int8))


@pytest.fixture(scope='module')
def artifacts():
    return fit_artifacts(90001, historical_partition(90001,0,4000),
        historical_partition(90001,1,2000),historical_partition(90001,2,2000))


def test_train_only_transform_and_no_monitor_mutation(artifacts):
    train = historical_partition(90001,0,4000)
    np.testing.assert_allclose(artifacts.scaler.mean_, train.x.mean(axis=0))
    before = [v.copy() for v in (artifacts.scaler.mean_, artifacts.model.coef_,
               artifacts.pca.components_,artifacts.clusters.cluster_centers_,
               artifacts.calibration.y_thresholds_)]
    p=monitoring_partition(90001,'C2')
    r,_,_=score_batch(artifacts,FeatureBatch(p.x[-500:],p.ids[-500:],90001,'C2',24))
    after = [artifacts.scaler.mean_, artifacts.model.coef_, artifacts.pca.components_,
             artifacts.clusters.cluster_centers_,artifacts.calibration.y_thresholds_]
    for a,b in zip(before,after):
        np.testing.assert_array_equal(a,b)
    assert r['cal_estimate'] == r['calibrated_ac_alone']
    assert all('accuracy' not in k or k == 'reference_accuracy' for k in r)
    assert 'y' not in FeatureBatch.__dataclass_fields__ and 'labels' not in FeatureBatch.__dataclass_fields__
    allids = np.concatenate(list(artifacts.fit_ids.values())+[p.ids])
    assert len(np.unique(allids)) == len(allids)


def test_future_label_tampering_and_delay_invariance(artifacts):
    p=monitoring_partition(90001,'C3')
    records,caches=[],[]
    for b in range(1,4):
        sl=slice((b-1)*500,b*500)
        r,c,_=score_batch(artifacts,FeatureBatch(p.x[sl],p.ids[sl],90001,'C3',b))
        records.append(r);caches.append(c)
    for d in [0,5,20]:
        for flipped in [False,True]:
            labels=1-p.y[:1500] if flipped else p.y[:1500]
            broker=LabelBroker(p.ids[:1500],labels,500,d)
            assert broker.release(0) == []
            events,verified=replay(records,caches,broker)
            assert len(verified)==3
            assert [v['available_at'] for v in verified]==[1+d,2+d,3+d]
            assert max(e['clock'] for e in events)==3+d
            assert sum(e['event']=='prediction_persisted' for e in events)==3
            if d == 0:
                assert [e['phase'] for e in events[:2]]==[0,1]
            if d:
                assert all(v['verified_accuracy'] is None for v in as_of(records,verified,d))
            # Tampering cannot enter any feature-only record or cached proxy.
            if not flipped:
                baseline_correct=[v['correct'] for v in verified]
            else:
                assert [v['correct'] for v in verified]==[500-v for v in baseline_correct]
    # Repeat feature scoring after altering future truth and compare all scientific fields.
    p.y[:] = 1-p.y
    r,c,_=score_batch(artifacts,FeatureBatch(p.x[:500],p.ids[:500],90001,'C3',1))
    for key in records[0]:
        if not key.endswith('_seconds'):
            np.testing.assert_equal(r[key],records[0][key])
    for key in c:
        np.testing.assert_array_equal(c[key],caches[0][key])


def test_id_join_rejects_mismatch(artifacts):
    p=monitoring_partition(90001,'C0')
    r,c,_=score_batch(artifacts,FeatureBatch(p.x[:500],p.ids[:500],90001,'C0',1))
    with pytest.raises(ValueError,match='ID mismatch'):
        replay([r],[c],LabelBroker(p.ids[:500]+1,p.y[:500],500,0))


def test_atc_tie_to_larger_threshold():
    s=np.array([.5,.6,.7,.8])
    t=atc_threshold(s,np.array([1,1,0,0]))
    assert t==.7


def test_future_release_changes_do_not_rewrite_earlier_snapshot(artifacts):
    p=monitoring_partition(90001,'C0')
    records,caches=[],[]
    for b in range(1,4):
        sl=slice((b-1)*500,b*500)
        r,c,_=score_batch(artifacts,FeatureBatch(p.x[sl],p.ids[sl],90001,'C0',b))
        records.append(r);caches.append(c)
    tampered=p.y[:1500].copy()
    tampered[1000:]=1-tampered[1000:]
    _,v1=replay(records,caches,LabelBroker(p.ids[:1500],p.y[:1500],500,5))
    _,v2=replay(records,caches,LabelBroker(p.ids[:1500],tampered,500,5))
    assert as_of(records,v1,7)==as_of(records,v2,7)
    assert as_of(records,v1,8)!=as_of(records,v2,8)
    assert all(x['verified_accuracy'] is None for x in as_of(records,v1,5))


def test_latent_ablation_recomputes_family_without_changing_impact(artifacts):
    p=monitoring_partition(90001,'C6')
    r,c,_=score_batch(artifacts,FeatureBatch(p.x[-500:],p.ids[-500:],90001,'C6',24))
    expected=holm([r[f'x{j}_p'] for j in range(8)])
    assert r['original_alarm']==bool(np.any(expected<=.05))
    assert r['min_original_adjusted_p']==expected.min()
    estimate,_=impact(c['correctness_probability'],artifacts.reference_correct,artifacts.reference_n)
    for key,value in estimate.items():
        assert r[f'cal_{key}']==value


def test_calibrator_uses_calibration_correctness_only(artifacts):
    calibration=historical_partition(90001,1,2000)
    p=artifacts.model.predict_proba(artifacts.scaler.transform(calibration.x))[:,1]
    raw=np.maximum(p,1-p)
    from sklearn.isotonic import IsotonicRegression
    independent=IsotonicRegression(out_of_bounds='clip').fit(raw,(p>=.5)==calibration.y)
    np.testing.assert_array_equal(artifacts.calibration.y_thresholds_,independent.y_thresholds_)
    np.testing.assert_array_equal(artifacts.calibration.X_thresholds_,independent.X_thresholds_)


@pytest.mark.parametrize('feature_count',[1,2,8,32,128])
def test_feature_contract_scores_all_dimensions_and_caps_pca(feature_count):
    from drift_monitoring.phase3.core import Partition, FeatureBatch, fit_artifacts, score_batch
    rng=np.random.default_rng(20261116+feature_count)
    def part(seed,n,offset):
        x=rng.normal(size=(n,feature_count))
        weights=np.linspace(.9,.2,min(feature_count,3))
        logits=x[:,:len(weights)]@weights
        y=(rng.random(n)<1/(1+np.exp(-logits))).astype(np.int8)
        ids=seed*10_000_000+offset+np.arange(n,dtype=np.int64)
        return Partition(x,y,ids)
    train,calibration,reference=(part(1,240,0),part(2,180,300),part(3,180,600))
    artifacts=fit_artifacts(20261116,train,calibration,reference,
                            n_pca_components=3,n_clusters=5)
    monitor=part(4,96,900)
    batch=FeatureBatch(monitor.x,monitor.ids,4,'dimension_sensitivity',1)
    expected_components=min(3,feature_count,len(train.x)-1)
    assert artifacts.pca.n_components_==expected_components
    assert artifacts.clusters.n_clusters==5
    assert batch.x.flags.writeable is False and batch.ids.flags.writeable is False
    record,cache,_=score_batch(artifacts,batch)
    assert record['n_features']==feature_count
    assert record['n_original_tests']==feature_count
    assert record['n_pca_components']==expected_components
    assert record['n_clusters']==5
    assert all(f'x{i}_ks' in record and f'x{i}_rank' in record for i in range(feature_count))
    assert all(f'pc{i}_ks' in record for i in range(expected_components))
    assert all(f'cluster{i}_occupancy' in record for i in range(5))
    assert cache['pca'].shape==(len(batch.x),expected_components)


def test_feature_batch_is_defensive_against_source_mutation():
    from drift_monitoring.phase3.core import FeatureBatch
    source=np.ones((2,2)); ids=np.array([11,12])
    batch=FeatureBatch(source,ids,1,'contract',1)
    source[0,0]=99; ids[0]=99
    assert batch.x[0,0]==1 and batch.ids[0]==11
    with pytest.raises(ValueError):
        batch.x[0,0]=3


@pytest.mark.parametrize('feature_count',[1,2,8,32,128])
def test_v2_scoring_uses_dynamic_feature_and_pca_counts(feature_count):
    from drift_monitoring.phase3.v2 import Part, fit_models, score_model
    rng=np.random.default_rng(861000+feature_count)
    def part(n,offset):
        x=rng.normal(size=(n,feature_count))
        weight=np.linspace(.8,.25,min(feature_count,3))
        z=x[:,:len(weight)]@weight
        y=(z+rng.normal(scale=.3,size=n)>0).astype(np.int8)
        ids=88000000+offset+np.arange(n,dtype=np.int64)
        return Part(x,y,ids)
    train,cal,ref=(part(220,0),part(160,1000),part(160,2000))
    artifacts=fit_models(3,train,cal,ref,n_pca_components=3)
    monitor=part(80,3000)
    batch=FeatureBatch(monitor.x,monitor.ids,3,'dim',1)
    record,cache=score_model(artifacts['logistic'],batch,'dimension_sensitivity','logistic')
    components=min(3,feature_count,len(train.x)-1)
    assert record['n_features']==feature_count
    assert record['n_pca_components']==components
    assert len([k for k in record if k.startswith('x') and k.endswith('_ks')])==feature_count
    assert len([k for k in record if k.startswith('pc') and k.endswith('_ks')])==components
    assert cache['pca'].shape==(len(batch.x),components)


def test_released_feedback_pairs_only_due_labels_with_source_features():
    ids = np.arange(12, dtype=np.int64) + 900
    features = np.arange(24, dtype=float).reshape(12, 2)
    labels = np.arange(12, dtype=np.int8) % 2
    broker = LabelBroker(ids, labels, batch_size=4, delay=0)
    assert release_feedback(broker, features, ids, clock=0) == []
    # Forecast clock 1 has no prior labels; delay-zero batch 1 is still not visible.
    assert release_feedback(broker, features, ids, clock=0) == []
    first = release_feedback(broker, features, ids, clock=1)
    assert len(first) == 1 and first[0].source_batch == 1 and first[0].available_at == 1
    np.testing.assert_array_equal(first[0].features, features[:4])
    np.testing.assert_array_equal(first[0].labels, labels[:4])
    with pytest.raises(ValueError, match="do not align"):
        release_feedback(LabelBroker(ids, labels, 4, 0), features, ids[::-1], clock=1)
    drained = release_feedback(broker, features, ids, clock=3)
    assert [v.source_batch for v in drained] == [2, 3]

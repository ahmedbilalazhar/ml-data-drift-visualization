"""Feature-only monitoring. No monitoring function accepts current labels.

Historical labels enter fit_artifacts; monitoring truth belongs to evaluation.py.
All fitted objects are frozen by protocol after historical fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
import warnings
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Partition:
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class FeatureBatch:
    """The entire live scoring interface: deliberately has no label field."""
    x: np.ndarray
    ids: np.ndarray
    seed: int
    condition: str
    batch: int  # one based, also input completion clock

    def __post_init__(self):
        # Defensive, read-only copies keep caller mutation from changing a live score.
        x = np.array(self.x, copy=True)
        ids = np.array(self.ids, copy=True)
        x.setflags(write=False)
        ids.setflags(write=False)
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'ids', ids)


def innovations(seed, stage, n):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stage)]))
    return rng.normal(size=(n, 8)), rng.random(n)


def baseline_x(z, rho=0.3):
    x = z.copy()
    x[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    return x


def label_probability(x):
    return expit(1.2*x[:, 0] - 0.9*x[:, 1] + 0.6*x[:, 2])


def historical_partition(seed, stage, n):
    z, u = innovations(seed, stage, n)
    x = baseline_x(z)
    return Partition(x, (u < label_probability(x)).astype(np.int8),
                     seed*1_000_000 + stage*100_000 + np.arange(n))


def monitoring_partition(seed, condition, n=12000, onset=4000, transition_end=6000,
                         return_row=8000):
    """Paired innovations/uniforms; modify X before computing its label law.

    C6: Var(X1)=rho^2+(1-rho^2)=1, Cov(X0,X1)=rho.
    C2 changes X0 while preserving the predeclared construction for X1.
    Row onset is zero based (first changed row is one-based 4001).
    """
    if condition not in [f'C{i}' for i in range(7)]:
        raise ValueError(condition)
    z, u = innovations(seed, 3, n)
    x = baseline_x(z)
    rows = np.arange(n)
    post = rows >= onset
    ramp = np.clip((rows-onset+1) / (transition_end-onset), 0, 1)
    if condition == 'C1':
        x[post, 7] += 1.5
    elif condition == 'C2':
        x[:, 0] += 1.5*ramp
    elif condition == 'C6':
        x[post, 1] = 0.9*z[post, 0] + np.sqrt(1-0.9**2)*z[post, 1]
    p = label_probability(x)
    if condition == 'C3':
        p[post] = 1-p[post]
    elif condition == 'C4':
        p = (1-ramp)*p + ramp*(1-p)
    elif condition == 'C5':
        active = post & (rows < return_row)
        p[active] = 1-p[active]
    ids = seed*1_000_000 + 300_000 + int(condition[1:])*20_000 + rows
    return Partition(x, (u < p).astype(np.int8), ids)


def holm(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm correction requires a non-empty vector of finite p-values in [0,1]')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def poisson_binomial(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('correctness probabilities must be finite and in [0,1]')
    pmf = np.zeros(len(q)+1)
    pmf[0] = 1
    for j, prob in enumerate(q):
        pmf[1:j+2] = (1-prob)*pmf[1:j+2] + prob*pmf[:j+1]
        pmf[0] *= 1-prob
    # Normalization only removes floating point accumulation error.
    return pmf / pmf.sum()


def harm_kmax(n, reference_correct, reference_n, margin=Fraction(1, 20)):
    """Largest integer K with K/n < reference_accuracy-margin, exactly."""
    if (not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0
            or not isinstance(reference_n, (int, np.integer))
            or isinstance(reference_n, (bool, np.bool_)) or reference_n <= 0
            or not isinstance(reference_correct, (int, np.integer))
            or isinstance(reference_correct, (bool, np.bool_))
            or not 0 <= reference_correct <= reference_n):
        raise ValueError('harm calculation requires valid positive sample sizes and correct counts')
    margin = Fraction(margin)
    if not 0 <= margin <= 1:
        raise ValueError('harm margin must be a fraction in [0,1]')
    threshold = n*(Fraction(int(reference_correct), int(reference_n))-margin)
    return (threshold.numerator-1)//threshold.denominator


def is_harm(correct, n, reference_correct, reference_n):
    return int(correct) <= harm_kmax(n, reference_correct, reference_n)


def impact(q, reference_correct, reference_n):
    if len(q) == 0:
        raise ValueError('impact requires at least one prediction')
    harm_kmax(len(q), reference_correct, reference_n)
    pmf = poisson_binomial(q)
    cdf = np.cumsum(pmf)
    cdf[-1] = 1
    n = len(q)
    lo, hi = np.searchsorted(cdf, [0.025, 0.975]) / n
    k = harm_kmax(n, reference_correct, reference_n)
    concern = 0.0 if k < 0 else (1.0 if k >= n else float(cdf[k]))
    return dict(estimate=float(np.mean(q)), lower=float(lo), upper=float(hi),
                concern=concern, concern_flag=concern >= 0.95,
                point_flag=(reference_correct/reference_n-float(np.mean(q))) > 0.05), pmf


def atc_threshold(confidence, correct):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct)
    if (confidence.ndim != 1 or correct.ndim != 1 or len(confidence) == 0
            or len(confidence) != len(correct) or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))
            or not np.isin(correct, (0, 1)).all()):
        raise ValueError('ATC calibration requires aligned confidence and binary correctness vectors')
    scores, counts = np.unique(confidence, return_counts=True)
    # Include endpoints, with an upper endpoint that can select zero rows.
    candidates = np.unique(np.r_[0., scores, 1., np.nextafter(1., np.inf)])
    proportions = (len(confidence)-np.searchsorted(np.sort(confidence), candidates,
                                                  side='left'))/len(confidence)
    errors = np.abs(proportions-np.mean(correct))
    return float(candidates[np.flatnonzero(errors == errors.min())[-1]])


@dataclass
class Artifacts:
    scaler: StandardScaler
    model: LogisticRegression
    pca: PCA
    clusters: MiniBatchKMeans
    calibration: IsotonicRegression
    reference_x: np.ndarray
    reference_z: np.ndarray
    reference_correct: int
    reference_n: int
    atc: float
    fit_seconds: float
    fit_ids: dict


def fit_artifacts(seed, train, calibration, reference, *, n_pca_components=3, n_clusters=3,
                  model_max_iter=500):
    t = perf_counter()
    parts = (train, calibration, reference)
    feature_count = train.x.shape[1] if train.x.ndim == 2 else 0
    if feature_count < 1:
        raise ValueError('training features must be a non-empty matrix with at least one column')
    if (not isinstance(n_pca_components, (int, np.integer)) or isinstance(n_pca_components, (bool, np.bool_))
            or n_pca_components < 1):
        raise ValueError('requested PCA components must be a positive integer')
    if (not isinstance(n_clusters, (int, np.integer)) or isinstance(n_clusters, (bool, np.bool_))
            or n_clusters < 1 or n_clusters > len(train.x)):
        raise ValueError('cluster count must be a positive integer no larger than the training set')
    for name, part in zip(('training', 'calibration', 'reference'), parts):
        if part.x.ndim != 2 or part.x.shape[1] != feature_count or len(part.x) == 0:
            raise ValueError(f'{name} features must be a non-empty matrix with {feature_count} columns')
        if part.y.ndim != 1 or part.ids.ndim != 1 or len(part.y) != len(part.x) or len(part.ids) != len(part.x):
            raise ValueError(f'{name} labels and IDs must be one-dimensional and row-aligned')
        if not np.isfinite(part.x).all() or not np.isin(part.y, (0, 1)).all():
            raise ValueError(f'{name} features must be finite and labels must be binary')
        if len(np.unique(part.ids)) != len(part.ids):
            raise ValueError(f'{name} IDs must be unique')
    if len(np.unique(train.y)) != 2:
        raise ValueError('training partition must contain both binary classes')
    ids = [set(p.ids.tolist()) for p in parts]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError('historical partitions overlap')
    scaler = StandardScaler().fit(train.x)
    xt = scaler.transform(train.x)
    if (not isinstance(model_max_iter, (int, np.integer)) or isinstance(model_max_iter, (bool, np.bool_))
            or model_max_iter < 1):
        raise ValueError('model max_iter must be a positive integer')
    model = LogisticRegression(C=1., max_iter=int(model_max_iter), random_state=seed)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(xt, train.y)
    if any(issubclass(w.category, ConvergenceWarning) for w in seen):
        raise RuntimeError('LogisticRegression did not converge; no paired seed may be dropped')
    actual_components = min(int(n_pca_components), feature_count, len(train.x) - 1)
    if actual_components < 1:
        raise ValueError('training set is too small to fit one PCA component')
    pca = PCA(n_components=actual_components, svd_solver='full').fit(xt)
    clusters = MiniBatchKMeans(n_clusters=int(n_clusters), n_init=3, batch_size=256,
                              random_state=seed).fit(xt)
    cp = model.predict_proba(scaler.transform(calibration.x))[:, 1]
    confidence = np.maximum(cp, 1-cp)
    correctness = ((cp >= .5) == calibration.y).astype(int)
    iso = IsotonicRegression(out_of_bounds='clip').fit(confidence, correctness)
    rx = scaler.transform(reference.x)
    rp = model.predict_proba(rx)[:, 1]
    return Artifacts(scaler, model, pca, clusters, iso, rx[-1000:],
                     pca.transform(rx[-1000:]), int(((rp >= .5) == reference.y).sum()),
                     len(reference.y), atc_threshold(confidence, correctness),
                     perf_counter()-t, dict(train=train.ids, calibration=calibration.ids,
                                           reference=reference.ids))


def score_batch(a: Artifacts, b: FeatureBatch):
    """Score exclusively from historical artifacts, current features and IDs."""
    started = perf_counter()
    if (b.x.ndim != 2 or b.x.shape[1] != a.scaler.n_features_in_ or len(b.x) == 0
            or b.ids.ndim != 1 or len(b.x) != len(b.ids)):
        raise ValueError('feature cohort must be non-empty, have the fitted feature width, and align with 1D IDs')
    if not np.isfinite(b.x).all() or len(np.unique(b.ids)) != len(b.ids):
        raise ValueError('feature cohort values must be finite and IDs unique')
    if any(np.intersect1d(b.ids, fit_ids).size for fit_ids in a.fit_ids.values()):
        raise ValueError('monitoring IDs overlap historical fit IDs')
    if not isinstance(b.batch, (int, np.integer)) or b.batch < 1:
        raise ValueError('batch must be a positive one-based integer')
    x = a.scaler.transform(b.x)
    p = a.model.predict_proba(x)[:, 1]
    pred = (p >= .5).astype(np.int8)
    raw = np.maximum(p, 1-p)
    q = a.calibration.predict(raw)
    prediction_s = perf_counter()-started
    t = perf_counter()
    original, latent, messages = [], [], []
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for j in range(a.scaler.n_features_in_):
            original.append(ks_2samp(a.reference_x[:, j], x[:, j], method='auto'))
        original_s = perf_counter()-t
        t = perf_counter()
        z = a.pca.transform(x)
        for j in range(a.pca.n_components_):
            latent.append(ks_2samp(a.reference_z[:, j], z[:, j], method='auto'))
        latent_s = perf_counter()-t
        messages = [f'{w.category.__name__}: {w.message}' for w in seen]
    tests = original+latent
    adjusted = holm([v.pvalue for v in tests])
    original_adjusted = holm([v.pvalue for v in original])
    t = perf_counter()
    calibrated, pmf = impact(q, a.reference_correct, a.reference_n)
    calibrated_s = perf_counter()-t
    t = perf_counter()
    uncalibrated, raw_pmf = impact(raw, a.reference_correct, a.reference_n)
    raw_s = perf_counter()-t
    t = perf_counter()
    cluster = a.clusters.predict(x)
    cluster_s = perf_counter()-t
    feature_count = a.scaler.n_features_in_
    component_count = a.pca.n_components_
    cluster_count = a.clusters.n_clusters
    record = dict(seed=b.seed, condition=b.condition, batch=b.batch,
                  cohort=f'{b.seed}:{b.condition}:{b.batch}', n=len(x),
                  n_features=feature_count, n_original_tests=feature_count,
                  n_pca_components=component_count, n_clusters=cluster_count,
                  observed_at=b.batch, prediction_phase='before_label_release',
                  reference_correct=a.reference_correct, reference_n=a.reference_n,
                  reference_accuracy=a.reference_correct/a.reference_n,
                  input_alarm=bool(np.any(adjusted <= .05)),
                  original_alarm=bool(np.any(original_adjusted <= .05)),
                  latent_rejection=bool(np.any(adjusted[feature_count:] <= .05)),
                  min_adjusted_p=float(adjusted.min()),
                  min_original_adjusted_p=float(original_adjusted.min()),
                  atc_estimate=float(np.mean(raw >= a.atc)), atc_threshold=a.atc,
                  constant_estimate=a.reference_correct/a.reference_n,
                  prediction_seconds=prediction_s, original_ks_seconds=original_s,
                  latent_ks_seconds=latent_s, calibrated_impact_seconds=calibrated_s,
                  raw_impact_seconds=raw_s, cluster_seconds=cluster_s,
                  ks_warning_count=len(messages))
    for prefix, result in [('cal', calibrated), ('raw', uncalibrated)]:
        record.update({f'{prefix}_{k}': v for k, v in result.items()})
    # Identity baseline is explicit and checked downstream.
    record['calibrated_ac_alone'] = calibrated['estimate']
    ranking = np.argsort([-v.statistic for v in original], kind='stable')
    for j in range(feature_count):
        record.update({f'x{j}_ks': float(original[j].statistic),
                       f'x{j}_p': float(original[j].pvalue),
                       f'x{j}_adjusted_p': float(adjusted[j]),
                       f'x{j}_flag': bool(adjusted[j] <= .05),
                       f'x{j}_rank': int(np.where(ranking == j)[0][0])+1})
    for j in range(component_count):
        record[f'pc{j}_ks'] = float(latent[j].statistic)
        record[f'pc{j}_adjusted_p'] = float(adjusted[feature_count+j])
    for j in range(cluster_count):
        mask = cluster == j
        record[f'cluster{j}_n'] = int(mask.sum())
        record[f'cluster{j}_occupancy'] = float(mask.mean())
        record[f'cluster{j}_q'] = float(q[mask].mean()) if mask.sum() >= 20 else np.nan
    record['score_seconds'] = perf_counter()-started
    cache = dict(ids=b.ids, probability=p, prediction=pred, raw_confidence=raw,
                 correctness_probability=q, pca=z, cluster=cluster,
                 cal_pmf=pmf, raw_pmf=raw_pmf)
    return record, cache, messages

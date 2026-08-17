"""Synthetic disk-drive telemetry for lesson 4.

A data centre watches its drives and wants to replace the ones about to fail.
The columns imitate SMART counters — the self-monitoring statistics a real
drive reports — and the label says whether the drive failed in the following
thirty days.

Two properties make it the right dataset for this lesson.

**It is imbalanced the way the real problem is.** 306 drives out of 8,000 fail —
3.8% — so a model that predicts "healthy" for every drive is 96.2% accurate and
catches nothing. That single fact is what the lesson is built on.

**Its answer is written down.** The log-odds that generated each label are a
known linear function of the transformed, standardised columns, published in
``TRUE_COEFFICIENTS``, so an estimate can be compared against the truth rather
than merely admired.

    from disk_data import load_drives, transform_features, TRUE_COEFFICIENTS

    df = load_drives()
    X = transform_features(df)

Everything is offline and reproducible: same seed, same numbers, on any
machine.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

#: Change in the log-odds of failure per standard deviation of each column,
#: *after* the transform below. The lesson recovers these from data and checks
#: them against these values. `seek_error_rate` is deliberately zero: a column
#: that looks like telemetry, carries no signal, and is there to be found out.
TRUE_COEFFICIENTS = {
    "reallocated_sectors": 1.80,
    "spin_retry_count": 1.15,
    "read_error_rate": 0.85,
    "temperature_c": 0.45,
    "power_on_hours": 0.35,
    "seek_error_rate": 0.00,
}

#: Intercept on the transformed, standardised features. Chosen so that about
#: 4% of drives fail, which is the imbalance the lesson is built on. It is the
#: log-odds of failure for a drive with perfectly average telemetry: a 0.2%
#: chance over thirty days, which is what a healthy drive should have.
TRUE_INTERCEPT = -6.20

FEATURES = list(TRUE_COEFFICIENTS)

#: Counters and error rates act on a logarithmic scale: going from 0 to 4
#: reallocated sectors matters far more than going from 40 to 44. The truth is
#: linear in these transformed columns, not in the raw ones — which is exactly
#: the feature engineering of lesson 2, and the reason a model fitted on the
#: raw columns recovers the coefficients only approximately.
LOG_COLUMNS = ("reallocated_sectors", "spin_retry_count",
               "read_error_rate", "seek_error_rate")


def transform_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the skewed columns on a log scale, leave the rest alone."""
    out = frame[FEATURES].copy()
    for column in LOG_COLUMNS:
        out[column] = np.log1p(out[column])
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_drives(n_samples: int = 8_000,
                random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return drive telemetry with a `failed` column of zeros and ones.

    The features are skewed and correlated the way SMART counters are: most
    drives report zero reallocated sectors, hot drives are the ones that have
    been running longest, and error rates rise together. That is what makes the
    problem worth a model rather than a threshold on one column.
    """
    rng = np.random.default_rng(random_state)

    # Hours in service. Gamma, so a long tail of veterans.
    power_on_hours = rng.gamma(shape=3.2, scale=5_600, size=n_samples).clip(120, 62_000)

    # Older drives run hotter, because they sit in the oldest racks.
    temperature_c = np.clip(
        rng.normal(34 + 6e-5 * power_on_hours, 4.2, n_samples), 18, 63)

    # Most drives never reallocate a sector; a few reallocate many. The rate
    # rises with age, which is the correlation that makes this realistic.
    sector_rate = 0.35 + 9e-5 * power_on_hours
    reallocated_sectors = rng.poisson(sector_rate, n_samples)
    heavy = rng.random(n_samples) < 0.015
    reallocated_sectors = reallocated_sectors + heavy * rng.poisson(16, n_samples)

    # Spin retries: rarer still, and correlated with reallocations.
    spin_retry_count = rng.poisson(0.08 + 0.045 * reallocated_sectors, n_samples)

    # Error rates, lognormal because they span orders of magnitude.
    read_error_rate = np.exp(
        rng.normal(-6.4 + 0.02 * reallocated_sectors, 0.85, n_samples))
    # The decoy: same shape, same plausibility, no connection to failure.
    seek_error_rate = np.exp(rng.normal(-6.1, 0.9, n_samples))

    frame = pd.DataFrame({
        "power_on_hours": power_on_hours.round(0).astype(int),
        "temperature_c": temperature_c.round(1),
        "reallocated_sectors": reallocated_sectors.astype(int),
        "spin_retry_count": spin_retry_count.astype(int),
        "read_error_rate": read_error_rate.round(6),
        "seek_error_rate": seek_error_rate.round(6),
    })

    # The truth acts on transformed, standardised columns, so the published
    # coefficients read as "per standard deviation" and do not depend on units.
    logit = TRUE_INTERCEPT + _latent_severity(frame)
    frame["failed"] = (rng.random(n_samples) < _sigmoid(logit)).astype(int)

    return frame


def _latent_severity(frame: pd.DataFrame) -> pd.Series:
    """The part of the log-odds that depends on the telemetry."""
    transformed = transform_features(frame)
    standardised = (transformed - transformed.mean()) / transformed.std(ddof=0)
    return sum(weight * standardised[name]
               for name, weight in TRUE_COEFFICIENTS.items())


def load_two_features(n_samples: int = 8_000,
                      random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """The same problem cut down to two columns, so a boundary can be drawn.

    Used where the point is to *see* the decision boundary. Keeps the two
    strongest predictors and drops the rest.
    """
    frame = load_drives(n_samples, random_state)
    return frame[["reallocated_sectors", "temperature_c", "failed"]]


def load_multiclass(n_samples: int = 8_000,
                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Three outcomes instead of two, for the multiclass section.

    A drive either stays healthy, degrades (readable, but reporting errors and
    due for replacement), or fails outright. Degradation is the middle state
    the binary label hides, and models confuse it with both neighbours — which
    is the point of looking at a three-by-three confusion matrix.
    """
    frame = load_drives(n_samples, random_state)
    rng = np.random.default_rng(random_state + 7)

    severity = _latent_severity(frame)
    outcome = np.where(frame["failed"] == 1, 2, 0)
    # Among the survivors, the ones with the worst telemetry are degrading.
    at_risk = (outcome == 0) & (rng.random(len(frame)) < _sigmoid(severity - 3.1))
    outcome = np.where(at_risk, 1, outcome)

    frame = frame.drop(columns="failed")
    frame["condition"] = outcome  # 0 healthy, 1 degraded, 2 failed
    return frame


if __name__ == "__main__":
    df = load_drives()
    rate = df["failed"].mean()
    print(df.head())
    print(f"\n{len(df):,} drives, {df['failed'].sum()} failures "
          f"({rate:.1%} of the fleet)")
    print(f"always-healthy baseline accuracy: {1 - rate:.1%}")
    print("\nmulticlass:")
    print(load_multiclass()["condition"].value_counts().sort_index())

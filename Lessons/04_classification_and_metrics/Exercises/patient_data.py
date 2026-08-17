"""Synthetic hospital readmission data for exercise 4.

A hospital wants to predict which discharged patients will be readmitted within
thirty days. The columns are the kind of thing a discharge summary carries, and
the label says whether the patient came back.

The dataset is built to the same recipe as the lesson's disk fleet, for the same
reasons: it is **imbalanced** the way the real problem is, and its answer is
**written down**, so you can check your estimates rather than admire them.

Two things are deliberately planted, and the exercise asks about both.

- One column carries **no signal at all**. Its coefficient in
  ``TRUE_COEFFICIENTS`` is exactly zero.
- The positive rate is low enough that accuracy is useless as a summary.

    from patient_data import load_patients, transform_features

    df = load_patients()
    X = transform_features(df)

Do not read ``TRUE_COEFFICIENTS`` before attempting question 13. Reading it
afterwards, to mark yourself, is the point of it existing.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 7

#: Change in the log-odds of readmission per standard deviation of each column,
#: after the transform below.
TRUE_COEFFICIENTS = {
    "prior_admissions": 1.55,
    "days_in_hospital": 0.95,
    "n_medications": 0.70,
    "age_years": 0.50,
    "haemoglobin": -0.60,
    "distance_to_clinic_km": 0.00,
}

#: Intercept on the transformed, standardised features. Gives a readmission
#: rate of roughly 6%.
TRUE_INTERCEPT = -4.65

FEATURES = list(TRUE_COEFFICIENTS)

#: Counts act on a log scale: a second admission matters far more than a
#: fourteenth. The truth is linear in these transformed columns.
LOG_COLUMNS = ("prior_admissions", "days_in_hospital", "n_medications")


def transform_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the skewed count columns on a log scale, leave the rest alone."""
    out = frame[FEATURES].copy()
    for column in LOG_COLUMNS:
        out[column] = np.log1p(out[column])
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_patients(n_samples: int = 9_000,
                  random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return discharge records with a `readmitted` column of zeros and ones."""
    rng = np.random.default_rng(random_state)

    age_years = np.clip(rng.normal(63, 16, n_samples), 18, 99)
    # Older patients have been admitted before, with a long tail.
    prior_admissions = rng.poisson(np.clip(0.25 + 0.022 * (age_years - 40), 0.05, 4))
    days_in_hospital = np.clip(
        rng.gamma(2.1, 2.4, n_samples) + 0.35 * prior_admissions, 1, 60)
    n_medications = np.clip(
        rng.poisson(6 + 0.9 * prior_admissions + 0.05 * age_years), 1, 45)
    haemoglobin = np.clip(rng.normal(13.4 - 0.02 * age_years, 1.6, n_samples), 6, 18)
    # The column with nothing in it: plausible, measurable, irrelevant.
    distance_to_clinic_km = np.clip(rng.gamma(2.0, 6.5, n_samples), 0.2, 90)

    frame = pd.DataFrame({
        "age_years": age_years.round(0).astype(int),
        "prior_admissions": prior_admissions.astype(int),
        "days_in_hospital": days_in_hospital.round(0).astype(int),
        "n_medications": n_medications.astype(int),
        "haemoglobin": haemoglobin.round(1),
        "distance_to_clinic_km": distance_to_clinic_km.round(1),
    })

    transformed = transform_features(frame)
    standardised = ((transformed - transformed.mean())
                    / transformed.std(ddof=0))
    logit = TRUE_INTERCEPT + sum(
        weight * standardised[name]
        for name, weight in TRUE_COEFFICIENTS.items())
    frame["readmitted"] = (rng.random(n_samples) < _sigmoid(logit)).astype(int)

    return frame


if __name__ == "__main__":
    df = load_patients()
    rate = df["readmitted"].mean()
    print(df.head())
    print(f"\n{len(df):,} discharges, {df['readmitted'].sum()} readmissions "
          f"({rate:.1%})")
    print(f"never-readmitted baseline accuracy: {1 - rate:.1%}")

"""Data for Lesson 9 — Neural networks.

One narrative runs through the three notebooks: **Meridian Instruments**, a
fictional maker of optical distance sensors, has two problems that a linear
model cannot touch and one that it can — and the third is included precisely
so that "add a hidden layer" does not become a reflex.

1. ``make_acceptance_test`` — every sensor off the line is measured on two
   calibration axes, and is accepted when *both* offsets are jointly inside a
   tolerance. The accept region is therefore the inside of an ellipse, which
   no single straight boundary can describe. Notebooks 01 and 03.
2. ``make_channel_drift`` — a two-channel sensor whose drift is correctable
   when the two channels drift the *same* way and not when they drift
   oppositely. That is the exclusive-or (XOR) function with noise, and it is
   the smallest problem that needs a hidden layer. Notebook 01.
3. ``load_digit_images`` — the 8x8 handwritten digits bundled with
   scikit-learn, used for the multiclass (softmax) work in notebooks 02 and
   03. Real data, ten classes, and no download: scikit-learn ships a copy.

The ``TRUE_*`` constants below record exactly how the synthetic labels were
generated, including the rate at which the test rig records the wrong verdict.
That last one matters: it puts a **ceiling** on this lesson, so every accuracy
is read against 0.97 rather than against 1.000.

No network access, no API keys — the two synthetic sets are generated locally
with numpy and the digits ship inside scikit-learn.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Acceptance testing — Notebooks 01 and 03
# ---------------------------------------------------------------------------
#
# Each unit is characterised by two calibration offsets, reported in the
# instrument's own physical units:
#
#   gain_offset_db    — decibels, nominal 0, spread 0.40 dB across production
#   phase_offset_deg  — degrees,  nominal 0, spread 3.00 deg
#
# Firmware applies a single global correction, which succeeds only when the
# combined offset is small. "Combined" is the elliptical norm below: each axis
# is divided by its own tolerance and the two are added in quadrature, which
# is how a two-axis tolerance is actually written in a specification.

TRUE_N_UNITS = 3000

TRUE_GAIN_SIGMA_DB = 0.40       # production spread on the gain axis
TRUE_PHASE_SIGMA_DEG = 3.00     # production spread on the phase axis

TRUE_GAIN_TOL_DB = 0.50         # 1.25 sigma
TRUE_PHASE_TOL_DEG = 3.75       # 1.25 sigma — the same multiple, by design

# Because both tolerances are the same multiple of their axis's spread, the
# accept region is a *circle* of this radius once each axis is standardised.
TRUE_ACCEPT_RADIUS_SIGMA = 1.25

# The rig that records pass/fail is itself imperfect: this fraction of
# verdicts is recorded as the opposite of the truth. It is the reason no
# model in this lesson can pass 97% — see the Bayes ceiling below.
TRUE_RIG_ERROR_RATE = 0.03

# The best accuracy any classifier can reach on the *recorded* labels.
TRUE_BAYES_ACCURACY = 1.0 - TRUE_RIG_ERROR_RATE


def make_acceptance_test(seed=9001, n_units=TRUE_N_UNITS):
    """Meridian's end-of-line acceptance test: 3,000 sensors, two measurements.

    Each unit's two calibration offsets are drawn independently from centred
    normal distributions with the production spreads above. The unit is
    *truly* within tolerance when

        (gain / TRUE_GAIN_TOL_DB)**2 + (phase / TRUE_PHASE_TOL_DEG)**2 < 1

    and the test rig records that verdict correctly with probability
    1 - TRUE_RIG_ERROR_RATE, flipping it otherwise.

    Returns a DataFrame with four columns:

        gain_offset_db, phase_offset_deg   the two features a model may use
        accepted                           the *recorded* label (noisy) — 0/1
        truly_within_tolerance             the noise-free label, for checking

    A model is only ever shown the first three. The fourth exists so that the
    3% rig error can be separated from a model's own mistakes, which is the
    kind of separation a real acceptance dataset never allows.
    """
    rng = np.random.default_rng(seed)

    gain = rng.normal(0.0, TRUE_GAIN_SIGMA_DB, n_units)
    phase = rng.normal(0.0, TRUE_PHASE_SIGMA_DEG, n_units)

    elliptical_norm = (gain / TRUE_GAIN_TOL_DB) ** 2 + (phase / TRUE_PHASE_TOL_DEG) ** 2
    truth = (elliptical_norm < 1.0).astype(int)

    flip = rng.random(n_units) < TRUE_RIG_ERROR_RATE
    recorded = np.where(flip, 1 - truth, truth)

    return pd.DataFrame({
        "gain_offset_db": gain,
        "phase_offset_deg": phase,
        "accepted": recorded,
        "truly_within_tolerance": truth,
    })


# ---------------------------------------------------------------------------
# 2. Two-channel drift — Notebook 01, the smallest problem needing a hidden layer
# ---------------------------------------------------------------------------
#
# A two-channel sensor drifts with temperature. When both channels drift the
# same way the drift is *common mode* and one global correction removes it;
# when they drift in opposite directions it is *differential* and no single
# correction can. So the unit is correctable exactly when the two drifts share
# a sign — which is exclusive-or (XOR) on the signs, the textbook example of a
# function no straight line separates.

TRUE_N_DRIFT_UNITS = 800
TRUE_DRIFT_MAGNITUDE = 1.0      # centre of each of the four clouds, in mV
TRUE_DRIFT_SPREAD = 0.28        # spread within a cloud

# (channel_a sign, channel_b sign, correctable)
TRUE_DRIFT_QUADRANTS = [
    (+1, +1, 1),   # both high  — common mode, correctable
    (-1, -1, 1),   # both low   — common mode, correctable
    (+1, -1, 0),   # opposed    — differential, not correctable
    (-1, +1, 0),   # opposed    — differential, not correctable
]


def make_channel_drift(seed=9002, n_units=TRUE_N_DRIFT_UNITS):
    """800 two-channel units in four clouds: XOR with Gaussian noise.

    Returns a DataFrame with ``drift_a_mv``, ``drift_b_mv`` and
    ``correctable``. The labels carry no noise here — the point of this
    dataset is the *shape* of the problem, not its difficulty, and adding
    noise would only blur the hidden-layer picture the notebook draws.
    """
    rng = np.random.default_rng(seed)
    per_cloud = n_units // len(TRUE_DRIFT_QUADRANTS)

    frames = []
    for sign_a, sign_b, correctable in TRUE_DRIFT_QUADRANTS:
        frames.append(pd.DataFrame({
            "drift_a_mv": rng.normal(sign_a * TRUE_DRIFT_MAGNITUDE,
                                     TRUE_DRIFT_SPREAD, per_cloud),
            "drift_b_mv": rng.normal(sign_b * TRUE_DRIFT_MAGNITUDE,
                                     TRUE_DRIFT_SPREAD, per_cloud),
            "correctable": correctable,
        }))

    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Handwritten digits — Notebooks 02 and 03, the multiclass problem
# ---------------------------------------------------------------------------

TRUE_N_CLASSES = 10
DIGIT_IMAGE_SHAPE = (8, 8)
DIGIT_MAX_INTENSITY = 16.0      # pixels are integers 0..16


def load_digit_images(seed=9003, val_size=0.2, test_size=0.2):
    """The scikit-learn 8x8 handwritten digits, split three ways and scaled.

    This is the one genuinely *real* dataset in the lesson. scikit-learn
    bundles a copy, so nothing is downloaded. Pixels are integers from 0 to
    16; they are divided by 16 here so every input lies in [0, 1], which is
    what keeps the first layer's pre-activations in a sensible range.

    The split is stratified and three-way — train / validation / test — for
    the reason lesson 5 gives: the validation set is used repeatedly while
    choosing the architecture, so it stops being an unbiased estimate, and
    the test set is touched once at the very end.

    Returns ``(X_train, y_train, X_val, y_val, X_test, y_test)`` with X of
    shape (m, 64) and y integer class labels 0..9.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data / DIGIT_MAX_INTENSITY
    y = digits.target

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1.0 - test_size),
        stratify=y_rest, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test

#!/usr/bin/env python3
"""Recompute every number the lesson 9 handout works out by hand.

The rule this file lives by: **start from the raw inputs, never from the
handout's own intermediate values.** So the forward pass and the gradient step
are re-implemented here from the handout's formulas, the gradient formula is
checked against a numerical derivative, the hand-built network is pushed
through from its weights, and the fitted models are refitted from the data
module, at other seeds too where a result could be one lucky draw.

Run it directly, or let `tools/verify_lesson.py 09` discover it.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))

from instrument_data import (                                    # noqa: E402
    make_acceptance_test, make_channel_drift, load_digit_images,
    TRUE_GAIN_SIGMA_DB, TRUE_PHASE_SIGMA_DEG, TRUE_GAIN_TOL_DB, TRUE_PHASE_TOL_DEG,
    TRUE_RIG_ERROR_RATE,
)

CHECKS = 0
FAILURES = []


def check(label, computed, expected, tol, note=""):
    """Assert one handout number, and say how it was reached."""
    global CHECKS
    CHECKS += 1
    ok = abs(computed - expected) <= tol
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {label:<60s} {computed:>12.6f}  handout {expected:<10.6g}"
          f"{'  ' + note if note else ''}")
    if not ok:
        FAILURES.append(f"{label}: computed {computed!r}, handout says {expected!r}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =====================================================================
section("1-2. The acceptance test and one line")
check("gain tolerance in spreads, 0.50 / 0.40", TRUE_GAIN_TOL_DB / TRUE_GAIN_SIGMA_DB, 1.25, 1e-12)
check("phase tolerance in spreads, 3.75 / 3.00", TRUE_PHASE_TOL_DEG / TRUE_PHASE_SIGMA_DEG, 1.25, 1e-12)
check("the ceiling, 1 - rig error", 1 - TRUE_RIG_ERROR_RATE, 0.97, 1e-12)

# 2.1 the unit of the slides, at four points
w, b = np.array([1.0, 2.0]), -1.0
check("2.1 z at (1, 1)", float(np.array([1.0, 1.0]) @ w + b), 2.0, 1e-12)
check("2.1 sigma(2)", float(sigmoid(2.0)), 0.8808, 5e-5)
check("2.1 sigma(-1), the point (0, 0)", float(sigmoid(np.array([0.0, 0.0]) @ w + b)), 0.2689, 5e-5)
check("2.1 sigma at (1, 0), on the line", float(sigmoid(np.array([1.0, 0.0]) @ w + b)), 0.5, 1e-12)

from sklearn.exceptions import ConvergenceWarning                # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.model_selection import train_test_split             # noqa: E402
from sklearn.neural_network import MLPClassifier                 # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)

units = make_acceptance_test()
X = units[["gain_offset_db", "phase_offset_deg"]].values
y = units["accepted"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=9101)
check("2.2 training sensors", len(X_tr), 2250, 0)
check("2.2 test sensors", len(X_te), 750, 0)
scaler = StandardScaler().fit(X_tr)
Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
line = LogisticRegression().fit(Z_tr, y_tr)
line_acc = line.score(Z_te, y_te)
check("2.3 always predict 'accepted'", max(y_te.mean(), 1 - y_te.mean()), 0.5480, 5e-5)
check("2.3 logistic regression, test accuracy", line_acc, 0.5507, 5e-5)
check("2.3 its gain coefficient", line.coef_[0, 0], 0.0096, 5e-5)
check("2.3 its phase coefficient", line.coef_[0, 1], 0.0701, 5e-5)
check("2.3 its intercept", line.intercept_[0], 0.1892, 5e-5)


# =====================================================================
section("3. A hidden layer")
drift = make_channel_drift()
D = drift[["drift_a_mv", "drift_b_mv"]].values
d = drift["correctable"].values
check("3.1 logistic regression on the drift data", LogisticRegression().fit(D, d).score(D, d), 0.5, 1e-12)

relu = lambda z: np.maximum(0, z)
W1, b1 = np.array([[1.0, -1.0], [1.0, -1.0]]), np.array([-1.0, -1.0])
W2, b2 = np.array([10.0, 10.0]), -1.0
for (a_, b_), (h1, h2, s, yh) in zip(
        [(1, 1), (-1, -1), (1, -1), (-1, 1)],
        [(1, 0, 9, 0.9999), (0, 1, 9, 0.9999), (0, 0, -1, 0.2689), (0, 0, -1, 0.2689)]):
    h = relu(np.array([a_, b_], dtype=float) @ W1 + b1)
    out = h @ W2 + b2
    check(f"3.2 centre ({a_:+d}, {b_:+d}): h1", h[0], h1, 1e-12)
    check(f"3.2 centre ({a_:+d}, {b_:+d}): h2", h[1], h2, 1e-12)
    check(f"3.2 centre ({a_:+d}, {b_:+d}): 10h1 + 10h2 - 1", out, s, 1e-12)
    check(f"3.2 centre ({a_:+d}, {b_:+d}): y-hat", float(sigmoid(out)), yh, 5e-5)
hand = ((sigmoid(relu(D @ W1 + b1) @ W2 + b2) > 0.5) == d).mean()
check("3.2 hand-built network on all 800 sensors", hand, 0.9938, 5e-5)

def solved(width, seeds):
    return sum(MLPClassifier(hidden_layer_sizes=(width,), max_iter=3000, random_state=s)
               .fit(D, d).score(D, d) > 0.99 for s in seeds)


for width, count in ((2, 6), (4, 17), (8, 20)):
    check(f"3.4 drift, {width} units: runs of 20 reaching 0.99", solved(width, range(20)), count, 0)
# The claim is the ordering, so check it at twenty other starting points.
other = [solved(width, range(20, 40)) for width in (2, 4, 8)]
check("3.4 at 20 other seeds, wider succeeds more often", float(other[0] < other[1] <= other[2]), 1.0, 0,
      f"{other} of 20")

eight = np.mean([MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000, random_state=s)
                 .fit(Z_tr, y_tr).score(Z_te, y_te) for s in range(5)])
check("3.4 acceptance, 8 units, mean of 5 runs", eight, 0.9445, 5e-5)
eight_other = np.mean([MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000, random_state=s)
                       .fit(Z_tr, y_tr).score(Z_te, y_te) for s in range(5, 10)])
check("3.4 acceptance, 8 units, five other seeds", eight_other, 0.9445, 0.01, "same within a point")
check("3.4 within three points of the ceiling", 0.97 - eight, 0.0255, 5e-5)
check("1, 3.4 what the hidden layer is worth, points", 100 * (eight - line_acc), 39.4, 0.05)


# =====================================================================
section("4. One gradient step on one unit")
x, y0, alpha = np.array([1.0, 1.0]), 0.0, 0.5


def loss(weights, bias):
    p = sigmoid(x @ weights + bias)
    return -(y0 * np.log(p) + (1 - y0) * np.log(1 - p))


y_hat = sigmoid(x @ w + b)
check("4.3 forward: y-hat", float(y_hat), 0.881, 5e-4)
check("4.3 forward: loss", float(loss(w, b)), 2.127, 5e-4)
blame = y_hat - y0
# The formula dL/dw_j = (y-hat - y) x_j, against a numerical derivative of the loss.
h = 1e-6
numerical_w1 = (loss(w + [h, 0], b) - loss(w - [h, 0], b)) / (2 * h)
numerical_b = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
check("4.3 dL/dw1: (y-hat - y) x1 against a numerical derivative", float(blame * x[0]), float(numerical_w1), 1e-7)
check("4.3 dL/db: (y-hat - y) against a numerical derivative", float(blame), float(numerical_b), 1e-7)
check("4.3 the step, 0.5 x 0.881", float(alpha * blame), 0.440, 5e-4)
w_new, b_new = w - alpha * blame * x, b - alpha * blame
check("4.3 new w1", w_new[0], 0.5596, 5e-5)
check("4.3 new w2", w_new[1], 1.5596, 5e-5)
check("4.3 new b", b_new, -1.4404, 5e-5)
check("4.3 new z", float(x @ w_new + b_new), 0.679, 5e-4)
check("4.3 new y-hat", float(sigmoid(x @ w_new + b_new)), 0.663, 5e-4)
check("4.3 new loss", float(loss(w_new, b_new)), 1.089, 5e-4)

X_d, y_d, X_v, y_v, X_t, y_t = load_digit_images()
check("4.1, 5.1 training digits", len(X_d), 1077, 0)
check("5.1 validation digits", len(X_v), 360, 0)
check("5.1 test digits", len(X_t), 360, 0)
check("4.1 mini-batches of 32 per epoch", int(np.ceil(len(X_d) / 32)), 34, 0)
check("4.1 rounds of the loop in 60 epochs", 60 * int(np.ceil(len(X_d) / 32)), 2040, 0)


# =====================================================================
section("5-6. Counting, and the vanishing gradient")
check("5.2 hidden layer, 64 x 32 + 32", 64 * 32 + 32, 2080, 0)
check("5.2 output layer, 32 x 10 + 10", 32 * 10 + 10, 330, 0)
check("5.2, 5.5 one hidden layer of 32, total", 64 * 32 + 32 + 32 * 10 + 10, 2410, 0)
check("5.5 softmax alone, 64 x 10 + 10", 64 * 10 + 10, 650, 0)
check("5.5 one hidden layer of 64", 64 * 64 + 64 + 64 * 10 + 10, 4810, 0)
check("6.1 two layers of 512", 64 * 512 + 512 + 512 * 512 + 512 + 512 * 10 + 10, 301066, 0)
check("5.5 parameters, 2,410 against 650 (almost four times)", 2410 / 650, 3.71, 0.01)

zz = np.linspace(-8, 8, 160001)
slope = sigmoid(zz) * (1 - sigmoid(zz))
check("6.3 the sigmoid's largest slope", float(slope.max()), 0.25, 1e-9)
check("6.3 six layers: 0.25 ** 6 = 1/4096", 0.25 ** 6, 1 / 4096, 1e-15)
check("6.2 best minus worst remedy, under two sds", float(0.9481 - 0.9426 < 2 * 0.0035), 1.0, 0)

# 5.5 the digits gap, refitted in Keras when it is available (the course's full image).
try:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    import keras
    from keras import layers
    tf.config.experimental.enable_op_determinism()

    def digits(hidden, seed):
        keras.utils.set_random_seed(seed)
        m = keras.Sequential([keras.Input(shape=(64,))]
                             + [layers.Dense(u, activation="relu") for u in hidden]
                             + [layers.Dense(10, activation="softmax")])
        m.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(X_d, y_d, epochs=60, batch_size=32, verbose=0)
        return m.evaluate(X_v, y_v, verbose=0)[1]

    alone = np.mean([digits([], s) for s in range(3)])
    hidden32 = np.mean([digits([32], s) for s in range(3)])
    check("5.5 softmax alone, validation, mean of 3", alone, 0.9324, 5e-5)
    check("5.5 one hidden layer of 32, mean of 3", hidden32, 0.9602, 5e-5)
    check("1, 5.5 what the hidden layer is worth on digits, points", 100 * (hidden32 - alone), 2.8, 0.05)
    # Another split of the digits and other seeds: the handout says the exact figure
    # moves (4.7 there) while the order of magnitude against the sensors does not.
    Xd2, yd2, Xv2, yv2, _, _ = load_digit_images(seed=1234)

    def digits2(hidden, seed):
        keras.utils.set_random_seed(seed)
        m = keras.Sequential([keras.Input(shape=(64,))]
                             + [layers.Dense(u, activation="relu") for u in hidden]
                             + [layers.Dense(10, activation="softmax")])
        m.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(Xd2, yd2, epochs=60, batch_size=32, verbose=0)
        return m.evaluate(Xv2, yv2, verbose=0)[1]

    gap2 = 100 * (np.mean([digits2([32], s) for s in (10, 11, 12)])
                  - np.mean([digits2([], s) for s in (10, 11, 12)]))
    check("5.5 the digits gap on another split", gap2, 4.7, 0.05)
    for name, gap in (("this split", 100 * (hidden32 - alone)), ("another split", gap2)):
        check(f"5.5 'roughly ten times', sensors / digits ({name}), between 8 and 15",
              float(8 <= 100 * (eight - line_acc) / gap <= 15), 1.0, 0)
except ImportError:
    print("  (TensorFlow not installed here: the digits gap is checked in the course's full image)")


# =====================================================================
section("Result")
print(f"\n  {CHECKS} hand-worked numbers recomputed from the raw inputs.")
if FAILURES:
    print(f"  {len(FAILURES)} DISAGREE with the handout:\n")
    for failure in FAILURES:
        print(f"    - {failure}")
    sys.exit(1)
print("  All agree with the handout.\n")

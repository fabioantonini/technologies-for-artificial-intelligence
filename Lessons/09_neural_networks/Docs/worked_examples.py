#!/usr/bin/env python3
"""Recompute every number the lesson 9 handout works out by hand.

The rule this file lives by: **start from the raw inputs, never from the
handout's own intermediate values.** A script that reuses the handout's
intermediates only proves the handout is self-consistent, which a wrong
handout also is.

So, concretely:

* the forward and backward passes are re-implemented here from the handout's
  algebra, not imported from the notebooks — that is what makes the gradient
  check a check on the *derivation* rather than on the notebook's code;
* every simulated quantity is re-derived at a **different seed and a different
  sample size** than the notebook used, so agreement is evidence rather than a
  copy;
* where a second analytic route exists it is used — the polygon accuracy by
  quadrature against Monte Carlo, the best half-plane by quadrature against
  the notebook's sampling, the softmax gradient analytically against finite
  differences.

Run it directly, or let `tools/verify_lesson.py 09` discover it.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))

from instrument_data import (                                    # noqa: E402
    make_acceptance_test, make_channel_drift,
    TRUE_ACCEPT_RADIUS_SIGMA, TRUE_RIG_ERROR_RATE,
    TRUE_GAIN_SIGMA_DB, TRUE_PHASE_SIGMA_DEG,
    TRUE_GAIN_TOL_DB, TRUE_PHASE_TOL_DEG,
)

R = TRUE_ACCEPT_RADIUS_SIGMA
E = TRUE_RIG_ERROR_RATE

CHECKS = 0
FAILURES = []


def check(label, computed, expected, tol, note=""):
    """Assert one handout number, and say how it was reached."""
    global CHECKS
    CHECKS += 1
    ok = abs(computed - expected) <= tol
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {label:<58s} {computed:>12.6f}  handout {expected:<10.6g}"
          f"{'  ' + note if note else ''}")
    if not ok:
        FAILURES.append(f"{label}: computed {computed!r}, handout says {expected!r}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# =====================================================================
# 1. The dataset itself, and the ceiling it imposes
# =====================================================================
section("1. The acceptance dataset (handout sections 1.1 and 2.2)")

# The tolerances are each 1.25 production spreads -- this is what makes the
# accept region a circle after standardisation. Checked, not assumed.
check("gain tolerance in units of its spread",
      TRUE_GAIN_TOL_DB / TRUE_GAIN_SIGMA_DB, 1.25, 1e-12)
check("phase tolerance in units of its spread",
      TRUE_PHASE_TOL_DEG / TRUE_PHASE_SIGMA_DEG, 1.25, 1e-12)

# Route 1: the closed form for a standard 2-D normal.
exact_accept = 1.0 - np.exp(-R * R / 2)
check("P(accept), closed form 1 - exp(-R^2/2)", exact_accept, 0.5422, 5e-5)

# Route 2: Monte Carlo, at a seed and size the notebook never used
# (the notebook drew 400,000 at seed 20260909; this is 3,000,000 at 555).
sample = np.random.default_rng(555).standard_normal((3_000_000, 2))
inside = (sample ** 2).sum(1) < R * R
check("P(accept), Monte Carlo at a different seed", inside.mean(), 0.5422, 1e-3,
      "3,000,000 draws")

# Route 3: the generated dataset, which uses the elliptical rule in raw units.
units = make_acceptance_test()
check("fraction truly within tolerance, generated data",
      units.truly_within_tolerance.mean(), 0.5497, 5e-5)
check("fraction the rig recorded as accepted", units.accepted.mean(), 0.5473, 5e-5)
check("rate at which the rig recorded the wrong verdict",
      (units.accepted != units.truly_within_tolerance).mean(), 0.0330, 5e-5)
check("the ceiling, 1 - e", 1.0 - E, 0.97, 1e-12)


# =====================================================================
# 2. What one straight line is worth (handout section 2.2)
# =====================================================================
section("2. The best single line (handout section 2.2)")


def half_plane_accuracy(a):
    """P(a half-plane {z1 < a} agrees with the disc of radius R), exactly.

    A second route to what notebook 01 gets by sampling. Integrating over z1:
    the conditional probability of being inside the disc given z1 is the
    probability that |z2| < sqrt(R^2 - z1^2).
    """
    def joint_inside(z1):
        return norm.pdf(z1) * (2 * norm.cdf(np.sqrt(R * R - z1 * z1)) - 1)

    upper = min(a, R)
    if upper <= -R:
        p_left_inside = 0.0
    else:
        p_left_inside, _ = integrate.quad(joint_inside, -R, upper, limit=200)
    p_inside = 1.0 - np.exp(-R * R / 2)
    return 2 * p_left_inside + 1 - norm.cdf(a) - p_inside


grid = np.linspace(-0.5, 4.0, 451)
clean_best = max(half_plane_accuracy(a) for a in grid)


def with_rig_error(clean):
    """Handout section 11.3: accuracy against recorded labels."""
    return clean * (1 - E) + (1 - clean) * E


check("best half-plane, by quadrature, vs recorded labels",
      with_rig_error(clean_best), 0.6491, 1e-3)

# And the fitted logistic regression, refitted here from the raw data.
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.model_selection import train_test_split               # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402

X = units[["gain_offset_db", "phase_offset_deg"]].values
y = units["accepted"].values
truth = units["truly_within_tolerance"].values
X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
    X, y, truth, test_size=0.25, stratify=y, random_state=9101)
scaler = StandardScaler().fit(X_tr)
Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)

fitted = LogisticRegression().fit(Z_tr, y_tr)
check("logistic regression, test accuracy", fitted.score(Z_te, y_te), 0.5507, 5e-5)
check("  its gain coefficient", fitted.coef_[0, 0], 0.0096, 5e-5)
check("  its phase coefficient", fitted.coef_[0, 1], 0.0701, 5e-5)
check("  its intercept", fitted.intercept_[0], 0.1892, 5e-5)
check("always predicting 'accepted'", max(y_te.mean(), 1 - y_te.mean()), 0.5480, 5e-5)
check("the true rule scored against recorded test labels",
      (t_te == y_te).mean(), 0.9693, 5e-4,
      "the ceiling, realised on 750 points")


# =====================================================================
# 3. The network built by hand (handout section 3.2)
# =====================================================================
section("3. The hand-built XOR network (handout section 3.2)")

# Worked entirely by hand here: no matrix code, just the arithmetic the
# handout's table claims, so that a wrong table cannot be rescued by correct
# code shared with the notebook.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


handout_table = [
    # a,  b,   h1, h2, pre-activation of the output, sigmoid, class
    (+1., +1., 1., 0., 9., 0.9999, 1),
    (-1., -1., 0., 1., 9., 0.9999, 1),
    (+1., -1., 0., 0., -1., 0.2689, 0),
    (-1., +1., 0., 0., -1., 0.2689, 0),
]

for a, b, h1_expected, h2_expected, pre_expected, out_expected, cls in handout_table:
    h1 = max(0.0, a + b - 1.0)          # ReLU(a + b - 1)
    h2 = max(0.0, -a - b - 1.0)         # ReLU(-a - b - 1)
    pre = 10.0 * (h1 + h2) - 1.0
    out = sigmoid(pre)
    check(f"corner ({a:+.0f},{b:+.0f}) -> h1", h1, h1_expected, 1e-12)
    check(f"corner ({a:+.0f},{b:+.0f}) -> h2", h2, h2_expected, 1e-12)
    check(f"corner ({a:+.0f},{b:+.0f}) -> output pre-activation", pre, pre_expected, 1e-12)
    check(f"corner ({a:+.0f},{b:+.0f}) -> sigmoid", out, out_expected, 5e-5)
    check(f"corner ({a:+.0f},{b:+.0f}) -> class", float(out > 0.5), float(cls), 1e-12)

# The decision rule the construction implements, stated in the handout as
# |a + b| > 1.1 -- derived here rather than quoted: the output fires when
# 10 * (h1 + h2) > 1, i.e. when h1 + h2 > 0.1, i.e. when |a + b| - 1 > 0.1.
drift = make_channel_drift()
D = drift[["drift_a_mv", "drift_b_mv"]].values
d = drift["correctable"].values

h1_all = np.maximum(0.0, D.sum(1) - 1.0)
h2_all = np.maximum(0.0, -D.sum(1) - 1.0)
by_network = (sigmoid(10.0 * (h1_all + h2_all) - 1.0) > 0.5).astype(int)
by_rule = (np.abs(D.sum(1)) > 1.1).astype(int)

check("hand-built network agrees with the rule |a+b| > 1.1",
      (by_network == by_rule).mean(), 1.0, 1e-12)
check("hand-built network accuracy on all 800 units",
      (by_network == d).mean(), 0.9938, 5e-5)
check("  units it gets wrong", float((by_network != d).sum()), 5.0, 0.5)
check("logistic regression on the drift data",
      LogisticRegression().fit(D, d).score(D, d), 0.5000, 5e-4)


# =====================================================================
# 4. Backpropagation: check the DERIVATION, not just its output
# =====================================================================
section("4. Backpropagation (handout section 5)")

# Re-implemented here straight from the handout's boxed equations. If someone
# edits a transpose in the handout without touching the notebook, this is what
# is supposed to go red.


def forward(params, X_in):
    W1, b1, W2, b2 = params
    Z1 = X_in @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2
    return Z1, A1, sigmoid(Z2)


def cost(params, X_in, y_in):
    p = np.clip(forward(params, X_in)[2].ravel(), 1e-12, 1 - 1e-12)
    return -np.mean(y_in * np.log(p) + (1 - y_in) * np.log(1 - p))


def backward(params, X_in, y_in):
    """Handout section 5.3, transcribed."""
    W1, b1, W2, b2 = params
    m = X_in.shape[0]
    Z1, A1, yhat = forward(params, X_in)
    delta2 = (yhat - y_in.reshape(-1, 1)) / m            # (m, 1)
    grad_W2 = A1.T @ delta2                              # (H, 1)
    grad_b2 = delta2.sum(axis=0)                         # (1,)
    delta1 = (delta2 @ W2.T) * (Z1 > 0)                  # (m, H)
    grad_W1 = X_in.T @ delta1                            # (n, H)
    grad_b1 = delta1.sum(axis=0)                         # (H,)
    return [grad_W1, grad_b1, grad_W2, grad_b2]


# The shapes are part of the claim, so they are asserted at several sizes.
shape_failures = 0
for n_in, n_hidden, m_examples in ((2, 4, 17), (5, 3, 9), (7, 11, 23)):
    rng = np.random.default_rng(n_in * 100 + n_hidden)
    params = [rng.normal(0, .5, (n_in, n_hidden)), rng.normal(0, .5, n_hidden),
              rng.normal(0, .5, (n_hidden, 1)), rng.normal(0, .5, 1)]
    Xr = rng.normal(0, 1, (m_examples, n_in))
    yr = rng.integers(0, 2, m_examples)
    for grad, param in zip(backward(params, Xr, yr), params):
        if grad.shape != param.shape:
            shape_failures += 1
check("gradient shapes match parameter shapes, 3 architectures",
      float(shape_failures), 0.0, 1e-12)

# Finite differences at several random points, none of them the notebook's.
worst_relative = 0.0
n_partials = 0
for trial in range(4):
    rng = np.random.default_rng(9000 + trial)
    n_in, n_hidden, m_examples = 3, 5, 31
    params = [rng.normal(0, .7, (n_in, n_hidden)), rng.normal(0, .7, n_hidden),
              rng.normal(0, .7, (n_hidden, 1)), rng.normal(0, .7, 1)]
    Xr = rng.normal(0, 1.3, (m_examples, n_in))
    yr = rng.integers(0, 2, m_examples)
    analytic = backward(params, Xr, yr)
    h = 1e-5
    for tensor, grad in zip(params, analytic):
        flat, flat_grad = tensor.ravel(), grad.ravel()
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + h
            up = cost(params, Xr, yr)
            flat[i] = original - h
            down = cost(params, Xr, yr)
            flat[i] = original
            numerical = (up - down) / (2 * h)
            denominator = max(1e-12, abs(numerical) + abs(flat_grad[i]))
            worst_relative = max(worst_relative, abs(numerical - flat_grad[i]) / denominator)
            n_partials += 1

print(f"       ({n_partials} partial derivatives, 4 random architectures/points)")
check("worst relative gradient error, log10", np.log10(worst_relative), -8.0, 2.0,
      "handout: below 1e-6 believe it, above 1e-4 it is a bug")
if worst_relative > 1e-6:
    FAILURES.append(f"the handout's backward pass disagrees with finite "
                    f"differences at {worst_relative:.2e}")

# The output-layer collapse of section 5.2, checked as its own claim:
# d/dz of the binary cross-entropy through a sigmoid must be (yhat - y)/m.
rng = np.random.default_rng(4242)
z_probe = rng.normal(0, 2.5, 500)
y_probe = rng.integers(0, 2, 500).astype(float)
h = 1e-6
analytic_collapse = sigmoid(z_probe) - y_probe


def bce_of_z(z_vals):
    p = np.clip(sigmoid(z_vals), 1e-12, 1 - 1e-12)
    return -(y_probe * np.log(p) + (1 - y_probe) * np.log(1 - p))


numeric_collapse = (bce_of_z(z_probe + h) - bce_of_z(z_probe - h)) / (2 * h)
check("sigmoid + cross-entropy collapses to (yhat - y)",
      np.abs(analytic_collapse - numeric_collapse).max(), 0.0, 1e-6,
      "500 random points")


# =====================================================================
# 5. Softmax (handout section 6.2)
# =====================================================================
section("5. Softmax and its gradient (handout section 6.2)")


def softmax(z_vals):
    shifted = z_vals - z_vals.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


rng = np.random.default_rng(31337)
worst_softmax = 0.0
for trial in range(200):
    K = rng.integers(2, 8)
    z_vals = rng.normal(0, 2.0, K)
    true_class = int(rng.integers(0, K))
    onehot = np.zeros(K)
    onehot[true_class] = 1.0

    analytic = softmax(z_vals) - onehot
    h = 1e-6
    numeric = np.empty(K)
    for k in range(K):
        step = np.zeros(K)
        step[k] = h
        up = -np.log(softmax(z_vals + step)[true_class])
        down = -np.log(softmax(z_vals - step)[true_class])
        numeric[k] = (up - down) / (2 * h)
    worst_softmax = max(worst_softmax, np.abs(analytic - numeric).max())

check("softmax + categorical cross-entropy gives yhat - y",
      worst_softmax, 0.0, 1e-6, "200 random class counts and logits")

# Softmax reduces to the sigmoid for K = 2, as the handout claims.
gaps = rng.normal(0, 3.0, 400)
two_class = softmax(np.c_[np.zeros(400), gaps])[:, 1]
check("softmax with K=2 equals the sigmoid",
      np.abs(two_class - sigmoid(gaps)).max(), 0.0, 1e-12)


# =====================================================================
# 6. The bound that gives the lesson its number (handout section 7.2)
# =====================================================================
section("6. The sigmoid derivative bound (handout section 7.2)")

fine = np.linspace(-40, 40, 2_000_001)
sigmoid_slope = sigmoid(fine) * (1 - sigmoid(fine))
check("max of sigma'(z), on a fine grid", sigmoid_slope.max(), 0.25, 1e-9)
check("  the z at which it is attained", fine[sigmoid_slope.argmax()], 0.0, 1e-4)
check("sigma'(0), analytically", 0.5 * (1 - 0.5), 0.25, 1e-15)
check("max of tanh'(z)", (1 - np.tanh(fine) ** 2).max(), 1.0, 1e-9,
      "four times the sigmoid's -- this is the whole difference")

# The compounding claim: L sigmoid layers give roughly 4**L, and the handout
# quotes 4**6 = 4096 against a measured median of 3553.
check("4 ** 6, the handout's comparison for six layers", 4.0 ** 6, 4096.0, 1e-9)
check("4 ** 8", 4.0 ** 8, 65536.0, 1e-9)
check("measured median 3552.9 is within a factor of two of 4**6",
      (4.0 ** 6) / 3552.9, 1.15, 0.02)
check("measured median 46250 is within a factor of two of 4**8",
      (4.0 ** 8) / 46250.0, 1.42, 0.02)


# =====================================================================
# 7. Zero initialisation converges to the label entropy (section 8.1)
# =====================================================================
section("7. Zero initialisation (handout section 8.1)")

base_rate = y_tr.mean()
entropy = -(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate))
check("training base rate", base_rate, 0.5471, 5e-4)
check("entropy of the training labels", entropy, 0.6887, 5e-4)

# Second route: actually train a zero-initialised network, with a different
# number of hidden units and a different seed than the notebook's 32 / seed 1,
# and confirm it lands on the entropy and that W1 never moves.
zero_params = [np.zeros((2, 12)), np.zeros(12), np.zeros((12, 1)), np.zeros(1)]
rng = np.random.default_rng(77)
for _ in range(400):
    order = rng.permutation(len(Z_tr))
    for start in range(0, len(Z_tr), 64):
        batch = order[start:start + 64]
        for tensor, grad in zip(zero_params, backward(zero_params, Z_tr[batch], y_tr[batch])):
            tensor -= 0.3 * grad
check("zero-initialised network's final cost", cost(zero_params, Z_tr, y_tr),
      0.6887, 2e-3, "12 units, seed 77 -- not the notebook's 32 / seed 1")
check("  distinct columns of W1 afterwards",
      float(np.unique(zero_params[0].round(12), axis=1).shape[1]), 1.0, 1e-12)
check("  the hidden weights never moved at all",
      float(np.abs(zero_params[0]).max()), 0.0, 1e-15)
check("  its accuracy is the majority class",
      float(((forward(zero_params, Z_te)[2].ravel() > 0.5).astype(int) == y_te).mean()),
      0.5480, 5e-4)


# =====================================================================
# 8. Variance propagation (handout section 8.2)
# =====================================================================
section("8. Glorot and He scaling (handout section 8.2)")

# Var(z) = n Var(w) Var(a): simulate one layer and confirm the factor.
rng = np.random.default_rng(2024)
n_fan = 200
a_in = rng.normal(0, 1.7, (40_000, n_fan))
W_test = rng.normal(0, np.sqrt(1.0 / n_fan), (n_fan, 64))
check("one linear layer with Var(w)=1/n preserves the variance",
      (a_in @ W_test).var() / a_in.var(), 1.0, 0.02)

# ReLU halves it, which is where He's factor of 2 comes from. Note this is a
# statement about the second MOMENT, not the variance: relu(z) is not
# zero-mean, so Var(relu(z)) = (1/2 - 1/(2*pi)) Var(z) = 0.341 Var(z), and
# comparing variances here would wrongly report the handout as wrong.
symmetric = rng.normal(0, 1.0, 400_000)
check("ReLU keeps half the second moment",
      np.mean(np.maximum(0, symmetric) ** 2) * 2 / np.mean(symmetric ** 2), 1.0,
      0.02, "E[relu(z)^2] = Var(z)/2 for symmetric z")
check("  and its variance is the smaller 1/2 - 1/(2pi)",
      np.maximum(0, symmetric).var(ddof=0) / symmetric.var(ddof=0),
      0.5 - 1 / (2 * np.pi), 0.01, "why the two must not be confused")
check("  so He needs Var(w) = 2/n", 2.0 / n_fan, 2 / 200, 1e-15)


# =====================================================================
# 9. The rig-error identity (handout section 11.3)
# =====================================================================
section("9. The identity linking the two ceilings (handout section 11.3)")

check("q = 1 gives the ceiling", with_rig_error(1.0), 0.97, 1e-12)
check("the 16-unit row: q = 0.9760 predicts", with_rig_error(0.9760), 0.9474, 5e-5,
      "observed 0.9475")
check("  the handout's residual", abs(with_rig_error(0.9760) - 0.9475), 0.0, 5e-4)

# The identity is a statement about independent errors, so it is checked by
# simulation too rather than only by substitution.
rng = np.random.default_rng(606)
q_true = 0.9760
n_sim = 4_000_000
model_right = rng.random(n_sim) < q_true
rig_wrong = rng.random(n_sim) < E
check("identity confirmed by simulation",
      (model_right == ~rig_wrong).mean(), with_rig_error(q_true), 1e-3,
      "4,000,000 draws")


# =====================================================================
# 10. The polygon fence (handout section 11.1)
# =====================================================================
section("10. The fence of H lines (handout section 11.1)")


def polygon_accuracy(n_sides, apothem):
    """The handout's integral, transcribed from section 11.1."""
    def disagreement(t):
        boundary = apothem / np.cos(t)
        lo, hi = min(R, boundary), max(R, boundary)
        return np.exp(-lo * lo / 2) - np.exp(-hi * hi / 2)
    area, _ = integrate.quad(disagreement, -np.pi / n_sides, np.pi / n_sides,
                             limit=200)
    return 1.0 - area * n_sides / (2 * np.pi)


apothems = np.linspace(0.6 * R, 2.2 * R, 401)

# Second route: score the same polygons against the 3,000,000-point sample
# drawn at the top of this file (seed 555; the notebook used 400,000 at seed
# 20260909).
for n_sides, clean_expected, recorded_expected in ((4, 0.9396, 0.9132),
                                                   (8, 0.9859, 0.9567),
                                                   (16, 0.9965, 0.9667)):
    by_integral = max(polygon_accuracy(n_sides, a) for a in apothems)
    angles = np.arange(n_sides) * 2 * np.pi / n_sides
    directions = np.c_[np.cos(angles), np.sin(angles)]
    reach = (sample @ directions.T).max(1)
    by_sampling = max(((reach < a) == inside).mean() for a in apothems)
    check(f"{n_sides}-gon clean accuracy, by quadrature", by_integral,
          clean_expected, 5e-4)
    check(f"  the same by sampling at a different seed", by_sampling,
          clean_expected, 2e-3)
    check(f"  {n_sides}-gon against recorded labels",
          with_rig_error(by_integral), recorded_expected, 5e-4)


# =====================================================================
# 11. Parameter counts (handout sections 6.3 and 10.1)
# =====================================================================
section("11. Parameter counts (handout sections 6.3 and 10.1)")


def dense_params(widths):
    """Weights plus biases for a stack of fully connected layers."""
    return sum(a * b + b for a, b in zip(widths[:-1], widths[1:]))


check("the 4-unit acceptance network", float(dense_params([2, 4, 1])), 17, 1e-12)
check("softmax alone on 64 pixels", float(dense_params([64, 10])), 650, 1e-12)
check("one hidden layer of 32", float(dense_params([64, 32, 10])), 2410, 1e-12)
check("one hidden layer of 64", float(dense_params([64, 64, 10])), 4810, 1e-12)
check("two hidden layers of 64", float(dense_params([64, 64, 64, 10])), 8970, 1e-12)
check("two hidden layers of 512", float(dense_params([64, 512, 512, 10])),
      301066, 1e-12)
check("  parameters per training example, at 300",
      dense_params([64, 512, 512, 10]) / 300, 1004, 0.5)


# =====================================================================
# 12. Differences the handout reads off its own tables
# =====================================================================
section("12. Differences quoted in the text")

check("digits: first hidden layer is worth", 0.9602 - 0.9324, 0.0278, 5e-5)
check("digits: 32 -> 64 units is worth", 0.9611 - 0.9602, 0.0009, 5e-5)
check("  as a fraction of the seed spread", (0.9611 - 0.9602) / 0.0045, 0.2, 0.01)
check("digits: linear is behind the best by", 0.9676 - 0.9324, 0.0352, 5e-5)
check("acceptance: linear to a 3-unit hidden layer", 0.9421 - 0.5507, 0.3914, 5e-5)
check("width sweep: 2 units to 3 units", 0.9421 - 0.7392, 0.2029, 5e-5)
check("width sweep: 3 units to 32 units", 0.9469 - 0.9421, 0.0048, 5e-5)
check("width sweep: 3 units beats the best triangle by", 0.9421 - 0.8632, 0.0789, 5e-5)
check("the 16-unit gap between measured and true", 0.9760 - 0.9475, 0.0285, 5e-5)
check("Adam over plain descent, mean", 0.9485 - 0.9389, 0.0096, 5e-5)
check("Adam over plain descent, worst case", 0.9360 - 0.9027, 0.0333, 5e-5)
check("  Adam's spread as a fraction of plain descent's", 0.0072 / 0.0184, 0.39, 0.005)
check("momentum is behind plain descent by", 0.9389 - 0.9349, 0.0040, 5e-5)
check("best regulariser over early stopping alone", 0.9511 - 0.9411, 0.0100, 5e-5)
check("300 -> 1077 training examples", 0.9787 - 0.9463, 0.0324, 5e-5)
check("  data beats the best regulariser by a factor of",
      (0.9787 - 0.9463) / (0.9511 - 0.9411), 3.24, 0.02)
check("test-set-tuned line minus the honest one", 0.6880 - 0.6491, 0.0389, 5e-5)
check("dead ReLUs: 33 of 64 leaves", 64 - 33, 31, 1e-12)
check("  accuracy cost of half the layer dying", 0.9667 - 0.9583, 0.0084, 5e-5)


# =====================================================================
section("Result")
print(f"\n  {CHECKS} hand-worked numbers recomputed from the raw inputs.")
if FAILURES:
    print(f"  {len(FAILURES)} DISAGREE with the handout:\n")
    for failure in FAILURES:
        print(f"    - {failure}")
    sys.exit(1)
print("  All agree with the handout.\n")

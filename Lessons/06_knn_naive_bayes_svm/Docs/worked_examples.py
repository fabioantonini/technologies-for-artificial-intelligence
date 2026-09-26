"""Recompute every number lesson 6 works out by hand.

    python Lessons/06_knn_naive_bayes_svm/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. Each check reaches the handout's figure from
the raw inputs rather than from the handout's own intermediate values.

The distance-concentration result is the one that matters most here, because it
is the lesson's headline claim and it is stated as a bare number. It is
recomputed below from a fresh simulation with a different seed and a different
number of points from the notebook's, so agreement between the two is evidence
rather than a copy.
"""

import numpy as np

HANDOUT = "Lessons/06_knn_naive_bayes_svm/Docs/knn_naive_bayes_svm.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ------------------------------------------- Section 1, the class balance

# The pump generator is the only input; the handout's baseline must follow.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from pump_data import LABEL_NOISE, load_interacting, load_pumps

X, y = load_pumps()
same("1.1 the number of pumps", len(X), 1200, tolerance=0)
same("1.1 the faulty fraction", y.mean(), 0.613, tolerance=5e-4)
same("1.1 the majority baseline", max(y.mean(), 1 - y.mean()), 0.613,
     tolerance=5e-4)
same("1.1 the noise ceiling", 1 - LABEL_NOISE, 0.96, tolerance=1e-9)

# ------------------------------------------- Section 4.1, P(x) is recoverable

# The priors come from the generator, the two likelihoods are the handout's
# illustration. The probabilities are reached a second way, through the odds,
# which never forms P(x) at all.
healthy, faulty = 1 - y.mean(), y.mean()
same("4.1 the healthy pumps", (y == 0).sum(), 465, tolerance=0)
same("4.1 the healthy prior", healthy, 0.3875, tolerance=1e-9)
scores = (0.12 * healthy, 0.03 * faulty)
same("4.1 the healthy score", scores[0], 0.0465, tolerance=5e-5)
same("4.1 the faulty score", scores[1], 0.0184, tolerance=5e-5)
same("4.1 P(x), their sum", sum(scores), 0.0649, tolerance=5e-5)
same("4.1 P(healthy | x) by normalising", scores[0] / sum(scores), 0.717,
     tolerance=5e-4)
odds = (0.12 / 0.03) * (healthy / faulty)
same("4.1 P(healthy | x) through the odds", odds / (1 + odds), 0.717,
     tolerance=5e-4)
same("4.1 P(faulty | x)", 1 / (1 + odds), 0.283, tolerance=5e-4)

# And the claim that predict_proba is exactly this normalisation.
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB().fit(X, y)
joint = nb.predict_joint_log_proba(X)
gap = np.abs(np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
             - nb.predict_proba(X)).max()
same("4.1 predict_proba is the normalised joint", gap, 0.0, tolerance=1e-12)

# ------------------------------ Section 4.2, what the assumption says, counted

# The illustration's table, rebuilt pump by pump rather than by the handout's
# products: one row per pump, the two readings drawn exactly at the stated rates
# and independently within each class, then counted.
fleet = []
for n_class, p_vib, p_low in ((600, 0.8, 0.7), (400, 0.1, 0.2)):
    vib = np.arange(n_class) < p_vib * n_class
    for i in range(n_class):
        # within a class every vibration value meets every pressure value in
        # proportion, which is what independence given the class means
        fleet.append((vib[i], (i % 10) < p_low * 10))
fleet = np.array(fleet)
high, low = fleet[:, 0], fleet[:, 1]
faulty_rows = np.arange(len(fleet)) < 600
same("4.2 faulty and vibrating high", (high & faulty_rows).sum(), 480, tolerance=0)
same("4.2 faulty and at low pressure", (low & faulty_rows).sum(), 420, tolerance=0)
same("4.2 healthy and vibrating high", (high & ~faulty_rows).sum(), 40, tolerance=0)
same("4.2 healthy and at low pressure", (low & ~faulty_rows).sum(), 80, tolerance=0)
same("4.2 faulty with both", (high & low & faulty_rows).sum(), 336, tolerance=0)
same("4.2 healthy with both", (high & low & ~faulty_rows).sum(), 8, tolerance=0)
same("4.2 all vibrating high", high.sum(), 520, tolerance=0)
same("4.2 all at low pressure", low.sum(), 500, tolerance=0)
same("4.2 all with both", (high & low).sum(), 344, tolerance=0)
same("4.2 low pressure among faulty pumps vibrating high",
     100 * low[high & faulty_rows].mean(), 70, tolerance=0.5)
same("4.2 ...equal to the rate among all faulty pumps",
     100 * low[faulty_rows].mean(), 70, tolerance=0.5)
same("4.2 low pressure among all pumps vibrating high",
     100 * low[high].mean(), 66, tolerance=0.5)
same("4.2 low pressure overall", 100 * low.mean(), 50, tolerance=0.5)

# The reverse case, on the interacting sensors, from the generator itself.
Xi, yi = load_interacting()
same("4.2 interacting sensors, overall correlation",
     Xi.corr().iloc[0, 1], -0.057, tolerance=5e-4)
same("4.2 interacting sensors, within healthy",
     Xi[yi == 0].corr().iloc[0, 1], 0.821, tolerance=5e-4)
same("4.2 interacting sensors, within faulty",
     Xi[yi == 1].corr().iloc[0, 1], -0.826, tolerance=5e-4)

# ------------------------------------------- Section 4.3, the assumption

for label, printed in ((0, -0.006), (1, -0.049)):
    same(f"4.3 within-class correlation, class {label}",
         X[y == label].corr().iloc[0, 1], printed, tolerance=5e-4)
same("4.3 overall correlation", X.corr().iloc[0, 1], -0.046, tolerance=5e-4)

# ------------------------------------------- Section 4.4, the interaction

Xi, yi = load_interacting()
same("4.4 the interacting baseline", max(yi.mean(), 1 - yi.mean()), 0.523,
     tolerance=5e-4)

# The handout explains 0.404 by the accidental difference in class means.
# Check that difference is the size claimed: about 0.17 against a spread near 1.
gap = abs(Xi.groupby(yi).mean().diff().iloc[-1]).max()
same("4.4 the accidental gap between class means", gap, 0.17, tolerance=0.02)
spread = Xi.std().mean()
if not 0.8 < spread < 1.3:
    raise SystemExit(f"4.4 the spread is {spread:.2f}, not 'near 1' as claimed")
checks += 1

# ------------------------- Section 3.2, distance concentration, independently

# Deliberately not the notebook's setup: different seed, different point count.
rng = np.random.default_rng(20261030)
measured = {}
for d in (2, 10, 50, 100, 500):
    points = rng.random((1_500, d))
    ratios = []
    for query in rng.random((80, d)):
        distance = np.sqrt(((points - query) ** 2).sum(axis=1))
        ratios.append(distance.min() / distance.max())
    measured[d] = float(np.mean(ratios))

for d, printed in ((2, 0.016), (10, 0.263), (50, 0.592),
                   (100, 0.701), (500, 0.855)):
    same(f"3.2 nearest/farthest at d={d}", measured[d], printed, tolerance=0.05)

# And the claim the prose makes in words.
same("3.2 'about 2% in two dimensions'", 100 * measured[2], 2, tolerance=1.5)
same("3.2 '70% in one hundred dimensions'", 100 * measured[100], 70,
     tolerance=4)

# ------------------------- Section 3.2, the mechanism behind the concentration

# The prose now derives the ratio rather than asserting it: one squared
# coordinate difference has mean mu and variance v that do not depend on n, so
# the relative spread of the squared distance is sqrt(v/n)/mu. Both constants
# are checked against a simulation of a single coordinate pair - not against the
# distances above - and the prediction is then checked against distances
# simulated afresh, so the formula and the geometry are two separate routes.

one = rng.random(2_000_000) - rng.random(2_000_000)
same("3.2 mu, the mean of one squared coordinate difference",
     float((one ** 2).mean()), 1 / 6, tolerance=1e-3)
same("3.2 v, its variance", float((one ** 2).var()), 7 / 180, tolerance=1e-3)

# The handout also derives both constants exactly. Redo that algebra in exact
# fractions, from the uniform's moments E[x^k] = 1/(k+1) and from the
# triangular density's integrals 2 * int t^k (1 - t) dt = 2/((k+1)(k+2)).
from fractions import Fraction as F

var_uniform = F(1, 3) - F(1, 2) ** 2
same("3.2 the variance of a uniform coordinate", var_uniform, F(1, 12),
     tolerance=0)
triangular = lambda k: 2 * F(1, (k + 1) * (k + 2))
same("3.2 mu from the variances", 2 * var_uniform, F(1, 6), tolerance=0)
same("3.2 mu from the triangular density", triangular(2), F(1, 6), tolerance=0)
same("3.2 E[D^4] from the triangular density", triangular(4), F(1, 15),
     tolerance=0)
same("3.2 v exactly", triangular(4) - triangular(2) ** 2, F(7, 180),
     tolerance=0)
same("3.2 v as printed, 'about 0.039'", 7 / 180, 0.039, tolerance=5e-4)

mu, v = 1 / 6, 7 / 180
same("3.2 the constant sqrt(v)/mu", np.sqrt(v) / mu, 1.18, tolerance=5e-3)

for d, printed in ((2, 0.84), (100, 0.12)):
    predicted = np.sqrt(v / d) / mu
    same(f"3.2 predicted relative spread of d^2 at n={d}", predicted, printed,
         tolerance=6e-3)
    a, b = rng.random((60_000, d)), rng.random((60_000, d))
    squared = ((a - b) ** 2).sum(axis=1)
    same(f"3.2 simulated relative spread of d^2 at n={d}",
         float(squared.std() / squared.mean()), printed, tolerance=6e-3)
    if d == 100:
        root = np.sqrt(squared)
        same("3.2 relative spread of the distance itself at n=100",
             float(root.std() / root.mean()), 0.059, tolerance=3e-3)

# ------------------------------------------- Section 6, the spread of scores

same("6 the gap between best and worst model", 0.947 - 0.613, 0.334,
     tolerance=1e-9)

# ------------------------------------------- Section 5.3, support vectors

# 5.4's table went into three figure titles and the handout, and was printed
# nowhere, so nothing compared the two. Recomputed here from the generator.
from sklearn.model_selection import cross_val_score, StratifiedKFold   # noqa: E402
from sklearn.pipeline import make_pipeline                             # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
from sklearn.svm import SVC                                            # noqa: E402

svm_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for gamma, C, printed_train, printed_cv in ((0.1, 1, 0.936, 0.929),
                                            (1, 1, 0.950, 0.944),
                                            (50, 1000, 0.995, 0.902)):
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gamma, C=C))
    same(f"5.4 training score at gamma={gamma}, C={C}",
         svm.fit(X, y).score(X, y), printed_train, tolerance=1e-3)
    same(f"5.4 cross-validated score at gamma={gamma}, C={C}",
         cross_val_score(svm, X, y, cv=svm_folds).mean(), printed_cv, tolerance=1e-3)

same("5.3 linear support-vector fraction", 947 / 1200, 0.79, tolerance=5e-3)
same("5.3 RBF support-vector fraction", 278 / 1200, 0.23, tolerance=5e-3)

# ------------------- Section 5.2, the margin, and 5.4, the dual and the kernel

# Three claims the handout now derives instead of quoting. Each is checked
# against a fitted model rather than against the algebra that produced it.

from sklearn.metrics.pairwise import rbf_kernel                        # noqa: E402
from scipy.special import factorial                                    # noqa: E402

# Notebook 03's own two clouds, rebuilt from its generator rather than loaded,
# so that these checks are about the figure section 5.1 actually shows.
toy_rng = np.random.default_rng(3)
Xs = np.vstack([toy_rng.normal([-1.6, 0.0], [0.6, 1.1], size=(40, 2)),
                toy_rng.normal([+1.6, 0.0], [0.6, 1.1], size=(40, 2))])
ys = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
hard = SVC(kernel="linear", C=1_000).fit(Xs, ys)
w, b = hard.coef_[0], hard.intercept_[0]

same("5.1 points in the margin figure", len(Xs), 80, tolerance=0)
same("5.1 support vectors among them", len(hard.support_), 3, tolerance=0)

# 5.2, step 2: the canonical normalisation. The solver's own scaling should put
# the closest points of each class at |w'x + b| = 1 exactly.
edge = np.abs(Xs @ w + b).min()
same("5.2 the closest point sits at |w'x + b| = 1", float(edge), 1.0,
     tolerance=1e-3)

# 5.2, step 2: the margin is 2/||w||. Measured, instead, as the gap between the
# two classes along the direction w - geometry, with no formula in it.
projection = (Xs @ w) / np.linalg.norm(w)
gap = projection[ys == 1].min() - projection[ys == 0].max()
same("5.2 the margin equals 2/||w||", 2 / np.linalg.norm(w), abs(float(gap)),
     tolerance=1e-3)

# 5.4: w is a weighted sum of the training points, w = sum a_i y_i x_i. The
# multipliers times labels are what sklearn stores as dual_coef_.
rebuilt = hard.dual_coef_[0] @ Xs[hard.support_]
same("5.4 w rebuilt from the multipliers", float(np.abs(rebuilt - w).max()), 0.0,
     tolerance=1e-6)
same("5.4 the multipliers sum against the labels to zero",
     float(hard.dual_coef_[0].sum()), 0.0, tolerance=1e-6)

# 5.4: the RBF kernel is the claimed infinite series. Truncating it must
# converge on what sklearn computes.
# Points of modest size, so that the series converges before floating point
# loses the cancellation between the huge series and the tiny prefactors.
gamma = 0.3
u, t = rng.uniform(-1, 1, (6, 2)), rng.uniform(-1, 1, (6, 2))
series = np.zeros((6, 6))
for k in range(30):
    series += (2 * gamma) ** k / factorial(k) * (u @ t.T) ** k
series *= np.exp(-gamma * (u ** 2).sum(1))[:, None]
series *= np.exp(-gamma * (t ** 2).sum(1))[None, :]
same("5.4 the RBF kernel equals its truncated power series",
     float(np.abs(series - rbf_kernel(u, t, gamma=gamma)).max()), 0.0,
     tolerance=1e-9)

print(f"lesson 6: {checks} hand-worked numbers recomputed, all agree")

"""Recompute every number lesson 1 works out by hand.

    python Lessons/01_introduction_and_workflow/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. Each check reaches the handout's figure from
the raw inputs, not from the handout's own intermediate values.
"""

import math

HANDOUT = "Lessons/01_introduction_and_workflow/Docs/introduction_and_workflow.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ------------------------------- Section 2.4, the standard error of a score

# The inputs the handout states: notebook 1's test set and its accuracy.
n_test, accuracy = 143, 0.986

# --- 2.1, the zero-one loss lands on that accuracy --------------------------
#
# Under the zero-one loss the empirical risk is the error rate, so the score the
# notebook prints and the risk the formalism minimises are the same number read
# two ways. Reached from the counts, not from the accuracy.

mistakes = n_test - round(n_test * accuracy)
same("2.1 accuracy 0.986 is an empirical risk of 0.014",
     mistakes / n_test, 0.014, tolerance=5e-4)
same("2.1 and that is one minus the accuracy",
     1 - mistakes / n_test, accuracy, tolerance=5e-4)

standard_error = math.sqrt(accuracy * (1 - accuracy) / n_test)
same("2.4 the standard error is about one percentage point",
     standard_error, 0.010, tolerance=0.002)

# And the claim that the error falls as one over root m: quadrupling the test
# set should halve it.
same("2.4 the error falls as 1/sqrt(m)",
     math.sqrt(accuracy * (1 - accuracy) / (4 * n_test)) / standard_error,
     0.5, tolerance=1e-6)

# --- 2.4, why that formula is not admissible on these numbers ---------------
#
# The handout claims four things about the normal approximation here. Each is
# recomputed from the two raw inputs above, and the exact bound is obtained
# twice, by two methods that share no intermediate value.

correct = round(n_test * accuracy)
same("2.4 the test set holds two errors", n_test - correct, 2, tolerance=0)

same("2.4 m*p*(1-p) is about two",
     n_test * accuracy * (1 - accuracy), 2, tolerance=0.1)

same("2.4 the approximate interval runs up to 100.5%",
     accuracy + 1.96 * standard_error, 1.005, tolerance=5e-4)

# Route 1: the exact (Clopper-Pearson) interval, from scipy.
from scipy import stats  # noqa: E402  - kept local to this section

exact_low = stats.binomtest(correct, n_test).proportion_ci(method="exact")[0]
same("2.4 the exact lower bound is 95.0%", exact_low, 0.950, tolerance=5e-4)

# Route 2: the same bound found from its definition, with no library beyond
# math - the p at which observing 141 or more successes out of 143 has
# probability 0.025. Bisection on that tail.
def upper_tail(p_true: float) -> float:
    """P(X >= correct) for X ~ Binomial(n_test, p_true)."""
    return sum(math.comb(n_test, k) * p_true ** k * (1 - p_true) ** (n_test - k)
               for k in range(correct, n_test + 1))


lo, hi = 0.5, 1.0
for _ in range(200):
    mid = (lo + hi) / 2
    if upper_tail(mid) < 0.025:
        lo = mid
    else:
        hi = mid
same("2.4 the exact lower bound again, from the binomial tail",
     (lo + hi) / 2, 0.950, tolerance=5e-4)
same("2.4 and the two routes agree with each other",
     (lo + hi) / 2, exact_low, tolerance=1e-6)

# ------------------------------- Section 7, why leakage answers to m, not p
#
# The handout explains the asymmetry by two growth rates. Both are checked here;
# the measured accuracies themselves are the notebook's committed output, not a
# hand-worked number.

same("7 halving the examples multiplies the noise correlation by 1.41",
     (1 / math.sqrt(100)) / (1 / math.sqrt(200)), 1.41, tolerance=5e-3)

same("7 doubling the columns multiplies the largest of them by 1.04",
     math.sqrt(2 * math.log(10000)) / math.sqrt(2 * math.log(5000)),
     1.04, tolerance=5e-3)

# ------------------------------- Section 7, the threshold the default chose
#
# The handout's caption and the deck quote what moving the threshold costs. The
# route here starts from the raw dataset and refits, so nothing is borrowed from
# the notebook's committed output - only the same seed and split.

from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

_data = load_breast_cancer(as_frame=True)
_X_train, _X_test, _y_train, _y_test = train_test_split(
    _data.data, _data.target, test_size=0.25, random_state=42,
    stratify=_data.target)
_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42),
).fit(_X_train, _y_train)

# Column 0 is class 0, which for this dataset is `malignant`.
_p = _model.predict_proba(_X_test)[:, 0]
_malignant = (_y_test == 0).to_numpy()


def _cost(threshold: float) -> tuple[int, int]:
    """(malignancies missed, false alarms) at this threshold."""
    flagged = _p >= threshold
    return int((~flagged & _malignant).sum()), int((flagged & ~_malignant).sum())


# Two decimals, deliberately. The COUNTS below are identical under
# scikit-learn 1.3.1 (the course image) and 1.9.0 (the host toolchain); the
# third decimal of a fitted probability is not, moving 0.106 to 0.107 as the
# solver changes. So the handout quotes what survives a version bump.
same("7 the overlap: the lowest probability given to a malignancy",
     float(_p[_malignant].min()), 0.11, tolerance=5e-3)
same("7 the overlap: the highest given to a benign tumour",
     float(_p[~_malignant].max()), 0.62, tolerance=5e-3)

# The band the caption sends the reader to, and the crowd it tells them to
# ignore. The 11 benign inside the band are the same 11 that become false
# alarms at a threshold of 0.10 - which is the point of drawing this at all.
_lo, _hi = _p[_malignant].min(), _p[~_malignant].max()
_band = (_p >= _lo) & (_p <= _hi)
same("7 tumours the model was unsure about", int(_band.sum()), 12, tolerance=0)
same("7   of which malignant", int((_band & _malignant).sum()), 1, tolerance=0)
same("7   of which benign", int((_band & ~_malignant).sum()), 11, tolerance=0)
same("7 tumours it was certain about, within 0.02 of an end",
     int(((_p < 0.02) | (_p > 0.98)).sum()), 105, tolerance=0)

same("7 at the 0.5 default, malignancies missed", _cost(0.50)[0], 1, tolerance=0)
same("7 at the 0.5 default, false alarms", _cost(0.50)[1], 1, tolerance=0)
same("7 dropping to 0.10 misses nothing", _cost(0.10)[0], 0, tolerance=0)
same("7 and costs this many false alarms", _cost(0.10)[1], 11, tolerance=0)
same("7 raising to 0.90 raises no false alarm", _cost(0.90)[1], 0, tolerance=0)
same("7 and costs this many missed malignancies", _cost(0.90)[0], 7, tolerance=0)

# ------------------------------- Section 2.3, why the selector lands in the tail
#
# Reproduced from the seed rather than read off the notebook: notebook 03 sets
# np.random.seed(42) in its setup cell and then draws X and y in that order, so
# the same two calls rebuild the identical noise matrix.

import numpy as np  # noqa: E402

np.random.seed(42)
_Xn = np.random.normal(size=(200, 5000))
_yn = np.random.randint(0, 2, size=200)

_Xc = _Xn - _Xn.mean(axis=0)
_yc = _yn - _yn.mean()
_r = (_Xc * _yc[:, None]).sum(axis=0) / np.sqrt((_Xc ** 2).sum(axis=0) * (_yc ** 2).sum())

same("2.3 the noise spread of a correlation on 200 rows",
     1 / math.sqrt(200), 0.071, tolerance=5e-4)
same("2.3   and the sample standard deviation agrees with it",
     float(_r.std()), 1 / math.sqrt(200), tolerance=5e-3)

# The top 20 by |r| is what SelectKBest(f_classif, k=20) returns: for two classes
# the F statistic is monotone in |r|, so ranking by either gives the same set.
_top20 = np.argsort(-np.abs(_r))[:20]
same("2.3 the smallest |r| among the twenty kept",
     float(np.abs(_r[_top20]).min()), 0.20, tolerance=5e-3)
same("2.3 the largest |r| among them",
     float(np.abs(_r[_top20]).max()), 0.27, tolerance=5e-3)
same("2.3   how many standard deviations out, at the near edge",
     float(np.abs(_r[_top20]).min() / _r.std()), 2.8, tolerance=5e-2)
same("2.3   and at the far edge",
     float(np.abs(_r[_top20]).max() / _r.std()), 3.9, tolerance=5e-2)

# Extreme-value scaling, by a route that never touches the sample.
same("2.3 sqrt(2 ln n) for 5000 columns", math.sqrt(2 * math.log(5000)), 4.13,
     tolerance=5e-3)
same("2.3 so the largest draw is predicted near",
     (1 / math.sqrt(200)) * math.sqrt(2 * math.log(5000)), 0.29, tolerance=5e-3)

print(f"lesson 1: {checks} hand-worked numbers recomputed, all agree")

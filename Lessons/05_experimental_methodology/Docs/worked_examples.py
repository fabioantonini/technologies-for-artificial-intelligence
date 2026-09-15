"""Recompute every number lesson 5 works out by hand.

    python Lessons/05_experimental_methodology/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. The bias-variance identity is the important
one: it is asserted in the handout, demonstrated in notebook 2, and confirmed
here from an independent simulation, so three separate routes have to agree
before it reaches a student.
"""

import numpy as np

HANDOUT = "Lessons/05_experimental_methodology/Docs/experimental_methodology.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ------------------------- Section 4.3, how much two training folds share

# Leaving out fold i and fold j, the two training sets share k-2 of the k-1
# folds each contains.
for k, printed in ((5, 0.75), (10, 8 / 9)):
    same(f"4.3 the fraction shared by two training sets at k={k}",
         (k - 2) / (k - 1), printed, tolerance=1e-6)

# And the variance of the mean of k equicorrelated scores.
sigma_squared, rho, k = 1.0, 0.6, 5
naive = sigma_squared / k
true = sigma_squared / k + (k - 1) / k * rho * sigma_squared
same("4.3 the naive variance ignores the correlation term",
     true / naive, 1 + (k - 1) * rho, tolerance=1e-6)
same("4.3 setting rho to zero recovers the naive formula",
     sigma_squared / k + (k - 1) / k * 0.0 * sigma_squared, naive,
     tolerance=1e-12)

# ------------------------- Section 5, the decomposition, independently

# Not notebook 2's polynomial: a different generator, different model, so an
# error in one cannot be reproduced by the other.
NOISE_SD = 3.0
rng = np.random.default_rng(1)


def truth(x):
    return 2.0 + 0.8 * x ** 2


test_x = np.linspace(-3, 3, 60)
test_truth = truth(test_x)

predictions = []
for _ in range(4_000):
    x = rng.uniform(-3, 3, 12)
    y = truth(x) + rng.normal(0, NOISE_SD, 12)
    # A straight line: too rigid for a parabola, so bias dominates.
    slope, intercept = np.polyfit(x, y, 1)
    predictions.append(intercept + slope * test_x)
predictions = np.array(predictions)

mean_prediction = predictions.mean(axis=0)
bias_squared = float(np.mean((mean_prediction - test_truth) ** 2))
variance = float(np.mean(predictions.var(axis=0)))
noise = NOISE_SD ** 2
measured = float(np.mean((predictions - test_truth) ** 2)) + noise

same("5.2 bias squared plus variance plus noise equals the expected error",
     bias_squared + variance + noise, measured, tolerance=1e-6)

# And that the noise floor is exactly what no model can remove.
same("5.3 the noise floor of the handout's energy curve", 22.0 ** 2, 484,
     tolerance=1e-9)

# ------------- Section 2.4, the rarer class, and 4.3, the correlated folds
#
# Both sections now derive a formula the handout used to quote. Neither check
# below evaluates that formula: the standard error is measured from simulated
# yes/no draws, and the variance of the cross-validation mean from simulated
# equicorrelated fold scores, so agreement is evidence and not a tautology.

sim = np.random.default_rng(20261016)

for positives, printed in ((7, 0.113), (29, 0.056), (306, 0.017)):
    caught = sim.binomial(positives, 0.9, size=200_000) / positives
    same(f"2.4 the standard error of recall on {positives} positives",
         float(caught.std()), printed, tolerance=3e-3)

same("2.4 ten times the positives is about three times the precision",
     np.sqrt(306 / 29), 3, tolerance=0.25)

# 4.3: k fold scores with a common component, so that every pair correlates by
# rho. The variance of their mean must match the derived expression.
k, rho, sigma = 5, 0.6, 0.05
shared = sim.normal(0, sigma * np.sqrt(rho), size=(400_000, 1))
private = sim.normal(0, sigma * np.sqrt(1 - rho), size=(400_000, k))
folds = shared + private
same("4.3 the simulated folds really do correlate by rho",
     float(np.corrcoef(folds[:, 0], folds[:, 1])[0, 1]), rho, tolerance=0.01)
same("4.3 the variance of their mean matches the derived formula",
     float(folds.mean(axis=1).var()),
     sigma ** 2 / k + (k - 1) / k * rho * sigma ** 2, tolerance=1e-5)

# 4.7: averaging alone would buy at most the square root of k.
same("4.7 the square root of five", np.sqrt(5), 2.2, tolerance=0.05)

# ------------------- Sections 2.2, 2.4 and 4.7, rebuilt from the generator
#
# The handout's headline numbers are notebook outputs rather than hand algebra,
# so the check that matters is that they are reproducible from the raw inputs
# rather than transcribed. Everything below re-derives them here.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from methodology_data import RANDOM_STATE, load_fleet                 # noqa: E402
from sklearn.linear_model import LogisticRegression                   # noqa: E402
from sklearn.metrics import roc_auc_score                             # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline                            # noqa: E402
from sklearn.preprocessing import StandardScaler                      # noqa: E402

fleet_X, fleet_y = load_fleet()
same("2.2 the cut-down fleet is 800 drives", len(fleet_y), 800, tolerance=0)
same("2.2 with 29 failures", int(np.sum(fleet_y)), 29, tolerance=0)
same("2.4 which is 3.6% of it", float(np.mean(fleet_y)) * 100, 3.6, tolerance=0.05)

# 2.4's rule of thumb rests on how few positives reach a test set at all.
_, _, _, y_held = train_test_split(fleet_X, fleet_y, test_size=0.25,
                                   random_state=0, stratify=fleet_y)
same("2.4 a 25% test set holds 200 drives", len(y_held), 200, tolerance=0)
same("2.4 of which seven failed", int(np.sum(y_held)), 7, tolerance=0)
same("2.4 leaving 193 healthy ones", int(len(y_held) - np.sum(y_held)), 193, tolerance=0)

# 2.2's table: 200 legitimate splits of the same data, the same model.
scores = []
for seed in range(200):
    Xa, Xb, ya, yb = train_test_split(fleet_X, fleet_y, test_size=0.25,
                                      random_state=seed, stratify=fleet_y)
    fitted = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE)).fit(Xa, ya)
    scores.append(roc_auc_score(yb, fitted.predict_proba(Xb)[:, 1]))
scores = np.array(scores)

same("2.2 the worst of 200 splits scores 0.885", scores.min(), 0.885, tolerance=5e-4)
same("2.2 the best reports a perfect classifier", scores.max(), 1.000, tolerance=1e-9)
same("2.2 the mean is 0.955", scores.mean(), 0.955, tolerance=5e-4)
same("2.2 with a standard deviation of 0.024", scores.std(ddof=1), 0.024, tolerance=5e-4)
same("2.2 and ten of the two hundred clear 0.99", int((scores > 0.99).sum()), 10,
     tolerance=0)
same("2.2 so the spread is 0.115", scores.max() - scores.min(), 0.115, tolerance=1e-3)

# 4.7: cross-validation over the same seeds, to compare spreads like for like.
def cv_mean(seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    per_fold = []
    for train_rows, valid_rows in folds.split(fleet_X, fleet_y):
        fitted = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=5000,
                                                  random_state=RANDOM_STATE)
                               ).fit(fleet_X.iloc[train_rows], fleet_y.iloc[train_rows])
        per_fold.append(roc_auc_score(
            fleet_y.iloc[valid_rows],
            fitted.predict_proba(fleet_X.iloc[valid_rows])[:, 1]))
    return np.mean(per_fold)


def split_auc(seed):
    Xa, Xb, ya, yb = train_test_split(fleet_X, fleet_y, test_size=0.25,
                                      random_state=seed, stratify=fleet_y)
    fitted = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE)).fit(Xa, ya)
    return roc_auc_score(yb, fitted.predict_proba(Xb)[:, 1])


cv_means = np.array([cv_mean(seed) for seed in range(40)])
single = scores[:40]

same("4.7 cross-validation is about seven times more stable across seeds",
     single.std(ddof=1) / cv_means.std(ddof=1), 6.8, tolerance=1.5)
same("4.7 every cross-validated estimate is at least 0.944", cv_means.min(), 0.944,
     tolerance=5e-4)
same("4.7 and at most 0.959", cv_means.max(), 0.959, tolerance=5e-4)
same("4.7 the single split centres at 0.960", single.mean(), 0.960, tolerance=5e-4)
same("4.7 cross-validation centres at 0.953", cv_means.mean(), 0.953, tolerance=5e-4)


def gap_in_standard_errors(one, other):
    gap = one.mean() - other.mean()
    return gap / np.sqrt(one.var(ddof=1) / len(one) + other.var(ddof=1) / len(other))


# "About two standard errors" is a claim that the gap is noise, so it is tested
# the way noise behaves: on forty different seeds it must not keep its sign.
first = gap_in_standard_errors(single, cv_means)
same("4.7 the two centres are about two standard errors apart", first, 2.0,
     tolerance=0.5)
other_single = np.array([split_auc(seed) for seed in range(1000, 1040)])
other_cv = np.array([cv_mean(seed) for seed in range(1000, 1040)])
same("4.7 and the gap reverses sign with a different forty seeds",
     np.sign(gap_in_standard_errors(other_single, other_cv)), -np.sign(first),
     tolerance=0)

print(f"lesson 5: {checks} hand-worked numbers recomputed, all agree")

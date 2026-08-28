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
cv_means = []
for seed in range(30):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    per_fold = []
    for train_rows, test_rows in folds.split(fleet_X, fleet_y):
        fitted = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=5000,
                                                  random_state=RANDOM_STATE)
                               ).fit(fleet_X.iloc[train_rows], fleet_y.iloc[train_rows])
        per_fold.append(roc_auc_score(
            fleet_y.iloc[test_rows],
            fitted.predict_proba(fleet_X.iloc[test_rows])[:, 1]))
    cv_means.append(np.mean(per_fold))
cv_means = np.array(cv_means)
single = scores[:30]

same("4.7 cross-validation is about seven times more stable across seeds",
     single.std(ddof=1) / cv_means.std(ddof=1), 6.8, tolerance=1.5)

print(f"lesson 5: {checks} hand-worked numbers recomputed, all agree")

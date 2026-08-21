"""Recompute every number lesson 7 works out by hand.

    python Lessons/07_trees_and_ensembles/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. Each check reaches the handout's figure
from the raw inputs — ``loan_data.py``'s generator, or first principles —
rather than from the handout's own intermediate values. Where a second route
exists (a closed-form probability against a simulation, an independent
from-scratch split search against the notebook's), both are used.

Tolerances are looser here than in earlier lessons for anything that depends
on scikit-learn's internals (cross-validated accuracy, feature importance
shares): this script runs under whatever Python the verifier uses, which may
be a different scikit-learn version than the teaching container. Numbers
derived from pure arithmetic or from an independent from-scratch
implementation use tight tolerances throughout.
"""

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from loan_data import (LABEL_NOISE, TRUE_DEBT_CEIL, TRUE_INCOME_FLOOR,
                        load_loans, load_with_noise)

HANDOUT = "Lessons/07_trees_and_ensembles/Docs/trees_and_ensembles.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


def at_least(name: str, computed, floor) -> None:
    global checks
    checks += 1
    if computed < floor:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    expected at least {floor}, recomputing gives {computed}")


X, y = load_loans()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# ------------------------------------------- Section 1.1, the class balance

same("1.1 the number of applicants", len(X), 1200, tolerance=0)
same("1.1 the default rate", y.mean(), 0.387, tolerance=5e-4)
same("1.1 the majority baseline", max(y.mean(), 1 - y.mean()), 0.613, tolerance=5e-4)
same("1.1 the noise ceiling", 1 - LABEL_NOISE, 0.93, tolerance=1e-9)

# ------------------------------- Section 2.2, Gini impurity and best split

p = y.mean()
gini_parent = 1 - p ** 2 - (1 - p) ** 2
same("2.2 the parent Gini impurity", gini_parent, 0.474, tolerance=5e-4)
same("2.2 the two-class shortcut 2p(1-p)", 2 * p * (1 - p), gini_parent, tolerance=1e-12)


def gini(labels):
    if len(labels) == 0:
        return 0.0
    pc = np.bincount(labels, minlength=2) / len(labels)
    return 1.0 - np.sum(pc ** 2)


def best_split_independent(features, labels):
    """A from-scratch search, written independently of the notebook's."""
    best_gain, best_feature, best_threshold = -1.0, None, None
    parent = gini(labels)
    n = len(labels)
    for feature in features.columns:
        values = np.sort(features[feature].unique())
        for threshold in (values[:-1] + values[1:]) / 2:
            left = features[feature].values <= threshold
            if left.sum() == 0 or (~left).sum() == 0:
                continue
            weighted = (left.sum() * gini(labels[left])
                        + (~left).sum() * gini(labels[~left])) / n
            gain = parent - weighted
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feature, threshold
    return best_feature, best_threshold, best_gain


root_feature, root_threshold, _ = best_split_independent(X, y.values)
if root_feature != "debt_ratio" or abs(root_threshold - TRUE_DEBT_CEIL) > 0.02:
    raise SystemExit(
        f"{HANDOUT}: 2.2 the best root split\n"
        f"    expected debt_ratio near {TRUE_DEBT_CEIL}\n"
        f"    recomputing gives {root_feature} <= {root_threshold}")
checks += 1

# ------------------------------------- Section 3, depth is the bias-variance dial

cv_depth8 = cross_val_score(DecisionTreeClassifier(max_depth=8, random_state=0), X, y, cv=folds)
cv_unconstrained = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=folds)
same("3 depth-8 cross-validated accuracy", cv_depth8.mean(), 0.882, tolerance=0.02)
same("3 unconstrained cross-validated accuracy", cv_unconstrained.mean(), 0.852, tolerance=0.02)
train_unconstrained = DecisionTreeClassifier(random_state=0).fit(X, y).score(X, y)
same("3 unconstrained training accuracy", train_unconstrained, 1.000, tolerance=1e-9)
if not cv_depth8.mean() > cv_unconstrained.mean():
    raise SystemExit(f"{HANDOUT}: 3 the depth-8 tree should beat the unconstrained one on cv accuracy")
checks += 1

# --------------------------------------------- Section 4, instability

sample_a = X.sample(frac=0.65, random_state=1)
sample_b = X.sample(frac=0.65, random_state=2)
tree_a = DecisionTreeClassifier(max_depth=6, random_state=0).fit(sample_a, y.loc[sample_a.index])
tree_b = DecisionTreeClassifier(max_depth=6, random_state=0).fit(sample_b, y.loc[sample_b.index])
agreement = (tree_a.predict(X) == tree_b.predict(X)).mean()
same("4 the two resampled trees' agreement", agreement, 0.883, tolerance=0.02)
disagreement_pct = (1 - agreement) * 100
at_least("4 disagreement is at least the 'over a tenth' claimed", disagreement_pct, 10.0)

# ---------------------------------- Section 5.1, the bootstrap and OOB score

m = len(X)
theory = (1 - 1 / m) ** m
same("5.1 (1-1/m)^m at m=1200", theory, 0.3677, tolerance=5e-4)
same("5.1 the 1/e limit", np.exp(-1), 0.368, tolerance=5e-4)
same("5.1 the fraction kept, 1 - 1/e", 1 - np.exp(-1), 0.632, tolerance=5e-4)

rf_oob = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0).fit(X, y)
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0), X, y, cv=folds)
same("5.1 OOB score against cross-validated accuracy",
     rf_oob.oob_score_, rf_cv.mean(), tolerance=0.015)
same("5.1 OOB score against the handout's 0.9117", rf_oob.oob_score_, 0.9117, tolerance=0.02)

# The handout used to offer one seed's four-decimal match as the demonstration.
# It now says that match is luck, and THAT is the claim needing a check - one
# that runs the opposite way to a tolerance. Confirm some other seed disagrees
# materially: a check that could only ever pass would prove nothing.
oob_cv_gaps = []
for seed in range(6):
    _oob = RandomForestClassifier(n_estimators=150, oob_score=True,
                                  random_state=seed).fit(X, y).oob_score_
    _cv = cross_val_score(
        RandomForestClassifier(n_estimators=150, random_state=seed), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)).mean()
    oob_cv_gaps.append(abs(_oob - _cv))

at_least("5.1 four-decimal agreement is luck: some seed differs materially",
         max(oob_cv_gaps), 0.001)
same("5.1 typical |OOB - CV| gap across seeds",
     float(np.mean(oob_cv_gaps)), 0.0026, tolerance=0.004)

# ------------------------------- Section 5.2 and 6, variance reduction

# A different resample count from the notebook's 30, on the same recipe.
results = {"tree": [], "bagging": [], "rf": []}
for seed in range(15):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    results["tree"].append((tree.predict(Xte) == yte).mean())
    bag = BaggingClassifier(DecisionTreeClassifier(random_state=0),
                             n_estimators=60, random_state=0).fit(Xtr, ytr)
    results["bagging"].append((bag.predict(Xte) == yte).mean())
    rf = RandomForestClassifier(n_estimators=60, random_state=0).fit(Xtr, ytr)
    results["rf"].append((rf.predict(Xte) == yte).mean())

tree_std, bag_std, rf_std = (np.std(results[k]) for k in ("tree", "bagging", "rf"))
tree_mean, bag_mean, rf_mean = (np.mean(results[k]) for k in ("tree", "bagging", "rf"))
if not (tree_std > bag_std >= rf_std * 0.9):
    raise SystemExit(
        f"{HANDOUT}: 5.2/6 variance should shrink tree -> bagging -> random forest, "
        f"got stds {tree_std:.4f}, {bag_std:.4f}, {rf_std:.4f}")
checks += 1
if not (tree_mean < bag_mean <= rf_mean + 0.01):
    raise SystemExit(
        f"{HANDOUT}: 5.2/6 mean accuracy should rise tree -> bagging -> random forest, "
        f"got means {tree_mean:.4f}, {bag_mean:.4f}, {rf_mean:.4f}")
checks += 1

# --------------------------------------- Section 7, feature importance and noise

k_considered = int(np.sqrt(22))
same("7 features considered per split, sqrt(22)", k_considered, 4, tolerance=0)
p_included_symmetry = comb(21, k_considered - 1) / comb(22, k_considered)
same("7 P(included) via combinatorics equals k/p",
     p_included_symmetry, k_considered / 22, tolerance=1e-9)
same("7 P(a real feature is a candidate)", k_considered / 22, 0.182, tolerance=5e-3)
same("7 P(a real feature is excluded)", 1 - k_considered / 22, 0.818, tolerance=5e-3)

Xn, yn = load_with_noise(20)
rf20 = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xn, yn)
cv20 = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0),
                        Xn, yn, cv=folds)
importances = pd.Series(rf20.feature_importances_, index=Xn.columns)
noise_share = importances.filter(like="noise").sum()
at_least("7 over half the importance mass sits on 20 noise columns", noise_share, 0.5)
same("7 cross-validated accuracy with 20 noise columns", cv20.mean(), 0.837, tolerance=0.02)

# -------------------------------------- Section 8, the boosting toy

rng = np.random.default_rng(7)
x_toy = np.sort(rng.uniform(0, 10, 200))
true_function = (np.where(x_toy < 3, 1.0, np.where(x_toy < 7, 3.5, 1.5))
                  + 0.6 * np.sin(x_toy))
y_toy = true_function + rng.normal(0, 0.25, size=len(x_toy))
X_toy = x_toy.reshape(-1, 1)

alpha, prediction = 0.3, np.full_like(y_toy, y_toy.mean())
mse_at = {}
for t in range(1, 61):
    residual = y_toy - prediction
    tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_toy, residual)
    prediction = prediction + alpha * tree.predict(X_toy)
    if t in (1, 5, 20, 60):
        mse_at[t] = float(np.mean((prediction - true_function) ** 2))

same("8 boosting toy MSE after 1 tree", mse_at[1], 0.395, tolerance=0.01)
same("8 boosting toy MSE after 5 trees", mse_at[5], 0.068, tolerance=0.01)
same("8 boosting toy MSE after 20 trees", mse_at[20], 0.012, tolerance=0.01)
same("8 boosting toy MSE after 60 trees", mse_at[60], 0.018, tolerance=0.01)
if not mse_at[60] > mse_at[20]:
    raise SystemExit(f"{HANDOUT}: 8 error should rise again by 60 trees (overfitting to noise)")
checks += 1

# --------------------------------------- Section 9, log-loss gradient, symbolically

# p - y is the derivative of log-loss w.r.t. F; check it numerically at a few points.
for F, label in ((0.5, 1), (-1.2, 0), (2.0, 1)):
    p_hat = 1 / (1 + np.exp(-F))
    eps = 1e-6
    p_plus = 1 / (1 + np.exp(-(F + eps)))
    p_minus = 1 / (1 + np.exp(-(F - eps)))
    def logloss(pp, yy):
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    numerical_grad = (logloss(p_plus, label) - logloss(p_minus, label)) / (2 * eps)
    analytic_grad = p_hat - label
    same(f"9 dL/dF at F={F}, y={label}", analytic_grad, numerical_grad, tolerance=1e-4)

cv_default_gbm = cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=folds)
same("9 default gradient boosting cross-validated accuracy", cv_default_gbm.mean(), 0.902, tolerance=0.02)

# --------------------------------------- Section 11, the leaderboard ordering

leaderboard = {
    "logreg": None,
    "tree_unconstrained": DecisionTreeClassifier(random_state=0),
    "tree_depth8": DecisionTreeClassifier(max_depth=8, random_state=0),
    "bagging": BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=60, random_state=0),
    "rf": RandomForestClassifier(n_estimators=60, random_state=0),
    "gbm": GradientBoostingClassifier(n_estimators=30, learning_rate=0.3, max_depth=3, random_state=0),
}
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scores = {"majority": max(y.mean(), 1 - y.mean())}
for name, model in leaderboard.items():
    if name == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000))
    scores[name] = cross_val_score(model, X, y, cv=folds).mean()

order = ["majority", "logreg", "tree_unconstrained", "tree_depth8", "gbm", "bagging", "rf"]
for a, b in zip(order, order[1:]):
    if not scores[a] < scores[b] + 0.015:
        raise SystemExit(
            f"{HANDOUT}: 11 leaderboard ordering broken between {a} ({scores[a]:.3f}) "
            f"and {b} ({scores[b]:.3f})")
checks += 1

print(f"{checks} numbers recomputed from raw inputs and confirmed against "
      f"{HANDOUT}.")

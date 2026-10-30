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

# Section 1.1, the line logistic regression draws. Found by bisection on the
# fitted model's own probabilities, not from its coefficients as the notebook does.
from sklearn.linear_model import LogisticRegression                      # noqa: E402
from sklearn.pipeline import make_pipeline                               # noqa: E402
from sklearn.preprocessing import StandardScaler                         # noqa: E402
_line = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)).fit(X, y)


def _debt_on_line(income):
    lo, hi = -1.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p_mid = _line.predict_proba(pd.DataFrame({"income_k": [income], "debt_ratio": [mid]}))[0, 1]
        lo, hi = (mid, hi) if p_mid < 0.5 else (lo, mid)
    return (lo + hi) / 2


same("1.1 the logistic line at income 20", _debt_on_line(20), 0.40, tolerance=0.006)
same("1.1 the logistic line at income 100", _debt_on_line(100), 0.88, tolerance=0.006)
_inc, _debt = X["income_k"].values, X["debt_ratio"].values
_floor_low = (_inc < TRUE_INCOME_FLOOR) & (_debt <= TRUE_DEBT_CEIL)
same("1.1 applicants below the floor and the ceiling", _floor_low.sum(), 87, tolerance=0)
same("1.1 of them predicted repaid", (_line.predict(X)[_floor_low] == 0).sum(), 50, tolerance=0)

# Section 2.2, infinitely many thresholds and finitely many splits. Counted with
# pandas and the default counts directly, not with the gini() helper.
_n_debt, _n_income = X["debt_ratio"].nunique(), X["income_k"].nunique()
same("2.2 distinct debt ratios", _n_debt, 1139, tolerance=0)
same("2.2 distinct incomes", _n_income, 1097, tolerance=0)
same("2.2 candidates at the root", (_n_debt - 1) + (_n_income - 1), 2234, tolerance=0)
_below = X.loc[X["debt_ratio"] <= 0.82, "debt_ratio"].max()
_above = X.loc[X["debt_ratio"] > 0.82, "debt_ratio"].min()
same("2.2 the neighbour below 0.82", _below, 0.8198, tolerance=1e-9)
same("2.2 the neighbour above 0.82", _above, 0.8205, tolerance=1e-9)
same("2.2 their midpoint", (_below + _above) / 2, 0.82015, tolerance=1e-9)
for _t in (0.8199, 0.8200, 0.8204):
    same(f"2.2 threshold {_t} sends 971 left", int((X["debt_ratio"] <= _t).sum()), 971, tolerance=0)


def _weighted_from_counts(mask):
    total = len(y)
    out = 0.0
    for side in (mask, ~mask):
        m_side, d_side = int(side.sum()), int(y.values[side].sum())
        out += m_side / total * 2 * (d_side / m_side) * (1 - d_side / m_side)
    return out


for _feature, _t, _left, _pl, _pr, _g in (("debt_ratio", 0.5, 627, 0.18, 0.61, 0.382),
                                         ("debt_ratio", 0.82, 971, 0.26, 0.94, 0.330),
                                         ("debt_ratio", 0.9, 1071, 0.32, 0.92, 0.405),
                                         ("income_k", 28, 111, 0.91, 0.33, 0.418)):
    _mask = X[_feature].values <= _t
    same(f"2.2 {_feature} <= {_t}: left count", int(_mask.sum()), _left, tolerance=0)
    same(f"2.2 {_feature} <= {_t}: left default rate", y.values[_mask].mean(), _pl, tolerance=0.005)
    same(f"2.2 {_feature} <= {_t}: right default rate", y.values[~_mask].mean(), _pr, tolerance=0.005)
    same(f"2.2 {_feature} <= {_t}: weighted Gini", _weighted_from_counts(_mask), _g, tolerance=5e-4)
_best = min(_weighted_from_counts(X[f].values <= t)
            for f in ("debt_ratio", "income_k")
            for t in (lambda v: (v[:-1] + v[1:]) / 2)(np.sort(X[f].unique())))
same("2.2 the lowest weighted Gini of all 2,234 candidates", _best, 0.330, tolerance=5e-4)

# Section 2.1-2.2, the root split counted, and the gains of step 2 found by the
# independent search run inside each child (not read off scikit-learn's tree).
_root_left = X["debt_ratio"].values <= 0.82
same("2.2 defaults in the parent", int(y.sum()), 464, tolerance=0)
same("2.2 defaults on the left", int(y.values[_root_left].sum()), 249, tolerance=0)
same("2.2 defaults on the right", int(y.values[~_root_left].sum()), 215, tolerance=0)
same("2.2 the left weight", 971 / 1200, 0.809, tolerance=5e-4)
same("2.2 the right weight", 229 / 1200, 0.191, tolerance=5e-4)
same("2.2 the left term", 971 / 1200 * gini(y.values[_root_left]), 0.3086, tolerance=5e-5)
same("2.2 the right term", 229 / 1200 * gini(y.values[~_root_left]), 0.0219, tolerance=5e-5)
same("2.2 the 229 before their split, by 1 - sum of squares",
     1 - (215 / 229) ** 2 - (14 / 229) ** 2, 0.115, tolerance=5e-4)
same("2.2 the 229's sum of squares", 0.939 ** 2 + 0.061 ** 2, 0.885, tolerance=5e-4)
_, _, _gain_left = best_split_independent(X[_root_left].reset_index(drop=True),
                                          y.values[_root_left])
_, _, _gain_right = best_split_independent(X[~_root_left].reset_index(drop=True),
                                           y.values[~_root_left])
same("2.1 step 2, the income question's gain", _gain_left, 0.0836, tolerance=5e-5)
same("2.1 step 2, the debt question's gain", _gain_right, 0.0026, tolerance=5e-5)
_d2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
_d3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
_changed = np.flatnonzero(_d2.predict(X) != _d3.predict(X))
same("2.1 step 3 changes one prediction", len(_changed), 1, tolerance=0)
same("2.1 training accuracy at depth 2", _d2.score(X, y), 0.8400, tolerance=5e-5)
same("2.1 training accuracy at depth 3", _d3.score(X, y), 0.8408, tolerance=5e-5)
_r = _changed[0]
same("2.1 the changed applicant's income", X["income_k"].iloc[_r], 23.22, tolerance=1e-9)
same("2.1 the changed applicant's debt", X["debt_ratio"].iloc[_r], 0.9475, tolerance=1e-9)
if not (X["income_k"].iloc[_r] < TRUE_INCOME_FLOOR and X["debt_ratio"].iloc[_r] > TRUE_DEBT_CEIL
        and y.iloc[_r] == 0):
    raise SystemExit(f"{HANDOUT}: 2.1 the changed applicant should be a flipped label")
checks += 1

# Batch 3 (30 Oct): the slips, the folds, the leaves, the budgeted tree, the two trees.
same("2.2 the slip: 1 minus the weighted average", 1 - 0.3305, 0.6695, tolerance=1e-9)
same("2.2 the slip: a count rebuilt from 26%", 0.26 * 971, 252, tolerance=0.5)
same("2.2 the true left share", 249 / 971, 0.256, tolerance=5e-4)
same("2.2 the children's p, weighted, is the parent's",
     971 / 1200 * (249 / 971) + 229 / 1200 * (215 / 229), 464 / 1200, tolerance=1e-12)
same("2.2 the rounded version drifts to 0.386", 0.809 * 0.256 + 0.191 * 0.939, 0.386, tolerance=5e-4)
same("2.2 249 + 215 over 1,200", (249 + 215) / 1200, 0.387, tolerance=5e-4)
_f8 = cross_val_score(DecisionTreeClassifier(max_depth=8, random_state=0), X, y, cv=folds)
for _k, _c in enumerate((210, 211, 215, 214, 209)):
    same(f"3 depth 8, fold {_k + 1} correct of 240", round(_f8[_k] * 240), _c, tolerance=0)
same("3 depth 8, the folds' standard deviation", _f8.std(), 0.010, tolerance=5e-4)
from loan_data import _true_default                                    # noqa: E402
_rule = _true_default(X["income_k"].values, X["debt_ratio"].values).astype(int)
same("3 flipped labels in this sample", int((_rule != y.values).sum()), 85, tolerance=0)
same("3 the true rule's score", (_rule == y.values).mean(), 0.929, tolerance=5e-4)
for _d, _leaves, _median, _ones in ((2, 4, 123.5, 0), (None, 156, 2, 72)):
    _t = DecisionTreeClassifier(max_depth=_d, random_state=0).fit(X, y)
    _sizes = _t.tree_.n_node_samples[_t.tree_.children_left < 0]
    same(f"3 depth {_d}: leaves", len(_sizes), _leaves, tolerance=0)
    same(f"3 depth {_d}: median leaf", float(np.median(_sizes)), _median, tolerance=0)
    same(f"3 depth {_d}: one-applicant leaves", int((_sizes == 1).sum()), _ones, tolerance=0)
same("3 the unconstrained tree's real depth",
     DecisionTreeClassifier(random_state=0).fit(X, y).get_depth(), 24, tolerance=0)
_cv9 = cross_val_score(DecisionTreeClassifier(max_leaf_nodes=9, random_state=0), X, y, cv=folds)
same("4 nine leaves, cross-validated mean", _cv9.mean(), 0.901, tolerance=0.02)
same("4 nine leaves, cross-validated spread", _cv9.std(), 0.020, tolerance=0.01)
_sa = X.sample(frac=0.65, random_state=1)
_sb = X.sample(frac=0.65, random_state=2)
same("4 each sample holds 780", len(_sa), 780, tolerance=0)
same("4 the two samples share 499", len(set(_sa.index) & set(_sb.index)), 499, tolerance=0)
_ta = DecisionTreeClassifier(max_depth=6, random_state=0).fit(_sa, y.loc[_sa.index])
_tb = DecisionTreeClassifier(max_depth=6, random_state=0).fit(_sb, y.loc[_sb.index])
same("4 opposite verdicts", int((_ta.predict(X) != _tb.predict(X)).sum()), 140, tolerance=3)

# Section 2.2 now says what G *is*: the chance two draws from the group, labelled
# by its own frequencies, disagree. Checked by drawing pairs, not by the formula.
_rng7 = np.random.default_rng(20261106)
for share in (0.387, 0.1, 0.5):
    labels = _rng7.random(200_000) < share
    partner = _rng7.random(200_000) < share
    same(f"2.2 two draws at p={share} disagree as often as G says",
         float(np.mean(labels != partner)), 2 * share * (1 - share),
         tolerance=5e-3)

# Section 7's cancellation: the share of subsets containing a fixed feature.
from math import comb                                                   # noqa: E402
for p_total, k_drawn in ((22, 4), (10, 3), (7, 1)):
    same(f"7 one of {p_total} features is a candidate in {k_drawn} draws",
         comb(p_total - 1, k_drawn - 1) / comb(p_total, k_drawn),
         k_drawn / p_total, tolerance=1e-12)
same("7 which for this forest is 18.2%", 100 * 4 / 22, 18.2, tolerance=0.05)

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

# ---------------------------------------- 1.1, the ceiling is an average
# Regenerate the draws exactly as the loader makes them, apply the noise-free rule,
# and count how many observed labels disagree with it.
import loan_data as _ld                                                 # noqa: E402
_rng = np.random.default_rng(_ld.RANDOM_STATE)
_inc, _debt = _rng.uniform(20.0, 100.0, 1200), _rng.uniform(0.0, 1.0, 1200)
_clean = _ld._true_default(_inc, _debt)
same("1.1 labels flipped in this sample", int((_clean != y.to_numpy()).sum()), 85, tolerance=0)
same("1.1 the true rule scores 1,115 / 1,200", (_clean == y.to_numpy()).mean(), 0.929, tolerance=5e-4)

# ---------------------------------------- 3, a depth-2 tree asks three questions
d2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y).tree_
same("3 the depth-2 tree has three internal nodes", int((d2.feature >= 0).sum()), 3, tolerance=0)
right = d2.children_right[0]
leaves_right = [d2.children_left[right], d2.children_right[right]]
same("3 the right-hand split's two leaves both predict default",
     sum(int(d2.value[l, 0].argmax() == 1) for l in leaves_right), 2, tolerance=0)
same("3 that split divides 229 applicants", int(d2.n_node_samples[right]), 229, tolerance=0)

# ---------------------------------------- 5, bagging: averaging probabilities is voting
Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
bag_b = BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=100,
                          random_state=1).fit(Xtr_b, ytr_b)
probs = np.array([t.predict_proba(Xte_b.to_numpy()[:, f])[:, 1]
                  for t, f in zip(bag_b.estimators_, bag_b.estimators_features_)])
same("5 fully grown trees give only probabilities 0 or 1",
     float(((probs > 0) & (probs < 1)).mean()), 0.0, tolerance=0)
votes = (probs.mean(axis=0) > 0.5).astype(int)
same("5 scikit-learn's averaging equals a majority vote", (votes == bag_b.predict(Xte_b)).mean(),
     1.0, tolerance=0)
same("5 the test split has 360 applicants", len(yte_b), 360, tolerance=0)
same("5 BaggingClassifier test accuracy", (bag_b.predict(Xte_b) == yte_b.to_numpy()).mean(),
     0.872, tolerance=5e-4)
# the from-scratch version: m draws with replacement per tree, majority vote
_rng = np.random.default_rng(1)
_trees = []
for _ in range(100):
    idx = _rng.integers(0, len(Xtr_b), size=len(Xtr_b))
    _trees.append(DecisionTreeClassifier(random_state=int(_rng.integers(0, 10_000)))
                  .fit(Xtr_b.iloc[idx], ytr_b.iloc[idx]))
_vote = (np.stack([t.predict(Xte_b) for t in _trees]).mean(axis=0) >= 0.5).astype(int)
same("5 own bagging and BaggingClassifier agree on every test applicant",
     (_vote == bag_b.predict(Xte_b)).mean(), 1.0, tolerance=0)

# ---------------------------------------- 5.1, draws against distinct rows
distinct = 1200 * (1 - (1 - 1 / 1200) ** 1200)
same("5.1 about 759 distinct rows in one bootstrap sample", distinct, 759, tolerance=0.5)
same("5.1 the other 441 draws are repeats", 1200 - round(distinct), 441, tolerance=0)

# ---------------------------------------- 7, the full menu, and the permutation check
Xn20, yn20 = load_with_noise(20)
full = RandomForestClassifier(n_estimators=300, max_features=None, random_state=0)
same("7 cross-validated accuracy with every column on offer",
     cross_val_score(full, Xn20, yn20, cv=folds).mean(), 0.897, tolerance=5e-3)
share_full = full.fit(Xn20, yn20).feature_importances_[2:].sum()
same("7 noise share with every column on offer", share_full, 0.28, tolerance=0.01)

from sklearn.inspection import permutation_importance                  # noqa: E402
Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(Xn20, yn20, test_size=0.3, random_state=0,
                                              stratify=yn20)
forest_p = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr_p, ytr_p)
# a second route to permutation importance: shuffle by hand, 20 times per column
base = (forest_p.predict(Xte_p) == yte_p.to_numpy()).mean()
_rng = np.random.default_rng(0)
def lost(col):
    out = []
    for _ in range(20):
        Xs = Xte_p.copy()
        Xs[col] = _rng.permutation(Xs[col].to_numpy())
        out.append(base - (forest_p.predict(Xs) == yte_p.to_numpy()).mean())
    return float(np.mean(out))
same("7.1 shuffling debt_ratio costs 0.18", lost("debt_ratio"), 0.18, tolerance=0.01)
same("7.1 shuffling income_k costs 0.09", lost("income_k"), 0.09, tolerance=0.01)
noise_lost = [lost(c) for c in Xn20.columns[2:]]
same("7.1 no noise column costs more than 0.003", max(abs(v) for v in noise_lost), 0.0, tolerance=0.004)
# the training rows: there the noise columns do cost accuracy
perm_train = permutation_importance(forest_p, Xtr_p, ytr_p, n_repeats=5, random_state=0)
at_least("7.1 on the training rows the noise columns cost something",
         perm_train.importances_mean[2:].sum(), 0.01)

# ---------------------------------------- 9, scikit-learn's leaf values are a Newton step
gb1 = GradientBoostingClassifier(n_estimators=1, learning_rate=1.0, max_depth=2,
                                 random_state=0).fit(X, y)
leaf = gb1.estimators_[0, 0].apply(X.to_numpy())
p0 = y.mean(); r = y.to_numpy() - p0
for l in np.unique(leaf):
    m_l = leaf == l
    same(f"9 leaf {l} holds sum(y - p) / sum(p(1 - p))",
         gb1.estimators_[0, 0].tree_.value[l, 0, 0], r[m_l].sum() / (p0 * (1 - p0) * m_l.sum()),
         tolerance=1e-6)

# ---------------------------------------- 10, the loss falls at every tree, accuracy not
from sklearn.metrics import log_loss                                    # noqa: E402
gb800 = GradientBoostingClassifier(n_estimators=800, learning_rate=0.3, max_depth=3,
                                   random_state=0).fit(X, y)
losses = [log_loss(y, q) for q in gb800.staged_predict_proba(X)]
accs = [(q == y.to_numpy()).mean() for q in gb800.staged_predict(X)]
same("10 the training log-loss never rises", int((np.diff(losses) > 1e-12).sum()), 0, tolerance=0)
at_least("10 training accuracy dips at some steps", int((np.diff(accs) < 0).sum()), 1)

# ---------------------------------------- 2.2, the root split's gain, by counting
yy = y.to_numpy(); left = X["debt_ratio"].to_numpy() <= 0.82
G2 = lambda q: 2 * q * (1 - q)
same("2.2 left child size", int(left.sum()), 971, tolerance=0)
same("2.2 right child size", int((~left).sum()), 229, tolerance=0)
same("2.2 left p", yy[left].mean(), 0.256, tolerance=5e-4)
same("2.2 right p", yy[~left].mean(), 0.939, tolerance=5e-4)
same("2.2 parent G", G2(yy.mean()), 0.4743, tolerance=5e-5)
same("2.2 left G", G2(yy[left].mean()), 0.3814, tolerance=5e-5)
same("2.2 right G", G2(yy[~left].mean()), 0.1148, tolerance=5e-5)
weighted = left.mean() * G2(yy[left].mean()) + (~left).mean() * G2(yy[~left].mean())
same("2.2 weighted children", weighted, 0.3305, tolerance=5e-5)
same("2.2 the gain", G2(yy.mean()) - weighted, 0.1438, tolerance=5e-5)
# a second route: sklearn's own impurity decrease for a depth-1 tree
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
t = stump.tree_
gain_sk = t.impurity[0] - (t.weighted_n_node_samples[1] * t.impurity[1]
                           + t.weighted_n_node_samples[2] * t.impurity[2]) / t.weighted_n_node_samples[0]
same("2.2 the gain, from scikit-learn's stump", gain_sk, 0.1438, tolerance=5e-5)

# ---------------------------------------- 5.1, the bootstrap count distribution
from math import exp, factorial
for k, printed in ((0, 0.368), (1, 0.368), (2, 0.184), (3, 0.061)):
    same(f"5.1 limit share drawn {k} times", exp(-1) / factorial(k), printed, tolerance=5e-4)
same("5.1 'about one in four' drawn twice or more", 1 - 2 * exp(-1), 0.25, tolerance=0.02)

# ---------------------------------------- 5.2, the floor at rho = 0.5
same("5.2 a hundred trees at rho 0.5", 1 / 100 + 99 / 100 * 0.5, 0.505, tolerance=1e-9)

# ---------------------------------------- 7, both real features left out
same("7 C(20,4)", comb(20, 4), 4845, tolerance=0)
same("7 C(22,4)", comb(22, 4), 7315, tolerance=0)
same("7 neither real feature offered", comb(20, 4) / comb(22, 4), 0.662, tolerance=5e-4)
sim = np.random.default_rng(7)
hits = np.mean([not ({0, 1} & set(sim.choice(22, 4, replace=False))) for _ in range(40000)])
same("7 ...and by simulation", hits, 0.662, tolerance=0.01)

# ---------------------------------------- 9, the starting score and first residuals
p0 = yy.mean()
same("9 F0, the base rate's log-odds", np.log(p0 / (1 - p0)), -0.461, tolerance=5e-4)
same("9 first residual, defaulter", 1 - p0, 0.613, tolerance=5e-4)
same("9 first residual, repayer", -p0, -0.387, tolerance=5e-4)
init = GradientBoostingClassifier(n_estimators=1, random_state=0).fit(X, y)
same("9 F0, from scikit-learn's initial estimator",
     float(init.init_.predict_proba(X.iloc[:1])[0, 1]), 0.387, tolerance=5e-4)

print(f"{checks} numbers recomputed from raw inputs and confirmed against "
      f"{HANDOUT}.")

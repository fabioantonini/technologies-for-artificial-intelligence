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

# ------------------------------------- Section 5.3, the two variance columns
#
# The handout's claim is a comparison, not a figure: a misspecified straight
# line pivots with the sample of temperatures, so its variance beats a more
# flexible model's, and holding the design fixed removes the effect. Both
# columns are rebuilt here from the generator, and the ordering is then checked
# again on a seed the notebook never used - a comparison that held on one draw
# would be worth nothing.
from methodology_data import (NOISE_SD, TEST_TEMPERATURE, TEST_TRUTH,  # noqa: E402
                              sample_energy, sample_energy_at)
from sklearn.linear_model import LinearRegression                     # noqa: E402
from sklearn.preprocessing import PolynomialFeatures                  # noqa: E402

BV_DEGREES = [1, 2, 3, 5, 9, 12]


def universes(degree, redraw_temperatures, seed, n_train=25, n_samples=300):
    rng = np.random.default_rng(seed)
    # The fixed design spends its first draw on the temperatures; the redrawing
    # one must not, or the two run on different streams and neither reproduces
    # the notebook it is checking.
    fixed_temperature = (None if redraw_temperatures
                         else rng.uniform(-5, 38, n_train))
    predictions = []
    for _ in range(n_samples):
        if redraw_temperatures:
            temperature, energy = sample_energy(n_train, rng)
        else:
            temperature = fixed_temperature
            energy = sample_energy_at(temperature, rng)
        model = make_pipeline(PolynomialFeatures(degree),
                              LinearRegression()).fit(temperature.reshape(-1, 1),
                                                      energy)
        predictions.append(model.predict(TEST_TEMPERATURE.reshape(-1, 1)))
    predictions = np.array(predictions)
    return (float(np.mean((predictions.mean(axis=0) - TEST_TRUTH) ** 2)),
            float(np.mean(predictions.var(axis=0))))


redrawn = {degree: universes(degree, True, 0) for degree in BV_DEGREES}
fixed = {degree: universes(degree, False, 0)[1] for degree in BV_DEGREES}

same("5.3 degree 1's bias squared", redrawn[1][0], 5250.8, tolerance=0.1)
same("5.3 degree 1's variance, temperatures redrawn", redrawn[1][1], 723.8,
     tolerance=0.1)
same("5.3 which is larger than degree 2's 70.8", redrawn[2][1], 70.8,
     tolerance=0.1)
same("5.3 with the temperatures held fixed degree 1 falls to 33.9", fixed[1], 33.9,
     tolerance=0.1)
same("5.3 and degree 12 to 2,669.8", fixed[12], 2669.8, tolerance=0.1)
same("5.3 the fixed-design variance climbs at every degree",
     sum(1 for a, b in zip(BV_DEGREES, BV_DEGREES[1:]) if fixed[b] > fixed[a]),
     len(BV_DEGREES) - 1, tolerance=0)

# On another seed the digits move; what must survive is the comparison itself.
other_redrawn = {degree: universes(degree, True, 11)[1] for degree in (1, 2)}
other_fixed = {degree: universes(degree, False, 11)[1] for degree in BV_DEGREES}
same("5.3 the straight line is still the noisier of the two on another seed",
     float(other_redrawn[1] > other_redrawn[2]), 1.0, tolerance=0)
same("5.3 and holding the design fixed still orders them the other way",
     sum(1 for a, b in zip(BV_DEGREES, BV_DEGREES[1:])
         if other_fixed[b] > other_fixed[a]), len(BV_DEGREES) - 1, tolerance=0)

# 5.3's other asymmetry: bias squared is lowest in the middle, not at the end.
bias_by_degree = {degree: redrawn[degree][0] for degree in BV_DEGREES}
same("5.3 bias squared is smallest at degree 5",
     min(bias_by_degree, key=bias_by_degree.get), 5, tolerance=0)
same("5.3 and degree 12's is the worst in the table",
     max(bias_by_degree, key=bias_by_degree.get), 12, tolerance=0)

# 5.3: degree 12's two numbers are unstable measurements. The noise an average of
# 300 curves carries into bias squared is variance / 300; and other seeds give
# very different values for both columns.
same("5.3 degree 12's bias squared, seed 0", redrawn[12][0], 153537.3, tolerance=0.1)
same("5.3 degree 12's variance, seed 0", redrawn[12][1], 32174344.9, tolerance=0.1)
same("5.3 variance / 300 is about 107,000", redrawn[12][1] / 300, 107_000,
     tolerance=500)
degree12_other = [universes(12, True, seed) for seed in range(1, 6)]
same("5.3 seeds 1-5: bias squared from 39,016",
     min(b for b, _ in degree12_other), 39016, tolerance=0.5)
same("5.3 to 1,566,449", max(b for b, _ in degree12_other), 1566449, tolerance=0.5)
same("5.3 variance from 2.8 million", min(v for _, v in degree12_other) / 1e6, 2.8,
     tolerance=0.05)
same("5.3 to 547 million", max(v for _, v in degree12_other) / 1e6, 547, tolerance=0.5)

# 2.4: the healthy count that specificity rests on; 4.6: readings in training.
same("2.4 healthy drives 800 - 29 = 771", 800 - 29, 771, tolerance=0)
same("4.6 ten readings, four folds of five in training: 8", 10 * 4 / 5, 8, tolerance=0)

# --------------------------------------- Section 6, the shape of a curve
#
# Section 6.1 reads a slope, which is the kind of claim that survives being
# wrong: every number around it stays correct. So the curve is rebuilt here and
# the claim is tested as a claim - where the peak is, whether the end lies below
# it, and whether the final step up is larger than the fold-to-fold scatter.
import pandas as pd                                                    # noqa: E402
from sklearn.model_selection import learning_curve                     # noqa: E402

wide_rng = np.random.default_rng(0)
wide_noise = pd.DataFrame(
    wide_rng.normal(size=(len(fleet_X), 150)),
    columns=[f"noise_{i}" for i in range(150)], index=fleet_X.index)
wide_X = pd.concat([fleet_X, wide_noise], axis=1)

counts, _, valid = learning_curve(
    make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=5000, C=1000.0,
                                     random_state=RANDOM_STATE)),
    wide_X, fleet_y, train_sizes=np.linspace(0.15, 1.0, 8),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    scoring="roc_auc")
valid_mean = valid.mean(axis=1)

same("6.1 the validation curve peaks at 251 examples",
     int(counts[valid_mean.argmax()]), 251, tolerance=0)
same("6.1 at 0.857", valid_mean.max(), 0.857, tolerance=5e-4)
same("6.1 and ends lower, at 0.847", valid_mean[-1], 0.847, tolerance=5e-4)
same("6.1 so the end is below the peak",
     float(valid_mean[-1] < valid_mean.argmax() * 0 + valid_mean.max()), 1.0,
     tolerance=0)

# The last segment rises by 0.014. Whether that is a slope is a question about
# its scatter, so ask the five folds rather than the average.
last_step = valid[-1] - valid[-2]
same("6.1 the final step averages +0.014", last_step.mean(), 0.014, tolerance=5e-4)
same("6.1 with folds disagreeing on its sign",
     float(last_step.min() < 0 < last_step.max()), 1.0, tolerance=0)
same("6.1 by more than the step itself",
     float(last_step.std(ddof=1) > abs(last_step.mean())), 1.0, tolerance=0)

# And the plateau is not a property of one fold seed.
for fold_seed in (1, 2):
    other = learning_curve(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000, C=1000.0,
                                         random_state=RANDOM_STATE)),
        wide_X, fleet_y, train_sizes=np.linspace(0.15, 1.0, 8),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed),
        scoring="roc_auc")[2].mean(axis=1)
    same(f"6.1 the curve still stops climbing well before the end, fold seed {fold_seed}",
         float(int(counts[other.argmax()]) < counts[-1]), 1.0, tolerance=0)

# ------------------------------------------- Section 3, which C is the strong one
#
# The handout now warns that scikit-learn's C runs opposite to lesson 3's
# lambda. That is a claim about the library, so ask the library: the coefficient
# vector must shrink as C falls.
norms = []
for c_value in (0.01, 1.0, 100.0):
    fitted = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, C=c_value,
                                             random_state=RANDOM_STATE)
                          ).fit(fleet_X, fleet_y)
    norms.append(float(np.linalg.norm(fitted[-1].coef_)))

same("3 a smaller C really is the heavier penalty",
     float(norms[0] < norms[1] < norms[2]), 1.0, tolerance=0)
same("3 the heavy penalty shrinks the coefficient vector to 0.52", norms[0], 0.52,
     tolerance=5e-3)
same("3 against 2.45 with almost none", norms[2], 2.45, tolerance=5e-3)

# ------------------------------- Section 2.4, one failing drive moves 20 pairs
#
# Not the formula 20 / (7 x 193) again: build a ranking of 7 failing and 193
# healthy drives, move one failing drive below 20 more healthy ones, and ask
# roc_auc_score how much the AUC fell.
labels = np.r_[np.ones(7), np.zeros(193)]
ranking = np.r_[np.arange(300, 307), np.arange(100, 293)].astype(float)
before = roc_auc_score(labels, ranking)
moved = ranking.copy()
moved[0] = 272.5          # now 273..292, 20 healthy drives, sit above it
after = roc_auc_score(labels, moved)
same("2.4 there are 7 x 193 = 1,351 failing-healthy pairs", 7 * 193, 1351, tolerance=0)
same("2.4 slipping below 20 more healthy drives costs 0.015 of AUC",
     before - after, 0.015, tolerance=5e-4)
same("2.4 which is exactly 20 pairs out of 1,351", (before - after) * 1351, 20,
     tolerance=1e-9)

# ---------------------------------- Section 8, lesson 4's threshold, honestly
#
# Rebuilt from the fleet: the three-way split, the model, and the bill at each
# threshold counted as missed failures and false alarms.
from sklearn.pipeline import Pipeline                               # noqa: E402

X_rest, X_test8, y_rest, y_test8 = train_test_split(
    fleet_X, fleet_y, test_size=0.25, random_state=0, stratify=fleet_y)
X_fit, X_valid8, y_fit, y_valid8 = train_test_split(
    X_rest, y_rest, test_size=0.25, random_state=0, stratify=y_rest)
same("8 the test set holds 200 drives", len(y_test8), 200, tolerance=0)
same("8 7 of them failing", int(y_test8.sum()), 7, tolerance=0)
same("8 the validation set holds 150 drives", len(y_valid8), 150, tolerance=0)
same("8 5 of them failing", int(y_valid8.sum()), 5, tolerance=0)
same("8 and the model is fitted on 450", len(y_fit), 450, tolerance=0)

threshold_model = Pipeline([("scale", StandardScaler()),
                            ("model", LogisticRegression(max_iter=2_000))]
                           ).fit(X_fit, y_fit)
test_p = threshold_model.predict_proba(X_test8)[:, 1]
grid = np.linspace(0.01, 0.9, 200)
valid_p = threshold_model.predict_proba(X_valid8)[:, 1]


def bill(y_true, p, t):
    flagged = p >= t
    return (int((~flagged & (y_true == 1)).sum()),
            int((flagged & (y_true == 0)).sum()))


chosen = grid[int(np.argmin([2600 * bill(y_valid8, valid_p, t)[0]
                             + 140 * bill(y_valid8, valid_p, t)[1] for t in grid]))]
cheat = grid[int(np.argmin([2600 * bill(y_test8, test_p, t)[0]
                            + 140 * bill(y_test8, test_p, t)[1] for t in grid]))]
same("8 the validation set chooses 0.050", chosen, 0.050, tolerance=5e-4)
same("8 the test set would choose 0.171", cheat, 0.171, tolerance=5e-4)
same("8 next to lesson 4's t* = 140 / 2,740 = 0.051", 140 / 2740, 0.051,
     tolerance=5e-4)
for t, missed, alarms, cost in ((0.5, 4, 1, 10_540), (chosen, 1, 15, 4_700),
                                (cheat, 1, 5, 3_300)):
    got = bill(y_test8, test_p, t)
    same(f"8 at {t:.3f}: {missed} missed", got[0], missed, tolerance=0)
    same(f"8 at {t:.3f}: {alarms} false alarms", got[1], alarms, tolerance=0)
    same(f"8 at {t:.3f}: {cost:,} EUR", 2600 * got[0] + 140 * got[1], cost,
         tolerance=0)

# ------------------------- Sections 2.1, 2.2, 2.4: the small arithmetic
#
# Each from its inputs, not from the printed result.
same("2.1 the poll of forty: standard error of a proportion near one half",
     float(np.sqrt(0.5 * 0.5 / 40)), 0.079, tolerance=5e-4)
same("2.2 lesson 4's 25% test set: 306 x 0.25", 306 * 0.25, 76.5, tolerance=1e-9)
same("2.2 the histogram's bins are about 0.004 wide", (1.000 - 0.885) / 28, 0.004,
     tolerance=5e-4)
same("2.4 seven failures of 200 is 3.5%", 7 / 200 * 100, 3.5, tolerance=1e-9)
same("2.4 twenty pairs turned wrong: 1,331 of 1,351", 1331 / 1351, 0.985,
     tolerance=5e-4)

# ------------------------- Section 3: wins, ties and the winning margin
#
# The notebook's strict ">" once gave every tie to the first model listed. Here
# ties are found directly, as equal AUCs on the same split.
from collections import Counter                                       # noqa: E402


def three_models(seeds):
    rows = []
    for seed in seeds:
        Xa, Xb, ya, yb = train_test_split(fleet_X, fleet_y, test_size=0.25,
                                          random_state=seed, stratify=fleet_y)
        rows.append([roc_auc_score(yb, make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2_000, C=c_value)
        ).fit(Xa, ya).predict_proba(Xb)[:, 1]) for c_value in (0.01, 1.0, 100.0)])
    rows = np.array(rows)
    top = rows == rows.max(axis=1, keepdims=True)
    tied = top.sum(axis=1) > 1
    outright = [int((top[:, i] & ~tied).sum()) for i in range(3)]
    patterns = Counter(tuple(np.flatnonzero(r)) for r in top[tied])
    ordered = np.sort(rows, axis=1)
    return rows, outright, int(tied.sum()), patterns, ordered[:, -1] - ordered[:, -2]


rows3, outright3, ties3, patterns3, margins3 = three_models(range(200))
for name, got, printed in (("C = 0.01", outright3[0], 89), ("C = 1", outright3[1], 42),
                           ("C = 100", outright3[2], 38)):
    same(f"3 {name} wins outright on {printed} splits", got, printed, tolerance=0)
same("3 31 splits are ties", ties3, 31, tolerance=0)
same("3 29 of them between C = 1 and C = 100", patterns3[(1, 2)], 29, tolerance=0)
same("3 one between C = 0.01 and C = 1", patterns3[(0, 1)], 1, tolerance=0)
same("3 and one three-way", patterns3[(0, 1, 2)], 1, tolerance=0)
same("3 the median winning margin is 0.0015", float(np.median(margins3)), 0.0015,
     tolerance=5e-5)
same("3 its mean 0.0039", float(margins3.mean()), 0.0039, tolerance=5e-5)
same("3 its largest 0.0296", float(margins3.max()), 0.0296, tolerance=5e-5)
same("3 the mean AUCs: C = 0.01", float(rows3[:, 0].mean()), 0.9549, tolerance=5e-5)
same("3 the mean AUCs: C = 100", float(rows3[:, 2].mean()), 0.9544, tolerance=5e-5)
_, outright_other, _, _, _ = three_models(range(200, 400))
same("3 on seeds 200-399 C = 0.01 wins outright 86 times", outright_other[0], 86,
     tolerance=0)
same("3 and C = 1 and C = 100 41 each", outright_other[1] + outright_other[2], 82,
     tolerance=0)
same("3 two recipes: 3 candidates x 5 folds", 3 * 5, 15, tolerance=0)
same("3 two recipes on a validation set: one fit per candidate, refit is fit 4",
     3 * 1 + 1, 4, tolerance=0)
same("3 the validation split: 800 - 200 test - 150 validation leaves 450",
     800 - 200 - 150, 450, tolerance=0)

# Section 3's worked split: the first seed without a tie is seed 1.
first_untied = int(np.argmax(~(np.sort(rows3, axis=1)[:, -1]
                               == np.sort(rows3, axis=1)[:, -2])))
same("3 the worked split is seed 1", first_untied, 1, tolerance=0)
for column, printed in ((0, 0.9756), (2, 0.9667), (1, 0.9660)):
    same(f"3 seed 1, model {column}: AUC {printed}", rows3[1, column], printed,
         tolerance=5e-5)
same("3 seed 1's margin: 0.9756 - 0.9667 = 0.0089", margins3[1], 0.0089,
     tolerance=5e-5)
same("3 the printed margin from the printed AUCs", 0.9756 - 0.9667, 0.0089,
     tolerance=1e-9)

# ------------------------- Section 4.1 and 4.2: the five round models
from sklearn.model_selection import KFold, cross_val_score            # noqa: E402

same("4.1 five folds of 160 drives", len(fleet_y) / 5, 160, tolerance=0)
same("4.1 so each round trains on 640", len(fleet_y) * 4 / 5, 640, tolerance=0)

rounds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
round_models = [make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000))
                .fit(fleet_X.iloc[a], fleet_y.iloc[a]) for a, _ in rounds.split(fleet_X, fleet_y)]
power = [float(m[-1].coef_[0][list(fleet_X.columns).index("power_on_hours")])
         for m in round_models]
seek = [float(m[-1].coef_[0][list(fleet_X.columns).index("seek_error_rate")])
        for m in round_models]
same("4.2 power-on hours weighs 0.010 at the least", min(power), 0.010, tolerance=5e-4)
same("4.2 and 0.353 at the most", max(power), 0.353, tolerance=5e-4)
same("4.2 seek errors change sign across the rounds",
     float(min(seek) < 0 < max(seek)), 1.0, tolerance=0)
flags = np.array([m.predict_proba(fleet_X)[:, 1] >= 0.5 for m in round_models])
for r, printed in enumerate((15, 17, 13, 14, 13)):
    same(f"4.2 round {r + 1}'s model flags {printed} drives at 0.5",
         int(flags[r].sum()), printed, tolerance=0)
same("4.2 the five disagree on 8 drives",
     int((flags.any(axis=0) & ~flags.all(axis=0)).sum()), 8, tolerance=0)

# ------------------------- Section 4.3: the naive interval against new fleets
seed0 = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)),
                        fleet_X, fleet_y, scoring="roc_auc", cv=rounds)
same("4.3 s = 0.0187 at seed 0", float(seed0.std()), 0.0187, tolerance=5e-5)
same("4.3 so s / sqrt(5) = 0.0084", float(seed0.std() / np.sqrt(5)), 0.0084,
     tolerance=5e-5)
same("4.3 and the naive interval is +/- 0.016",
     float(1.96 * seed0.std() / np.sqrt(5)), 0.016, tolerance=5e-4)

# 4.3's Checks paragraph: the same five scores with the sample spread (ddof=1).
same("4.3 Checks: std(ddof=1) gives 0.0209", float(seed0.std(ddof=1)), 0.0209,
     tolerance=5e-5)
same("4.3 Checks: and 0.0209 / sqrt(5) = 0.0094",
     float(seed0.std(ddof=1) / np.sqrt(5)), 0.0094, tolerance=5e-5)

# 4.7: the mean of five fold AUCs against one AUC over the pooled out-of-fold
# predictions - a different calculation, recomputed by a different call.
from sklearn.model_selection import cross_val_predict                 # noqa: E402
pooled = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)),
                           fleet_X, fleet_y, cv=rounds, method="predict_proba")[:, 1]
same("4.7 pooled out-of-fold AUC at seed 0 is 0.9519",
     roc_auc_score(fleet_y, pooled), 0.9519, tolerance=5e-5)
same("4.7 against the mean of the five fold AUCs, 0.9514",
     float(seed0.mean()), 0.9514, tolerance=5e-5)

fleet_scores = []
for fleet_seed in range(1000, 1300):
    Xn, yn = load_fleet(800, random_state=fleet_seed)
    fleet_scores.append(cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)),
        Xn, yn, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)))
fleet_scores = np.array(fleet_scores)
fleet_means = fleet_scores.mean(axis=1)
fleet_se = fleet_scores.std(axis=1, ddof=1) / np.sqrt(5)
centre = fleet_means.mean()
same("4.3 across 300 new fleets the mean moves by 0.0165",
     float(fleet_means.std(ddof=1)), 0.0165, tolerance=5e-5)
same("4.3 that is +/- 0.032 at 95%", float(1.96 * fleet_means.std(ddof=1)), 0.032,
     tolerance=5e-4)
same("4.3 Checks: with ddof=1 the real spread is about 1.8 times the naive one",
     float(fleet_means.std(ddof=1) / (seed0.std(ddof=1) / np.sqrt(5))), 1.8,
     tolerance=0.05)
same("4.3 the round scores correlate by only 0.023",
     float(np.corrcoef(fleet_scores.T)[~np.eye(5, dtype=bool)].mean()), 0.023,
     tolerance=5e-4)
same("4.3 the naive interval covers the centre in 80% of fleets",
     float((np.abs(fleet_means - centre) <= 1.96 * fleet_se).mean()), 0.80,
     tolerance=5e-3)
same("4.3 and with t = 2.776 in 88%",
     float((np.abs(fleet_means - centre) <= 2.776 * fleet_se).mean()), 0.88,
     tolerance=5e-3)

# ------------------------- Section 4.4: one failure, a perfect fold
plain_auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)),
                            fleet_X, fleet_y, scoring="roc_auc",
                            cv=KFold(n_splits=5, shuffle=True, random_state=3))
plain_counts = [int(fleet_y.iloc[v].sum()) for _, v in
                KFold(n_splits=5, shuffle=True, random_state=3).split(fleet_X)]
same("4.4 KFold's fold 2 holds one failure", plain_counts[1], 1, tolerance=0)
same("4.4 and scores a perfect 1.000", float(plain_auc[1]), 1.0, tolerance=1e-12)
same("4.4 both columns hold 29 failures", sum(plain_counts), 29, tolerance=0)
same("4.4 a fold should hold 29 / 5 = 5.8", 29 / 5, 5.8, tolerance=1e-9)
same("4.4 a stratified fold has about 6 x 154 pairs", 6 * 154, 924, tolerance=0)

# ------------------------- Section 4.5: more folds against repeated folds
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold  # noqa: E402
import warnings                                                        # noqa: E402


def procedure_spread(make_cv):
    estimates = np.array([cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)),
        fleet_X, fleet_y, scoring="roc_auc", cv=make_cv(s)).mean() for s in range(40)])
    return float(estimates.mean()), float(estimates.std(ddof=1))


for label, make_cv, mean_printed, sd_printed in (
        ("5-fold", lambda s: StratifiedKFold(5, shuffle=True, random_state=s), 0.9532, 0.0033),
        ("10-fold", lambda s: StratifiedKFold(10, shuffle=True, random_state=s), 0.9539, 0.0036),
        ("5-fold x 2", lambda s: RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                                         random_state=s), 0.9530, 0.0025),
        ("5-fold x 10", lambda s: RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                                          random_state=s), 0.9533, 0.0009)):
    got_mean, got_sd = procedure_spread(make_cv)
    same(f"4.5 {label}: mean of the 40 estimates", got_mean, mean_printed, tolerance=5e-5)
    same(f"4.5 {label}: standard deviation across seeds", got_sd, sd_printed,
         tolerance=5e-5)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loo = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)),
                          fleet_X, fleet_y, scoring="roc_auc", cv=LeaveOneOut())
same("4.5 leave-one-out with AUC is nan in every round", float(np.isnan(loo).all()), 1.0,
     tolerance=0)

# ------------------------- Section 4.6: ten readings per drive
from sklearn.ensemble import RandomForestClassifier                    # noqa: E402
from sklearn.model_selection import GroupKFold                         # noqa: E402
from sklearn.neighbors import KNeighborsClassifier                     # noqa: E402

readings = 10
noise_rng = np.random.default_rng(0)
column_sd = fleet_X.std().values
X_read = pd.DataFrame(np.repeat(fleet_X.values, readings, axis=0)
                      + noise_rng.normal(0, 0.05, (len(fleet_X) * readings,
                                                   fleet_X.shape[1])) * column_sd,
                      columns=fleet_X.columns)
y_read = pd.Series(np.repeat(fleet_y.values, readings))
drive_id = np.repeat(np.arange(len(fleet_X)), readings)
for label, model, printed in (
        ("logistic regression", make_pipeline(StandardScaler(),
                                              LogisticRegression(max_iter=2_000)),
         (0.960, 0.950, 0.951)),
        ("1-nearest neighbour", make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
         (1.000, 0.620, 0.622)),
        ("random forest", RandomForestClassifier(300, random_state=0, n_jobs=-1),
         (0.999, 0.896, 0.890))):
    shuffled = cross_val_score(model, X_read, y_read, scoring="roc_auc",
                               cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    grouped = cross_val_score(model, X_read, y_read, scoring="roc_auc",
                              cv=GroupKFold(5), groups=drive_id).mean()
    single_row = cross_val_score(model, fleet_X, fleet_y, scoring="roc_auc",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    same(f"4.6 {label}, rows split at random", float(shuffled), printed[0], tolerance=5e-4)
    same(f"4.6 {label}, each drive kept together", float(grouped), printed[1],
         tolerance=5e-4)
    same(f"4.6 {label}, one row per drive", float(single_row), printed[2], tolerance=5e-4)

# GroupKFold with groups of different sizes: one drive of ten readings among 799
# drives of one, never split.
sizes = np.r_[np.zeros(10, int), np.arange(1, 800)]
for _, held in GroupKFold(5).split(np.zeros((len(sizes), 1)), groups=sizes):
    inside = int((sizes[held] == 0).sum())
    same("4.6 the ten readings of one drive stay in one fold",
         float(inside in (0, 10)), 1.0, tolerance=0)

# ------------------------- Section 4.7: what the right-hand column costs
same("4.7 forty seeds of five rounds are 200 fits", 40 * 5, 200, tolerance=0)

# ------------------------- Sections 5.3, 5.4 and 6: the second hour's arithmetic
same("5.3 printed parts of degree 2 add to 614.4", 59.6 + 70.8 + 484.0, 614.4,
     tolerance=1e-9)
same("5.3 degree 2's unrounded total is 614.3", sum(universes(2, True, 0)) + 484.0,
     614.3, tolerance=0.05)
same("5.1 about 4.7 of 25 uniform temperatures fall above 30 degrees",
     25 * (38 - 30) / (38 - (-5)), 4.7, tolerance=0.05)
same("5.4 degree 12 improves 4,239-fold from 25 to 60", 32_328_366.2 / 7_626.9, 4239,
     tolerance=0.5)
same("5.4 and 35,960-fold by 150", 32_328_366.2 / 899.0, 35960, tolerance=0.5)
deg1_150 = sum(universes(1, True, 0, n_train=150)) + 484.0
deg12_150 = sum(universes(12, True, 0, n_train=150)) + 484.0
same("5.4 at m = 150 the straight line's total is 5,861.5", deg1_150, 5861.5,
     tolerance=0.1)
same("5.4 so at 150 degree 12 is no longer the worst",
     float(deg12_150 < deg1_150), 1.0, tolerance=0)

# 6: one feature alone. The validation AUC cannot move, because the fitted
# logistic regression ranks drives in the feature's own order.
one_counts, _, one_valid = learning_curve(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
    fleet_X[["power_on_hours"]], fleet_y, train_sizes=np.linspace(0.15, 1.0, 8),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0), scoring="roc_auc")
same("6 the first learning-curve size is 96 drives", int(one_counts[0]), 96, tolerance=0)
same("6 and the last 640", int(one_counts[-1]), 640, tolerance=0)
same("6 one feature: the validation curve is exactly flat",
     float(np.ptp(one_valid, axis=0).max()), 0.0, tolerance=1e-12)
pooled = []
for _, valid_rows in StratifiedKFold(n_splits=5, shuffle=True,
                                     random_state=0).split(fleet_X, fleet_y):
    pooled.append(roc_auc_score(fleet_y.iloc[valid_rows],
                                fleet_X["power_on_hours"].iloc[valid_rows]))
same("6 at power_on_hours' own AUC, 0.718", float(np.mean(pooled)), 0.718,
     tolerance=5e-4)

# ------------------------- Section 5.2, the decomposition at one temperature
#
# The same 300 training sets as notebook 2 (same generator, same seed), but
# fitted with numpy's polyfit instead of the scikit-learn pipeline, and the
# truth written out as a formula rather than taken from true_curve.
def at_thirty(degree, seed=0, n_train=25, n_samples=300):
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(n_samples):
        temperature, energy = sample_energy(n_train, rng)
        predictions.append(np.polyval(np.polyfit(temperature, energy, degree), 30.0))
    return np.array(predictions)


truth_30 = 240 + 0.55 * (30 - 18) ** 2 + 12 * np.sin(30 / 4)
same("5.2 the truth at 30 degrees is 330.5", truth_30, 330.5, tolerance=0.05)
for degree, printed in ((1, (303.6, 723.9, 856.1, 2063.9, 215.6, 364.7)),
                        (2, (323.1, 53.7, 58.4, 596.1, 299.1, 346.5))):
    p30 = at_thirty(degree)
    average, low, high = p30.mean(), p30.min(), p30.max()
    bias_squared = (average - truth_30) ** 2
    variance = np.var(p30)
    total = bias_squared + variance + NOISE_SD ** 2
    for label, value, shown in (("average", average, printed[0]),
                                ("bias squared", bias_squared, printed[1]),
                                ("variance", variance, printed[2]),
                                ("expected error", total, printed[3]),
                                ("lowest prediction", low, printed[4]),
                                ("highest prediction", high, printed[5])):
        same(f"5.2 degree {degree} at 30 degrees: {label}", value, shown,
             tolerance=0.05)
    # Without the formula: new days at 30 degrees, on a noise stream of our own.
    new_days = truth_30 + np.random.default_rng(7).normal(0, NOISE_SD,
                                                          (len(p30), 2000))
    simulated = float(np.mean((new_days - p30[:, None]) ** 2))
    same(f"5.2 degree {degree}: simulated new days agree with the formula",
         simulated, total, tolerance=0.01 * total)
same("5.2 printed degree-1 parts add to 2,064.0", 723.9 + 856.1 + 484.0, 2064.0,
     tolerance=1e-9)
same("5.2 and degree 2's to 596.1", 53.7 + 58.4 + 484.0, 596.1, tolerance=1e-9)
same("5.2 the bias itself is a signed gap, -26.9 kWh", 303.6 - 330.5, -26.9,
     tolerance=1e-9)
same("5.2 squaring the printed gap gives 723.6", round((303.6 - 330.5) ** 2, 1),
     723.6, tolerance=1e-9)

# The figure's claim: degree 1's bias squared touches zero near 5 and 28 degrees.
profile = []
rng = np.random.default_rng(0)
for _ in range(300):
    temperature, energy = sample_energy(25, rng)
    profile.append(np.polyval(np.polyfit(temperature, energy, 1), TEST_TEMPERATURE))
profile_bias = (np.mean(profile, axis=0) - TEST_TRUTH) ** 2
cold, warm = TEST_TEMPERATURE < 15, TEST_TEMPERATURE >= 15
same("5.2 degree 1's bias squared is lowest near 5 degrees",
     TEST_TEMPERATURE[cold][profile_bias[cold].argmin()], 5, tolerance=1)
same("5.2 and near 28 degrees", TEST_TEMPERATURE[warm][profile_bias[warm].argmin()],
     28, tolerance=1)
same("5.2 its average over 220 temperatures is 5,250.8", profile_bias.mean(),
     5250.8, tolerance=0.1)

# 5.1: the ends of the plot are partly extrapolation. The formula for the
# expected minimum of n uniforms, checked against draws of our own.
draws = np.random.default_rng(3).uniform(-5, 38, (200_000, 25))
same("5.1 the coldest of 25 days is on average -3.3", -5 + 43 / 26, -3.3,
     tolerance=0.05)
same("5.1 and the simulation agrees", draws.min(axis=1).mean(), -5 + 43 / 26,
     tolerance=0.01)
same("5.1 the warmest is on average 36.3", 38 - 43 / 26, 36.3, tolerance=0.05)
same("5.1 and the simulation agrees, warm end", draws.max(axis=1).mean(),
     38 - 43 / 26, tolerance=0.01)
same("5.1 the coldest day is above -4 in 0.55 of the sets", (42 / 43) ** 25, 0.55,
     tolerance=0.01)
same("5.1 and the simulation agrees, 0.55", float(np.mean(draws.min(axis=1) > -4)),
     (42 / 43) ** 25, tolerance=0.005)


# ------------------ Section 6, training against validation, without the truth
# The handout's table: one sample of 25 days (and one of 150), 5-fold
# cross-validation. Rebuilt with the folds walked by hand and each polynomial
# fitted by numpy.polyfit rather than scikit-learn's pipeline.
from sklearn.model_selection import KFold                              # noqa: E402

def by_hand(n_days, degree):
    temperature, energy = sample_energy(n_days, np.random.default_rng(0))
    train_errors, valid_errors = [], []
    for fit_rows, held_rows in KFold(n_splits=5, shuffle=True,
                                     random_state=0).split(temperature):
        coefficients = np.polyfit(temperature[fit_rows], energy[fit_rows], degree)
        train_errors.append(np.mean((np.polyval(coefficients, temperature[fit_rows])
                                     - energy[fit_rows]) ** 2))
        valid_errors.append(np.mean((np.polyval(coefficients, temperature[held_rows])
                                     - energy[held_rows]) ** 2))
    return sum(train_errors) / 5, sum(valid_errors) / 5

for n_days, degree, train_printed, valid_printed in (
        (25, 1, 5155.7, 6064.3), (25, 2, 312.8, 450.4), (25, 3, 306.1, 525.6),
        (25, 5, 262.4, 615.1), (150, 1, 7281.1, 8058.6), (150, 2, 593.5, 627.0),
        (150, 3, 584.2, 628.4), (150, 5, 537.8, 596.0)):
    train, valid = by_hand(n_days, degree)
    same(f"6 {n_days} days, degree {degree}, training {train_printed}", train,
         train_printed, tolerance=0.06)
    same(f"6 {n_days} days, degree {degree}, validation {valid_printed}", valid,
         valid_printed, tolerance=0.06)

# Slide 25's 25 days are the first of section 5's 300 training sets. The
# straight line fitted to them must give the first of the 300 predictions at
# 30 degrees that section 5.2 prints, 300.8.
days_25, energy_25 = sample_energy(25, np.random.default_rng(0))
same("5.5 slide 25's sample is the first of the 300: its line gives 300.8 at 30",
     np.polyval(np.polyfit(days_25, energy_25, 1), 30.0), 300.8, tolerance=0.05)
same("5.5 and so does the first of at_thirty's 300", at_thirty(1)[0],
     np.polyval(np.polyfit(days_25, energy_25, 1), 30.0), tolerance=1e-9)

train5, valid5 = by_hand(25, 5)
same("6 degree 5's gap on 25 days is 352.6", valid5 - train5, 352.6, tolerance=0.06)
assert train5 < NOISE_SD ** 2, "degree 5's training error must sit below 484"
train5, valid5 = by_hand(150, 5)
same("6 and on 150 days 58.3", valid5 - train5, 58.3, tolerance=0.06)


# ------------------ Sections 5.3 and 5.5, the counts, the floor, the roots
# 5.3: the same 300 training sets serve every degree. Read from notebook 2
# itself: decompose() calls fit_many with seed 0 and the table loop passes no
# other seed, so every degree draws the same samples.
import json                                                            # noqa: E402
notebook_2 = json.loads((Path(__file__).resolve().parent.parent / "Notebooks"
                         / "02_bias_variance_and_learning_curves.ipynb").read_text())
source_2 = "".join("".join(cell["source"]) for cell in notebook_2["cells"]
                   if cell["cell_type"] == "code")
assert "def fit_many(degree, n_train=25, n_samples=300, seed=0)" in source_2
assert "def decompose(degree, n_train=25, n_samples=300, seed=0)" in source_2
assert "bias_squared, variance, noise, expected_mse = decompose(degree)" in source_2
checks += 1
same("5.3 1,800 models", 6 * 300, 1800, tolerance=0)
same("5.3 396,000 predictions", 6 * 300 * 220, 396000, tolerance=0)

# 5.3: even the true curve can score below half the floor on a small test set.
# Simulated here, not taken from the chi-squared table the handout quotes.
noise_draws = np.random.default_rng(11)
for n_days, printed, tolerance in ((5, 0.22, 0.01), (50, 0.0012, 0.0004)):
    errors = noise_draws.normal(0, NOISE_SD, (400_000, n_days)) ** 2
    share = float(np.mean(errors.mean(axis=1) < NOISE_SD ** 2 / 2))
    same(f"5.3 true curve below half the floor on {n_days} days", share, printed,
         tolerance=tolerance)

# 5.5: the roots, from the cross-validated errors rebuilt above.
for n_degree, train_rmse, valid_rmse in ((1, 72, 78), (2, 18, 21)):
    train, valid = by_hand(25, n_degree)
    same(f"5.5 degree {n_degree} trains at {train_rmse} kWh", train ** 0.5,
         train_rmse, tolerance=0.5)
    same(f"5.5 degree {n_degree} validates at {valid_rmse} kWh", valid ** 0.5,
         valid_rmse, tolerance=0.5)


# 7.3-7.4: the fits a search and a nested search make, counted by running them
# on a model that only counts its own fits, not by multiplying.
from sklearn.base import BaseEstimator, ClassifierMixin                # noqa: E402
from sklearn.model_selection import (GridSearchCV, ParameterGrid,      # noqa: E402
                                     cross_val_score)


class FitCounter(BaseEstimator, ClassifierMixin):
    fits, sizes, scored = 0, [], []

    def __init__(self, k=1, C=1.0):
        self.k, self.C = k, C

    def fit(self, X, y):
        FitCounter.fits += 1
        FitCounter.sizes.append(len(X))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        FitCounter.scored.append(len(X))
        p = np.random.default_rng(len(X)).random(len(X))
        return np.column_stack([1 - p, p])


count_grid = {"k": [2, 5, 10, 20, 40], "C": [0.01, 0.1, 1, 10, 100]}
count_y = fleet_y.to_numpy() if hasattr(fleet_y, "to_numpy") else np.asarray(fleet_y)
count_X = np.zeros((len(count_y), 1))
same("7.3 the grid has 25 combinations", len(ParameterGrid(count_grid)), 25,
     tolerance=0)
FitCounter.fits, FitCounter.sizes = 0, []
GridSearchCV(FitCounter(), count_grid, refit=False, scoring="roc_auc",
             cv=StratifiedKFold(5, shuffle=True, random_state=1)).fit(count_X, count_y)
same("7.3 the search makes 125 fits", FitCounter.fits, 125, tolerance=0)
FitCounter.fits, FitCounter.sizes, FitCounter.scored = 0, [], []
cross_val_score(GridSearchCV(FitCounter(), count_grid, scoring="roc_auc",
                             cv=StratifiedKFold(5, shuffle=True, random_state=1)),
                count_X, count_y, scoring="roc_auc",
                cv=StratifiedKFold(5, shuffle=True, random_state=2))
same("7.4 nested cross-validation makes 630 fits", FitCounter.fits, 630, tolerance=0)
same("7.4 inner models train on 512 drives", min(FitCounter.sizes), 512, tolerance=0)
same("7.4 each winner is refitted on 640", max(FitCounter.sizes), 640, tolerance=0)
same("7.4 inner folds score 128 drives", min(FitCounter.scored), 128, tolerance=0)
same("7.4 outer folds score 160 drives", max(FitCounter.scored), 160, tolerance=0)

# 7.4: the nested figure as the mean of its five outer scores, as printed.
outer_scores = (0.698, 0.852, 0.754, 0.487, 0.558)
same("7.4 the five outer scores sum to 3.349", sum(outer_scores), 3.349, tolerance=1e-9)
same("7.4 and their mean is 0.670", sum(outer_scores) / 5, 0.670, tolerance=5e-4)

# 7.2: a fresh test set is centred on the truth. Rebuilt from notebook 3's
# recipe (the noise from default_rng(0)), then scored on 40 new fleets with new
# noise; nothing is taken from the notebook's outputs.
from sklearn.feature_selection import SelectKBest, f_classif           # noqa: E402
from sklearn.metrics import roc_auc_score as fresh_auc                 # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402

columns = [f"n{i}" for i in range(2_000)]
noise_800 = pd.DataFrame(np.random.default_rng(0).normal(size=(800, 2_000)),
                         columns=columns)
honest_k10 = Pipeline([("select", SelectKBest(f_classif, k=10)),
                       ("scale", StandardScaler()),
                       ("model", LogisticRegression(max_iter=2_000))])
search_winner = GridSearchCV(
    honest_k10, {"select__k": [2, 5, 10, 20, 40], "model__C": [0.01, 0.1, 1, 10, 100]},
    cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc"
).fit(noise_800, fleet_y)
honest_k10.fit(noise_800, fleet_y)
on_fresh = {"honest": [], "winner": []}
for fresh_seed in range(1, 41):
    _, y_fresh = load_fleet(800, random_state=1000 + fresh_seed)
    X_fresh = pd.DataFrame(np.random.default_rng(10_000 + fresh_seed).normal(
        size=(800, 2_000)), columns=columns)
    on_fresh["honest"].append(fresh_auc(y_fresh, honest_k10.predict_proba(X_fresh)[:, 1]))
    on_fresh["winner"].append(fresh_auc(y_fresh, search_winner.predict_proba(X_fresh)[:, 1]))
same("7.2 fresh test sets: the honest pipeline averages 0.509",
     float(np.mean(on_fresh["honest"])), 0.509, tolerance=5e-4)
same("7.2 lowest fresh test set 0.40", min(on_fresh["honest"]), 0.40, tolerance=5e-3)
same("7.2 highest fresh test set 0.68", max(on_fresh["honest"]), 0.68, tolerance=5e-3)
same("7.2 the search's winner averages 0.512",
     float(np.mean(on_fresh["winner"])), 0.512, tolerance=5e-4)

# ------------------- 4 November study pass: numbers added to slides and handout

# 5.1 and slide 25: the straight line's cross-validated error, back in kWh.
same("5.1 sqrt(6,064.3) is about 78 kWh", float(np.sqrt(6064.3)), 78, tolerance=0.5)
# 5.2 and slide 30: the expected error at 30 °C, back in kWh.
same("5.2 sqrt(2,063.9) is about 45 kWh", float(np.sqrt(2063.9)), 45, tolerance=0.5)
# Slide 37: the gap from the printed errors.
same("5.5 615.1 - 262.4 is about 353", 615.1 - 262.4, 353, tolerance=0.5)

# 7.1 and slide 42: why pure noise offers ten good-looking columns. Rebuilt from
# notebook 3's recipe (default_rng(0), 800 x 2,000), not from its outputs.
fails = noise_800.values[np.asarray(fleet_y) == 1]
healthy = noise_800.values[np.asarray(fleet_y) == 0]
same("7.1 29 failing drives", len(fails), 29, tolerance=0)
same("7.1 a mean of 29 wobbles by 1/sqrt(29) = 0.19", 1 / np.sqrt(29), 0.19,
     tolerance=5e-3)
same("7.2 and of 290 by 1/sqrt(290) = 0.06", 1 / np.sqrt(290), 0.06, tolerance=5e-3)
mean_gap = fails.mean(axis=0) - healthy.mean(axis=0)
gap_se = np.sqrt(1 / len(fails) + 1 / len(healthy))
kept = SelectKBest(f_classif, k=10).fit(noise_800, fleet_y).get_support()
kept_gap = np.abs(mean_gap[kept])
same("7.1 the ten kept columns: smallest shift 0.52", kept_gap.min(), 0.52, tolerance=5e-3)
same("7.1 largest shift 0.77", kept_gap.max(), 0.77, tolerance=5e-3)
same("7.1 in standard errors: from 2.8", kept_gap.min() / gap_se, 2.8, tolerance=0.05)
same("7.1 to 4.1", kept_gap.max() / gap_se, 4.1, tolerance=0.05)
same("7.1 only 6 of 2,000 columns beyond 3 standard errors",
     int((np.abs(mean_gap) / gap_se > 3).sum()), 6, tolerance=0)
same("7.1 0.931 is the mean of its five folds",
     (0.928 + 0.979 + 0.975 + 0.943 + 0.832) / 5, 0.931, tolerance=5e-4)

# 7.3 and slide 47: best_score_ as the mean of the winner's five round AUCs, from
# the search rebuilt above.
results = search_winner.cv_results_
same("7.3 the winner is k = 20", search_winner.best_params_["select__k"], 20, tolerance=0)
same("7.3 and C = 10", search_winner.best_params_["model__C"], 10, tolerance=0)
winner_rounds = [results[f"split{j}_test_score"][search_winner.best_index_]
                 for j in range(5)]
for j, printed in enumerate((0.750, 0.754, 0.879, 0.740, 0.877)):
    same(f"7.3 the winner's round {j + 1}: {printed}", winner_rounds[j], printed,
         tolerance=5e-4)
same("7.3 the five printed rounds sum to 4.000",
     0.750 + 0.754 + 0.879 + 0.740 + 0.877, 4.000, tolerance=1e-9)
order = np.argsort(-results["mean_test_score"])
for rank, (k_cols, c_value, printed) in enumerate(((20, 1, 0.798), (20, 100, 0.798))):
    params = results["params"][order[rank + 1]]
    same(f"7.3 runner-up {rank + 1}: k = {k_cols}", params["select__k"], k_cols, tolerance=0)
    same(f"7.3 runner-up {rank + 1}: C = {c_value}", params["model__C"], c_value, tolerance=0)
    same(f"7.3 runner-up {rank + 1} scores {printed}",
         results["mean_test_score"][order[rank + 1]], printed, tolerance=5e-4)
last = results["params"][order[-1]]
same("7.3 the last is k = 2", last["select__k"], 2, tolerance=0)
same("7.3 with C = 0.01", last["model__C"], 0.01, tolerance=0)
for j, printed in enumerate((0.717, 0.641, 0.712, 0.610, 0.739)):
    same(f"7.3 the last's round {j + 1}: {printed}",
         results[f"split{j}_test_score"][order[-1]], printed, tolerance=5e-4)
same("7.3 the luck of the winner, best - average, 0.0734 as printed",
     0.7999 - 0.7265, 0.0734, tolerance=1e-9)
same("7.3 and unrounded about 0.073",
     search_winner.best_score_ - results["mean_test_score"].mean(), 0.073,
     tolerance=1e-3)
same("7.3 on the slide, 0.800 - 0.726 = 0.074", 0.800 - 0.726, 0.074, tolerance=1e-9)
same("7.3 fits: 5 values of k x 5 of C x 5 folds, then the refit",
     5 * 5 * 5 + 1, 126, tolerance=0)
same("7.4 what nested cross-validation leaves above the truth",
     0.670 - 0.500, 0.170, tolerance=1e-9)
same("7.4 the optimism of the search", 0.7999 - 0.6699, 0.1300, tolerance=1e-9)

# 8 and slide 52: what paying the debt costs.
same("8 the honest threshold costs 1,400 EUR more", 4_700 - 3_300, 1_400, tolerance=0)
same("8 42% more than the cheating figure", 1_400 / 3_300, 0.42, tolerance=5e-3)

print(f"lesson 5: {checks} hand-worked numbers recomputed, all agree")

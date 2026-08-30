"""Recompute every number lesson 2 works out by hand.

    python Lessons/02_data_exploration_and_preparation/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. The two derivations here — Tukey's constant
and the attenuation of a correlation under mean imputation — are exactly the
kind that read as authoritative and are never checked, so they are checked from
first principles below: the quartiles come from the normal distribution rather
than from the handout's rounded 0.6745, and the attenuation is confirmed by
simulation as well as by the formula.
"""

import math

import numpy as np
from scipy.stats import norm

HANDOUT = ("Lessons/02_data_exploration_and_preparation/Docs/"
           "data_exploration_and_preparation.md")
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ------------------------------------- Section 3.2, mean imputation

# The handout's two illustrations: 8% missing and 40% missing.
for missing, printed in ((0.08, 0.96), (0.40, 0.77)):
    same(f"3.2 correlation retained with {missing:.0%} missing",
         math.sqrt(1 - missing), printed, tolerance=5e-3)

# The formula says the correlation is multiplied by sqrt(1 - p). Confirm it by
# simulation rather than by rearranging the same algebra.
rng = np.random.default_rng(0)
n, missing = 400_000, 0.40
x = rng.normal(size=n)
y = 0.7 * x + rng.normal(scale=math.sqrt(1 - 0.7 ** 2), size=n)
imputed = np.where(rng.random(n) < missing, x.mean(), x)

same("3.2 the variance after imputation is (1-p) times the original",
     imputed.var() / x.var(), 1 - missing, tolerance=0.01)
same("3.2 the correlation is attenuated by sqrt(1-p)",
     np.corrcoef(imputed, y)[0, 1] / np.corrcoef(x, y)[0, 1],
     math.sqrt(1 - missing), tolerance=0.01)

# ------------------------------------- Section 4.1, Tukey's constant

# From the normal distribution, not from the handout's rounded quartile.
q1, q3 = norm.ppf(0.25), norm.ppf(0.75)
same("4.1 the first quartile of a standard normal", q1, -0.6745, tolerance=5e-4)
same("4.1 the interquartile range in standard deviations",
     q3 - q1, 1.349, tolerance=1e-3)
same("4.1 where Tukey's upper fence falls",
     q3 + 1.5 * (q3 - q1), 2.698, tolerance=1e-3)

# The z-score rule the handout compares it against.
same("4.1 how often a normal exceeds three standard deviations",
     2 * (1 - norm.cdf(3)), 0.0027, tolerance=5e-5)

# The two rules disagree on a normal: Tukey flags at 2.698 sigma, z at 3.
# Both fractions below count BOTH tails. The handout used to quote Tukey's
# per-tail 0.35% against the z-score rule's two-tail 0.27% and conclude the
# two were comparable; they are not, and this is the check that says so.
tukey_share = 2 * (1 - norm.cdf(q3 + 1.5 * (q3 - q1)))
z_share = 2 * (1 - norm.cdf(3))
same("4.1 how often Tukey's fence flags a normal value", tukey_share, 0.0070,
     tolerance=5e-4)
same("4.1 Tukey flags 2.6x as much of a normal column as k=3 does",
     tukey_share / z_share, 2.6, tolerance=0.05)
same("4.1 which is about 14 points in 2000", 2000 * tukey_share, 14, tolerance=0.5)
same("4.1 against about 5", 2000 * z_share, 5, tolerance=0.5)

# ------------------- Sections 5.1, 6 and 7, recomputed from the generator
#
# These reach the handout's figures from the dataset itself, not from the
# notebook's printed output: the generator is the raw input, and everything
# below is derived from it here.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from churn_data import make_churn_data                       # noqa: E402
from sklearn.model_selection import train_test_split         # noqa: E402

frame = make_churn_data(n=2000, seed=7)
features = frame.drop(columns=["churned", "zip_code"])
X_train, X_test, _, _ = train_test_split(
    features, frame["churned"], test_size=0.25, random_state=7,
    stratify=frame["churned"])

# --- 2, what the correlation matrix says about the features -------------

# The three numbers Section 2 reads off the heatmap, recomputed from the frame.
# The claim that matters is the third: the features carry no redundancy, so the
# lesson's difficulty is not collinearity between columns.
_numeric = ["tenure_months", "monthly_charges", "age", "num_support_calls"]
_corr = frame[_numeric + ["churned"]].corr()
same("2 num_support_calls against churn", _corr.loc["num_support_calls", "churned"],
     0.16, tolerance=5e-3)
same("2 tenure_months against churn", _corr.loc["tenure_months", "churned"],
     -0.12, tolerance=5e-3)
_pairs = _corr.loc[_numeric, _numeric].where(
    ~np.eye(len(_numeric), dtype=bool)).abs().max().max()
same("2 no two features correlate above 0.035", _pairs, 0.035, tolerance=5e-4)

# --- 3.2, covariance carries units and correlation does not -------------

_cov = frame["tenure_months"].cov(frame["churned"])
same("3.2 covariance of tenure and churn, in months", _cov, -2.330, tolerance=5e-3)
same("3.2 the same relationship, tenure in years",
     (frame["tenure_months"] / 12).cov(frame["churned"]), -0.194, tolerance=5e-3)
same("3.2 the correlation is the same either way",
     (frame["tenure_months"] / 12).corr(frame["churned"]),
     frame["tenure_months"].corr(frame["churned"]), tolerance=1e-9)

# --- 5.1, what an outlier does to each scaler ---------------------------
charges = X_train["monthly_charges"].fillna(X_train["monthly_charges"].median())
ordinary = charges[charges < 500]
customer = 80.0

same("5.1 ordinary customers top out at 128.4", ordinary.max(), 128.4, tolerance=0.1)
same("5.1 the largest billing error is 3344.7", charges.max(), 3344.7, tolerance=0.1)
same("5.1 there are 16 billing errors", (charges >= 500).sum(), 16, tolerance=0)
same("5.1 and 1484 ordinary customers", len(ordinary), 1484, tolerance=0)

span, span_clean = charges.max() - charges.min(), ordinary.max() - ordinary.min()
same("5.1 min-max puts an 80.0 customer at 0.020",
     (customer - charges.min()) / span, 0.020, tolerance=1e-3)
same("5.1 and at 0.573 without the errors",
     (customer - ordinary.min()) / span_clean, 0.573, tolerance=1e-3)
same("5.1 the z-score of that customer is -0.007",
     (customer - charges.mean()) / charges.std(), -0.007, tolerance=2e-3)
same("5.1 and +0.964 without the errors",
     (customer - ordinary.mean()) / ordinary.std(), 0.964, tolerance=2e-3)

same("5.1 ordinary customers occupy 3.4% of the min-max range",
     (ordinary.max() - ordinary.min()) / span, 0.034, tolerance=1e-3)
same("5.1 min-max's denominator grows by 29.4x", span / span_clean, 29.4, tolerance=0.1)
same("5.1 the standard deviation grows by 10.8x",
     charges.std() / ordinary.std(), 10.8, tolerance=0.1)

# --- 4, the planted dirt the section describes -------------------------
tenure_all = frame["tenure_months"].dropna()
same("4 tenure_months reaches -3", tenure_all.min(), -3, tolerance=0)
same("4 and 999", tenure_all.max(), 999, tolerance=0)
ordinary_tenure = tenure_all[(tenure_all >= 0) & (tenure_all < 200)]
same("4 while ordinary customers stop at 72", ordinary_tenure.max(), 72, tolerance=0)

# Section 4 describes the WHOLE dataset; Section 5.1 the training split. The
# two counts differ (20 against 16) and the handout once printed 16 in both
# places, so both are checked here against their own population.
charges_all = frame["monthly_charges"].dropna()
errors_all = charges_all > 200
same("4 twenty of the 2000 rows carry a billing error", errors_all.sum(), 20, tolerance=0)
same("4 the ordinary maximum is 128.4", charges_all[~errors_all].max(), 128.4, tolerance=0.1)

# --- 4.2, which rule caught what -----------------------------------------
#
# Both rules are applied here from their definitions to the raw column, so
# nothing below reuses a threshold the handout printed.
mean_all, sd_all = charges_all.mean(), charges_all.std(ddof=0)
clean = charges_all[~errors_all]
sd_clean = clean.std(ddof=0)

same("4.2 the 17.23 is computed over 1980 ordinary customers", len(clean), 1980,
     tolerance=0)
same("4.2 the errors inflate s from 17.23", sd_clean, 17.23, tolerance=0.01)
same("4.2 to 171.25", sd_all, 171.25, tolerance=0.01)
same("4.2 a factor of 9.9", sd_all / sd_clean, 9.9, tolerance=0.05)
same("4.2 dragging the fence from 115.0", clean.mean() + 3 * sd_clean, 115.0, tolerance=0.1)
same("4.2 out to 593.0", mean_all + 3 * sd_all, 593.0, tolerance=0.1)

z_flag = ((charges_all - mean_all) / sd_all).abs() > 3
qa1, qa3 = charges_all.quantile(0.25), charges_all.quantile(0.75)
iqr_a = qa3 - qa1
lo, hi = qa1 - 1.5 * iqr_a, qa3 + 1.5 * iqr_a
tukey_flag = (charges_all < lo) | (charges_all > hi)

same("4.2 the z-score rule flags 20", z_flag.sum(), 20, tolerance=0)
same("4.2 all of them genuine errors", (z_flag & errors_all).sum(), 20, tolerance=0)
same("4.2 and no ordinary customer", (z_flag & ~errors_all).sum(), 0, tolerance=0)
same("4.2 Tukey flags 32", tukey_flag.sum(), 32, tolerance=0)
same("4.2 the same 20 errors", (tukey_flag & errors_all).sum(), 20, tolerance=0)
same("4.2 plus 12 ordinary customers", (tukey_flag & ~errors_all).sum(), 12, tolerance=0)

# The z-score rule survived its own ruined fence only because the errors are
# an order of magnitude beyond it. That is the claim; this is the number.
same("4.2 the smallest billing error is 780.1", charges_all[errors_all].min(), 780.1,
     tolerance=0.1)
assert charges_all[errors_all].min() > mean_all + 3 * sd_all, \
    "4.2 claims every error clears the inflated fence, and one does not"

same("4.2 Tukey's lower fence sits at 17.3", lo, 17.3, tolerance=0.1)
same("4.2 and its upper fence at 110.0", hi, 110.0, tolerance=0.1)
false_positives = charges_all[tukey_flag & ~errors_all].sort_values()
same("4.2 eight of the twelve pay 15.0 to 16.7", (false_positives <= 100).sum(), 8,
     tolerance=0)
same("4.2 the cheapest pays 15.0", false_positives.min(), 15.0, tolerance=0.05)
same("4.2 the dearest of those eight pays 16.7",
     false_positives[false_positives <= 100].max(), 16.7, tolerance=0.05)
same("4.2 and four pay 111.6 to 128.4", (false_positives > 100).sum(), 4, tolerance=0)
same("4.2 the cheapest of those four pays 111.6",
     false_positives[false_positives > 100].min(), 111.6, tolerance=0.05)
same("4.2 the dearest pays 128.4", false_positives.max(), 128.4, tolerance=0.05)

# --- 6, what the three encodings claim ----------------------------------
by_contract = frame.groupby("contract_type")["churned"].agg(["mean", "size"])
for level, rate, rows in (("month-to-month", 0.266, 1092),
                          ("one-year", 0.137, 488),
                          ("two-year", 0.071, 420)):
    same(f"6 churn rate for {level}", by_contract.loc[level, "mean"], rate, tolerance=1e-3)
    same(f"6 rows for {level}", by_contract.loc[level, "size"], rows, tolerance=0)

first = by_contract.loc["month-to-month", "mean"] - by_contract.loc["one-year", "mean"]
second = by_contract.loc["one-year", "mean"] - by_contract.loc["two-year", "mean"]
same("6 the first ordinal step is 0.129", first, 0.129, tolerance=1e-3)
same("6 the second is 0.066", second, 0.066, tolerance=1e-3)
same("6 so the first is about twice the second", first / second, 2.0, tolerance=0.1)

# --- 7, what binning tenure into three bands exposes ---------------------
import pandas as pd                                              # noqa: E402

bands = pd.cut(frame["tenure_months"].clip(0, 72), [-0.1, 6, 24, 72],
               labels=["0-6", "6-24", "24+"])
by_band = frame.groupby(bands, observed=True)["churned"].mean()
for band, printed in (("0-6", 0.399), ("6-24", 0.314), ("24+", 0.122)):
    same(f"7 churn rate in the {band} month band", by_band.loc[band], printed,
         tolerance=1e-3)

# --- 6.3, the width of one-hot encoding ---------------------------------
levels = frame["zip_code"].nunique()
same("6.3 zip_code has 493 levels", levels, 493, tolerance=0)
same("6.3 averaging 4.06 rows per level", len(frame) / levels, 4.06, tolerance=0.05)

# The two readings the caption gives the histogram. Counted from the column
# itself rather than from the figure: the point of naming a bar's height in the
# caption is that a student can check it, and so can this.
sizes = frame["zip_code"].value_counts()
same("6.3 the bar at 4 reaches 96", (sizes == 4).sum(), 96, tolerance=0)

# What the section says about the training half, recomputed from the same
# split the notebooks use rather than from anything they printed.
zip_train = frame.loc[X_train.index, "zip_code"]
zip_test = frame.loc[X_test.index, "zip_code"]
train_sizes = zip_train.value_counts()
same("6.3 477 codes appear in the training half", zip_train.nunique(), 477,
     tolerance=0)
same("6.3 3.1 training rows per coefficient", len(zip_train) / zip_train.nunique(),
     3.1, tolerance=0.05)
same("6.3 82 of them rest on a single row", (train_sizes == 1).sum(), 82,
     tolerance=0)
unseen = set(zip_test) - set(zip_train)
same("6.3 16 codes reach the test half unseen", len(unseen), 16, tolerance=0)
same("6.3 carrying 26 test rows", zip_test.isin(unseen).sum(), 26, tolerance=0)

# The geometry claim: one-hot rows are unit vectors, so every pair of
# distinct levels is the same distance apart whatever the levels are.
eye = np.eye(4)
gaps = {round(float(np.linalg.norm(eye[i] - eye[j])), 12)
        for i in range(4) for j in range(4) if i != j}
same("6.3 every distinct pair sits sqrt(2) apart", gaps.pop(), 2 ** 0.5,
     tolerance=1e-9)

# --- 9.2, the five-customer worked example ------------------------------
#
# Recomputed from the counts, not from the handout's fractions.
n_c, churners = 5, 2
leaky = churners / n_c
same("9.2 the leaky value everyone in the category gets", leaky, 0.40, tolerance=1e-9)
same("9.2 an honest churner should have had 0.25",
     (churners - 1) / (n_c - 1), 0.25, tolerance=1e-9)
same("9.2 an honest non-churner should have had 0.50",
     churners / (n_c - 1), 0.50, tolerance=1e-9)
same("9.2 so a churner is pushed up by 0.15",
     leaky - (churners - 1) / (n_c - 1), 0.15, tolerance=1e-9)
same("9.2 and a non-churner down by 0.10",
     leaky - churners / (n_c - 1), -0.10, tolerance=1e-9)

# The same two gaps from the formula, which shares no intermediate value.
same("9.2 the formula agrees for the churner",
     (1 - (churners - 1) / (n_c - 1)) / n_c, 0.15, tolerance=1e-9)
same("9.2 and for the non-churner",
     (0 - churners / (n_c - 1)) / n_c, -0.10, tolerance=1e-9)

# The table of weights is one over the group size.
for size, share in ((1, 1.00), (2, 0.50), (5, 0.20), (50, 0.02), (500, 0.002)):
    same(f"9.2 a group of {size} gives your own label {share:.1%}",
         1 / size, share, tolerance=1e-9)

print(f"lesson 2: {checks} hand-worked numbers recomputed, all agree")

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
same("4.1 how often Tukey's fence flags a normal value",
     2 * (1 - norm.cdf(q3 + 1.5 * (q3 - q1))), 0.0070, tolerance=5e-4)

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
X_train, _, _, _ = train_test_split(
    features, frame["churned"], test_size=0.25, random_state=7,
    stratify=frame["churned"])

# --- 5.1, what an outlier does to each scaler ---------------------------
charges = X_train["monthly_charges"].fillna(X_train["monthly_charges"].median())
ordinary = charges[charges < 500]
customer = 80.0

same("5.1 ordinary customers top out at 132.3", ordinary.max(), 128.4, tolerance=0.1)
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

# --- 6.3, the width of one-hot encoding ---------------------------------
levels = frame["zip_code"].nunique()
same("6.3 zip_code has 493 levels", levels, 493, tolerance=0)
same("6.3 averaging 4.06 rows per level", len(frame) / levels, 4.06, tolerance=0.05)

print(f"lesson 2: {checks} hand-worked numbers recomputed, all agree")

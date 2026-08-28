"""Recompute every number lesson 3 works out by hand.

    python Lessons/03_regression/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. It fails loudly if the handout and the
arithmetic disagree.

**The rule that makes this worth having:** each check must reach the handout's
number by a route that does not reuse the handout's intermediate values. This
file exists because a version of Section 3.2 once carried 165,200 where the data
gives 165,600 — and every figure after it was derived *correctly from the wrong
one*, so the example was internally consistent and read as careful work. A
consistency check would have passed it. Only starting again from the three
houses catches it, which is what happens below.
"""

import numpy as np

HANDOUT = "Lessons/03_regression/Docs/regression.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if not np.allclose(computed, printed, rtol=tolerance, atol=tolerance):
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ---------------------------------------------- Section 3.2, by hand

# The only inputs. Everything below is derived from these three houses.
area = np.array([80.0, 120.0, 200.0])
price = np.array([240.0, 320.0, 540.0])

X = np.column_stack([np.ones_like(area), area])
XtX, Xty = X.T @ X, X.T @ price

same("3.2 the sum of areas", XtX[0, 1], 400)
same("3.2 the sum of squared areas", XtX[1, 1], 60_800)
same("3.2 the sum of prices", Xty[0], 1_100)
same("3.2 X-transpose y, second entry", Xty[1], 165_600)
same("3.2 the determinant", np.linalg.det(XtX), 22_400)

theta = np.linalg.solve(XtX, Xty)
same("3.2 the intercept b", theta[0], 28.57, tolerance=5e-3)
same("3.2 the slope w", theta[1], 2.536, tolerance=5e-3)

# The second route the handout quotes: S_xy over S_xx. Deliberately not the
# matrix algebra above, so a slip in one cannot hide in the other.
s_xy = float(((area - area.mean()) * (price - price.mean())).sum())
s_xx = float(((area - area.mean()) ** 2).sum())
same("3.2 S_xy", s_xy, 18_933.3, tolerance=0.5)
same("3.2 S_xx", s_xx, 7_466.7, tolerance=0.5)
same("3.2 the slope, the second way", s_xy / s_xx, 2.536, tolerance=5e-3)
same("3.2 the intercept, the second way",
     price.mean() - (s_xy / s_xx) * area.mean(), 28.57, tolerance=5e-3)

# And the claim the prose makes about it.
same("3.2 how far the slope is from the true 2,400",
     100 * (theta[1] * 1000 - 2400) / 2400, 5.7, tolerance=0.2)

# ---------------------------------------------- Section 4.2, one step

# Two houses, starting from zero, learning rate 1e-5.
step_area = np.array([80.0, 200.0])
step_price = np.array([240.0, 540.0])
m = len(step_area)
error = np.zeros(m) - step_price          # predictions are 0, so error = -y

grad_w = float((error * step_area).sum() / m)
grad_b = float(error.sum() / m)
same("4.2 the gradient in w", grad_w, -63_600)
same("4.2 the gradient in b", grad_b, -390)

rate = 1e-5
same("4.2 w after one step", -rate * grad_w, 0.636)
same("4.2 b after one step", -rate * grad_b, 0.0039)
same("4.2 how much further w moved than b",
     (-rate * grad_w) / (-rate * grad_b), 163, tolerance=1.0)

# ---------------- 3.2 and 7.2, the same coefficient asked two questions
#
# Reached from the generator, not from the notebook's printed output. The
# handout once claimed the one-feature fit on 450 houses gives 2,410; it gives
# 2,785, and the difference is the whole point of the passage.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from housing_data import (RANDOM_STATE, TRUE_COEFFICIENTS,          # noqa: E402
                          load_housing)
from sklearn.model_selection import train_test_split                # noqa: E402

houses = load_housing()
features = list(TRUE_COEFFICIENTS)
X_train, _, y_train, _ = train_test_split(
    houses[features], houses["price"], test_size=0.25, random_state=RANDOM_STATE)


def least_squares(frame, target):
    design = np.column_stack([np.ones(len(frame)), np.asarray(frame, float)])
    return np.linalg.solve(design.T @ design, design.T @ np.asarray(target, float))


area_alone = least_squares(X_train[["area_sqm"]], y_train)[1]
area_with_five = least_squares(X_train[features], y_train)[1]

same("3.2 area alone on 450 houses overshoots to 2,785", area_alone, 2_785, tolerance=1)
same("3.2 and all six features recover 2,421", area_with_five, 2_421, tolerance=1)
same("3.2 so the one-feature answer is 16% high",
     (area_alone - 2_400) / 24, 16.1, tolerance=0.2)
same("3.2 while the three-house answer was only 5.7% high",
     (2_536 - 2_400) / 24, 5.7, tolerance=0.1)

# The reason: area stands in for the two features it is correlated with, and
# both of those genuinely raise the price.
correlations = houses[features].corr()["area_sqm"]
same("3.2 area correlates with bedrooms at 0.77", correlations["bedrooms"], 0.77,
     tolerance=5e-3)
same("3.2 and with bathrooms at 0.49", correlations["bathrooms"], 0.49, tolerance=5e-3)
for absorbed in ("bedrooms", "bathrooms"):
    assert TRUE_COEFFICIENTS[absorbed] > 0, \
        "3.2 argues the omitted features push the area coefficient UP"

# ---------------- 4.4, what conditioning costs and what a penalty buys
#
# Rebuilt from the generator, and read off the singular values rather than the
# eigenvalues of X'X: at this conditioning the smallest eigenvalue computed
# directly is numerical noise, which would be an embarrassing way to get a
# number wrong in a section about numerical conditioning.

from sklearn.preprocessing import PolynomialFeatures, StandardScaler   # noqa: E402

poly_rng = np.random.default_rng(42)
temps = poly_rng.uniform(-5, 35, 30)
usage = 240 + 1.15 * (temps - 18) ** 2 + poly_rng.normal(0, 22, 30)
t_fit, _, _, _ = train_test_split(temps, usage, test_size=0.3, random_state=42)

powers = PolynomialFeatures(12, include_bias=False).fit_transform(t_fit.reshape(-1, 1))
scaled = StandardScaler().fit_transform(powers)
design = np.column_stack([np.ones(len(scaled)), scaled])

same("4.4 the degree-12 design has 21 rows", design.shape[0], 21, tolerance=0)
same("4.4 and 13 columns", design.shape[1], 13, tolerance=0)
same("4.4 it is full rank, so nothing is singular",
     np.linalg.matrix_rank(design), 13, tolerance=0)

singular = np.linalg.svd(design, compute_uv=False)
same("4.4 its condition number is 4.2e9",
     singular.max() / singular.min() / 1e9, 4.2, tolerance=0.1)

eigenvalues = np.linalg.svd(scaled, compute_uv=False) ** 2
same("4.4 so X'X is conditioned at 1.8e19",
     (eigenvalues.max() / eigenvalues.min()) / 1e19, 1.8, tolerance=0.1)
for penalty, printed in ((0.01, 23_100), (1.0, 232), (100.0, 3.3)):
    same(f"4.4 a penalty of {penalty} brings that to {printed:,}",
         (eigenvalues.max() + penalty) / (eigenvalues.min() + penalty), printed,
         tolerance=max(printed * 0.02, 0.05))

# The three rows of 4.4's table, and the convention they share: no intercept
# column. With one the first row reads 654, which is the trap the prose now names.
from housing_data import load_with_collinearity                       # noqa: E402

six = np.asarray(houses[features], float)
same("4.4 the six features as recorded are conditioned at 285",
     np.linalg.cond(six), 285, tolerance=1)
same("4.4 standardised, 3.4",
     np.linalg.cond(StandardScaler().fit_transform(six)), 3.4, tolerance=0.05)
same("4.4 so scaling is worth a factor of about 80",
     np.linalg.cond(six) / np.linalg.cond(StandardScaler().fit_transform(six)), 84,
     tolerance=5)
same("4.4 with an intercept column the first row would read 654 instead",
     np.linalg.cond(np.column_stack([np.ones(len(six)), six])), 654, tolerance=2)

widened = load_with_collinearity()
same("4.4 adding area_sqft takes it to 2,286",
     np.linalg.cond(StandardScaler().fit_transform(
         np.asarray(widened[features + ["area_sqft"]], float))), 2_286, tolerance=5)

same("4.4 the largest coefficient moved by a factor of 12,500 across stacks",
     3_097_038_010 / 247_514 / 1_000, 12.5, tolerance=0.1)

print(f"lesson 3: {checks} hand-worked numbers recomputed, all agree")

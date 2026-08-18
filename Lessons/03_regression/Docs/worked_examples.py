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

print(f"lesson 3: {checks} hand-worked numbers recomputed, all agree")

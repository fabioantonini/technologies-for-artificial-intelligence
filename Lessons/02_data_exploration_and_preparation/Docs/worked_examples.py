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

print(f"lesson 2: {checks} hand-worked numbers recomputed, all agree")

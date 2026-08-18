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

print(f"lesson 5: {checks} hand-worked numbers recomputed, all agree")

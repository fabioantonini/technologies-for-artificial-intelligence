"""Recompute every number lesson 1 works out by hand.

    python Lessons/01_introduction_and_workflow/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. Each check reaches the handout's figure from
the raw inputs, not from the handout's own intermediate values.
"""

import math

HANDOUT = "Lessons/01_introduction_and_workflow/Docs/introduction_and_workflow.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ------------------------------- Section 2.4, the standard error of a score

# The inputs the handout states: notebook 1's test set and its accuracy.
n_test, accuracy = 143, 0.986

# --- 2.1, the zero-one loss lands on that accuracy --------------------------
#
# Under the zero-one loss the empirical risk is the error rate, so the score the
# notebook prints and the risk the formalism minimises are the same number read
# two ways. Reached from the counts, not from the accuracy.

mistakes = n_test - round(n_test * accuracy)
same("2.1 accuracy 0.986 is an empirical risk of 0.014",
     mistakes / n_test, 0.014, tolerance=5e-4)
same("2.1 and that is one minus the accuracy",
     1 - mistakes / n_test, accuracy, tolerance=5e-4)

standard_error = math.sqrt(accuracy * (1 - accuracy) / n_test)
same("2.4 the standard error is about one percentage point",
     standard_error, 0.010, tolerance=0.002)

# And the claim that the error falls as one over root m: quadrupling the test
# set should halve it.
same("2.4 the error falls as 1/sqrt(m)",
     math.sqrt(accuracy * (1 - accuracy) / (4 * n_test)) / standard_error,
     0.5, tolerance=1e-6)

# --- 2.4, why that formula is not admissible on these numbers ---------------
#
# The handout claims four things about the normal approximation here. Each is
# recomputed from the two raw inputs above, and the exact bound is obtained
# twice, by two methods that share no intermediate value.

correct = round(n_test * accuracy)
same("2.4 the test set holds two errors", n_test - correct, 2, tolerance=0)

same("2.4 m*p*(1-p) is about two",
     n_test * accuracy * (1 - accuracy), 2, tolerance=0.1)

same("2.4 the approximate interval runs up to 100.5%",
     accuracy + 1.96 * standard_error, 1.005, tolerance=5e-4)

# Route 1: the exact (Clopper-Pearson) interval, from scipy.
from scipy import stats  # noqa: E402  - kept local to this section

exact_low = stats.binomtest(correct, n_test).proportion_ci(method="exact")[0]
same("2.4 the exact lower bound is 95.0%", exact_low, 0.950, tolerance=5e-4)

# Route 2: the same bound found from its definition, with no library beyond
# math - the p at which observing 141 or more successes out of 143 has
# probability 0.025. Bisection on that tail.
def upper_tail(p_true: float) -> float:
    """P(X >= correct) for X ~ Binomial(n_test, p_true)."""
    return sum(math.comb(n_test, k) * p_true ** k * (1 - p_true) ** (n_test - k)
               for k in range(correct, n_test + 1))


lo, hi = 0.5, 1.0
for _ in range(200):
    mid = (lo + hi) / 2
    if upper_tail(mid) < 0.025:
        lo = mid
    else:
        hi = mid
same("2.4 the exact lower bound again, from the binomial tail",
     (lo + hi) / 2, 0.950, tolerance=5e-4)
same("2.4 and the two routes agree with each other",
     (lo + hi) / 2, exact_low, tolerance=1e-6)

# ------------------------------- Section 7, why leakage answers to m, not p
#
# The handout explains the asymmetry by two growth rates. Both are checked here;
# the measured accuracies themselves are the notebook's committed output, not a
# hand-worked number.

same("7 halving the examples multiplies the noise correlation by 1.41",
     (1 / math.sqrt(100)) / (1 / math.sqrt(200)), 1.41, tolerance=5e-3)

same("7 doubling the columns multiplies the largest of them by 1.04",
     math.sqrt(2 * math.log(10000)) / math.sqrt(2 * math.log(5000)),
     1.04, tolerance=5e-3)

print(f"lesson 1: {checks} hand-worked numbers recomputed, all agree")

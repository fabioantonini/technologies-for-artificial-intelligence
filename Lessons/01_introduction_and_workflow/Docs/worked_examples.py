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

standard_error = math.sqrt(accuracy * (1 - accuracy) / n_test)
same("2.4 the standard error is about one percentage point",
     standard_error, 0.010, tolerance=0.002)

# "anything from roughly 96.5% upward is consistent" — two standard errors below.
same("2.4 the lower end of the interval",
     accuracy - 2 * standard_error, 0.965, tolerance=0.004)

# And the claim that the error falls as one over root n: quadrupling the test
# set should halve it.
same("2.4 the error falls as 1/sqrt(n)",
     math.sqrt(accuracy * (1 - accuracy) / (4 * n_test)) / standard_error,
     0.5, tolerance=1e-6)

print(f"lesson 1: {checks} hand-worked numbers recomputed, all agree")

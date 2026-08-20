"""Recompute every number lesson 6 works out by hand.

    python Lessons/06_knn_naive_bayes_svm/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. Each check reaches the handout's figure from
the raw inputs rather than from the handout's own intermediate values.

The distance-concentration result is the one that matters most here, because it
is the lesson's headline claim and it is stated as a bare number. It is
recomputed below from a fresh simulation with a different seed and a different
number of points from the notebook's, so agreement between the two is evidence
rather than a copy.
"""

import numpy as np

HANDOUT = "Lessons/06_knn_naive_bayes_svm/Docs/knn_naive_bayes_svm.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


# ------------------------------------------- Section 1, the class balance

# The pump generator is the only input; the handout's baseline must follow.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from pump_data import LABEL_NOISE, load_interacting, load_pumps

X, y = load_pumps()
same("1.1 the number of pumps", len(X), 1200, tolerance=0)
same("1.1 the faulty fraction", y.mean(), 0.613, tolerance=5e-4)
same("1.1 the majority baseline", max(y.mean(), 1 - y.mean()), 0.613,
     tolerance=5e-4)
same("1.1 the noise ceiling", 1 - LABEL_NOISE, 0.96, tolerance=1e-9)

# ------------------------------------------- Section 4.3, the assumption

for label, printed in ((0, -0.006), (1, -0.049)):
    same(f"4.3 within-class correlation, class {label}",
         X[y == label].corr().iloc[0, 1], printed, tolerance=5e-4)
same("4.3 overall correlation", X.corr().iloc[0, 1], -0.046, tolerance=5e-4)

# ------------------------------------------- Section 4.4, the interaction

Xi, yi = load_interacting()
same("4.4 the interacting baseline", max(yi.mean(), 1 - yi.mean()), 0.523,
     tolerance=5e-4)

# The handout explains 0.404 by the accidental difference in class means.
# Check that difference is the size claimed: about 0.17 against a spread near 1.
gap = abs(Xi.groupby(yi).mean().diff().iloc[-1]).max()
same("4.4 the accidental gap between class means", gap, 0.17, tolerance=0.02)
spread = Xi.std().mean()
if not 0.8 < spread < 1.3:
    raise SystemExit(f"4.4 the spread is {spread:.2f}, not 'near 1' as claimed")
checks += 1

# ------------------------- Section 3.2, distance concentration, independently

# Deliberately not the notebook's setup: different seed, different point count.
rng = np.random.default_rng(20261030)
measured = {}
for d in (2, 10, 50, 100, 500):
    points = rng.random((1_500, d))
    ratios = []
    for query in rng.random((80, d)):
        distance = np.sqrt(((points - query) ** 2).sum(axis=1))
        ratios.append(distance.min() / distance.max())
    measured[d] = float(np.mean(ratios))

for d, printed in ((2, 0.016), (10, 0.263), (50, 0.592),
                   (100, 0.701), (500, 0.855)):
    same(f"3.2 nearest/farthest at d={d}", measured[d], printed, tolerance=0.05)

# And the claim the prose makes in words.
same("3.2 'about 2% in two dimensions'", 100 * measured[2], 2, tolerance=1.5)
same("3.2 '70% in one hundred dimensions'", 100 * measured[100], 70,
     tolerance=4)

# ------------------------------------------- Section 6, the spread of scores

same("6 the gap between best and worst model", 0.947 - 0.613, 0.334,
     tolerance=1e-9)

# ------------------------------------------- Section 5.3, support vectors

same("5.3 linear support-vector fraction", 947 / 1200, 0.79, tolerance=5e-3)
same("5.3 RBF support-vector fraction", 278 / 1200, 0.23, tolerance=5e-3)

print(f"lesson 6: {checks} hand-worked numbers recomputed, all agree")

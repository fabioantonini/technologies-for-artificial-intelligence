"""Data for lesson 5, where the question is how to measure honestly.

Two sources, each chosen for what it lets us prove.

**The disk fleet from lesson 4**, deliberately reused and deliberately *small*.
Lesson 4 had 8,000 drives, which is enough that a single train/test split is
fairly stable. Cut it to 800 and the instability becomes visible — which is the
whole point of this lesson, and it is honest, because 800 records with 29
positives is a far more typical situation than 8,000.

**A fresh energy curve**, sampled again and again from a known function. Because
the truth is a formula rather than a dataset, we can draw three hundred
independent training sets from it and *measure* bias and variance instead of
arguing about them. No real dataset permits that: you only ever get one sample.

    from methodology_data import load_fleet, sample_energy, true_curve

    X, y = load_fleet(800)
"""

import sys
from pathlib import Path

import numpy as np

RANDOM_STATE = 42

# ---------------------------------------------------------------- the fleet

#: Lesson 4's generator is the single source of truth for this data; importing
#: it beats copying it, and the continuity is the point — same drives, same
#: model, a better question.
_LESSON_4 = (Path(__file__).resolve().parent.parent.parent
             / "04_classification_and_metrics" / "Notebooks")


def load_fleet(n_samples: int = 800, random_state: int = RANDOM_STATE):
    """Return (X, y) for the disk fleet, transformed as in lesson 4."""
    if str(_LESSON_4) not in sys.path:
        sys.path.insert(0, str(_LESSON_4))
    from disk_data import load_drives, transform_features

    frame = load_drives(n_samples=n_samples, random_state=random_state)
    return transform_features(frame), frame["failed"]


# --------------------------------------------------------- the energy curve

#: Standard deviation of the measurement noise, in kWh. This is the part of the
#: error no model can ever remove, and knowing it exactly is what makes the
#: decomposition in notebook 2 checkable.
NOISE_SD = 22.0


def true_curve(temperature):
    """Daily energy use against outdoor temperature.

    Heating below and cooling above, so the curve has a minimum near 18 °C; the
    sine term is a mild seasonal habit that a straight line cannot follow and a
    quadratic mostly can. No model in the lesson is given this function.
    """
    return 240 + 0.55 * (temperature - 18) ** 2 + 12 * np.sin(temperature / 4)


def sample_energy(n_samples: int, rng: np.random.Generator):
    """Draw one independent training set from the curve.

    Called repeatedly with the same generator, it produces the many parallel
    universes that the bias-variance decomposition averages over.
    """
    temperature = rng.uniform(-5, 38, n_samples)
    energy = true_curve(temperature) + rng.normal(0, NOISE_SD, n_samples)
    return temperature, energy


#: A fixed grid to evaluate on, so every model in notebook 2 is judged at the
#: same points and the numbers are comparable across degrees.
TEST_TEMPERATURE = np.linspace(-4, 37, 220)
TEST_TRUTH = true_curve(TEST_TEMPERATURE)


if __name__ == "__main__":
    X, y = load_fleet()
    print(f"fleet: {len(X)} drives, {int(y.sum())} failures ({y.mean():.1%})")

    rng = np.random.default_rng(0)
    temperature, energy = sample_energy(25, rng)
    print(f"energy: {len(temperature)} observations, "
          f"{energy.min():.0f} to {energy.max():.0f} kWh")
    print(f"irreducible noise: {NOISE_SD} kWh, so a variance of "
          f"{NOISE_SD ** 2:.0f} that no model can remove")

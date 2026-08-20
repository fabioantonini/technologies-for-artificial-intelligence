"""Industrial pump telemetry for lesson 6.

A pump has a design operating envelope. Inside it the machine is healthy;
outside it — whether the operating point is too low **or** too high — something
is wrong. So the healthy pumps form a *disc* and the faulty ones the annulus
around it, and no straight line separates the two.

That is the whole reason this dataset exists. Lessons 3 and 4 fitted straight
lines and got useful models. Here a straight line is worthless, and the three
methods of this lesson each attack the problem from a different direction:
k-nearest neighbours by remembering, Naive Bayes by assuming independence, and a
support vector machine by bending the space until a line does work.

Three loaders, three jobs.

``load_pumps``          two readings, a ring-shaped boundary, drawable.
``load_with_noise``     the same problem plus d columns of pure noise, for the
                        curse of dimensionality.
``load_interacting``    features that say nothing apart and everything
                        together, which is exactly the assumption Naive
                        Bayes makes and exactly where it breaks.

Everything is offline and reproducible: same seed, same numbers, anywhere.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

#: The pump's design operating point, in the two measured units, and the radius
#: of the envelope around it. A reading further than this from the target — in
#: either direction — is a fault.
TARGET_VIBRATION = 42.0     # Hz
TARGET_PRESSURE = 5.6       # bar
ENVELOPE_RADIUS = 1.0       # in standardised units, see below

#: The scales the two readings live on. Faults are defined on the standardised
#: distance, so the envelope is a circle there and an ellipse in raw units —
#: which is realistic, and a reminder that scaling is not cosmetic here.
VIBRATION_SD = 3.5
PRESSURE_SD = 0.45

#: Fraction of readings whose label is flipped: sensors are not perfect, and a
#: boundary with no noise at all teaches the wrong lesson about accuracy.
LABEL_NOISE = 0.04


def _standardised_radius(vibration, pressure):
    """Distance from the design point, in standard deviations."""
    dv = (vibration - TARGET_VIBRATION) / VIBRATION_SD
    dp = (pressure - TARGET_PRESSURE) / PRESSURE_SD
    return np.sqrt(dv ** 2 + dp ** 2)


def load_pumps(n_samples: int = 1_200, random_state: int = RANDOM_STATE):
    """Two readings per pump, and whether it is faulty.

    Returns (X, y) with X a DataFrame of two columns. About 61% of the pumps
    are faulty, because the envelope is small compared with the range the
    readings cover; the healthy ones form a disc around the design point and
    the faulty ones surround it.
    """
    rng = np.random.default_rng(random_state)

    # Sample radius and angle directly, so both classes are properly populated
    # rather than leaving the faulty class to a thin tail.
    radius = rng.uniform(0.0, 2.6, n_samples)
    angle = rng.uniform(0, 2 * np.pi, n_samples)

    vibration = TARGET_VIBRATION + VIBRATION_SD * radius * np.cos(angle)
    pressure = TARGET_PRESSURE + PRESSURE_SD * radius * np.sin(angle)

    faulty = (radius > ENVELOPE_RADIUS).astype(int)
    flip = rng.random(n_samples) < LABEL_NOISE
    faulty = np.where(flip, 1 - faulty, faulty)

    X = pd.DataFrame({"vibration_hz": vibration.round(3),
                      "pressure_bar": pressure.round(4)})
    return X, pd.Series(faulty, name="faulty")


def load_with_noise(n_noise: int, n_samples: int = 1_200,
                    random_state: int = RANDOM_STATE):
    """The same problem, with `n_noise` columns that contain nothing.

    The two real readings are unchanged, so any drop in performance is the
    dimensionality alone rather than a harder problem. Used to show that
    distance stops discriminating long before the signal disappears.
    """
    X, y = load_pumps(n_samples, random_state)
    if n_noise == 0:
        return X, y
    rng = np.random.default_rng(random_state + 1)
    noise = pd.DataFrame(
        rng.normal(size=(len(X), n_noise)),
        columns=[f"noise_{i:03d}" for i in range(n_noise)], index=X.index)
    return pd.concat([X, noise], axis=1), y


def load_interacting(n_samples: int = 1_200, random_state: int = RANDOM_STATE):
    """Two sensors that mean nothing apart and everything together.

    A pump is faulty when exactly one of the two readings is high. Both high is
    the designed high-load mode and both low is idle; either one alone is a
    mismatch between demand and delivery, and that is the fault.

    So each column on its own is uninformative — look at sensor A and the two
    classes have the same distribution. That is precisely what Naive Bayes
    cannot represent: it multiplies per-feature evidence, and here there is none
    to multiply. The structure lives entirely in the interaction.
    """
    rng = np.random.default_rng(random_state + 2)

    high_a = rng.random(n_samples) < 0.5
    high_b = rng.random(n_samples) < 0.5
    faulty = (high_a != high_b).astype(int)

    sensor_a = np.where(high_a, 1.0, -1.0) + rng.normal(0, 0.45, n_samples)
    sensor_b = np.where(high_b, 1.0, -1.0) + rng.normal(0, 0.45, n_samples)

    X = pd.DataFrame({"sensor_a": sensor_a.round(4),
                      "sensor_b": sensor_b.round(4)})
    return X, pd.Series(faulty, name="faulty")


if __name__ == "__main__":
    X, y = load_pumps()
    print(X.head())
    print(f"\n{len(X)} pumps, {int(y.sum())} faulty ({y.mean():.1%})")
    print(f"vibration {X['vibration_hz'].min():.1f} to "
          f"{X['vibration_hz'].max():.1f} Hz")
    print(f"pressure  {X['pressure_bar'].min():.2f} to "
          f"{X['pressure_bar'].max():.2f} bar")

    Xc, yc = load_interacting()
    print(f"\ninteracting set: {int(yc.sum())} faulty of {len(yc)}")
    print(f"strongest correlation between one sensor and the label: "
          f"{Xc.corrwith(yc).abs().max():.3f}")

"""Wind-turbine gearbox telemetry for exercise 5.

A wind farm records vibration and temperature from each turbine's gearbox once a
week, and wants to predict which gearboxes will need an unplanned repair in the
following quarter.

Two properties matter for this exercise, and both are deliberate.

**There are several readings per turbine.** Forty turbines, twelve weekly
readings each. The rows are therefore *not* independent draws — which is the
situation `GroupKFold` exists for, and the situation a random split handles
badly and silently. The turbine identifier is in the ``turbine`` column.

**The label belongs to the turbine, not to the reading.** A gearbox either
needed a repair that quarter or it did not, and every reading from that turbine
carries the same label. That is realistic, and it is what makes a random split
so misleading here.

    from turbine_data import load_turbines

    df = load_turbines()

The generating coefficients are in ``TRUE_COEFFICIENTS``. As in lesson 4, do not
read them until the exercise tells you to.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 11

N_TURBINES = 40
READINGS_PER_TURBINE = 12

#: Change in the log-odds of needing a repair per standard deviation, acting on
#: the turbine's *average* condition over the quarter. `ambient_humidity` is the
#: decoy: plausible, measurable, unrelated.
TRUE_COEFFICIENTS = {
    "vibration_rms": 1.05,
    "oil_temperature_c": 0.55,
    "particle_count": 0.45,
    "power_output_kw": -0.25,
    "ambient_humidity": 0.00,
}

TRUE_INTERCEPT = -0.60

#: The weekly telemetry.
FEATURES = list(TRUE_COEFFICIENTS)

#: Installation metadata: recorded once per turbine and repeated on every one of
#: its rows. None of it causes a gearbox failure — but because each turbine has
#: its own combination of values, together these columns act as a **fingerprint**
#: that identifies the turbine. A model given them, and a split that puts the
#: same turbine on both sides, can memorise which turbines failed instead of
#: learning anything about failing.
METADATA = ["hub_height_m", "rotor_diameter_m", "site_elevation_m",
            "commissioning_year", "gearbox_ratio", "blade_batch",
            "foundation_depth_m", "grid_distance_km"]

ALL_COLUMNS = FEATURES + METADATA


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_turbines(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return one row per weekly reading, with a per-turbine `needs_repair`."""
    rng = np.random.default_rng(random_state)

    # Each turbine has a persistent condition; readings vary around it. That
    # persistence is exactly what a random split lets a model memorise.
    condition = rng.normal(0, 1, N_TURBINES)

    # Installation metadata: drawn once per turbine, then repeated on its rows.
    metadata = pd.DataFrame({
        "hub_height_m": rng.choice([80, 90, 100, 110, 120], N_TURBINES),
        "rotor_diameter_m": rng.choice([90, 100, 112, 126], N_TURBINES),
        "site_elevation_m": rng.integers(120, 900, N_TURBINES),
        "commissioning_year": rng.integers(2009, 2021, N_TURBINES),
        "gearbox_ratio": rng.choice([97.0, 105.5, 113.0, 119.5], N_TURBINES),
        "blade_batch": rng.integers(1, 14, N_TURBINES),
        "foundation_depth_m": rng.uniform(2.5, 9.5, N_TURBINES).round(1),
        "grid_distance_km": rng.uniform(0.4, 31.0, N_TURBINES).round(1),
    })

    rows = []
    for turbine in range(N_TURBINES):
        for week in range(READINGS_PER_TURBINE):
            row = {
                "turbine": f"T{turbine:02d}",
                "week": week + 1,
                # Weekly readings are noisy around the turbine's condition.
                "vibration_rms": 2.4 + 0.85 * condition[turbine] + rng.normal(0, 0.55),
                "oil_temperature_c": 61 + 4.5 * condition[turbine] + rng.normal(0, 3.4),
                "particle_count": np.exp(3.1 + 0.55 * condition[turbine]
                                         + rng.normal(0, 0.55)),
                "power_output_kw": 1_850 - 90 * condition[turbine] + rng.normal(0, 120),
                "ambient_humidity": rng.uniform(35, 92),
            }
            row.update(metadata.iloc[turbine].to_dict())
            rows.append(row)
    frame = pd.DataFrame(rows)

    # The label acts on the turbine's average condition over the quarter, and is
    # then copied onto every reading from that turbine.
    averages = frame.groupby("turbine")[FEATURES].mean()
    standardised = (averages - averages.mean()) / averages.std(ddof=0)
    logit = TRUE_INTERCEPT + sum(
        weight * standardised[name] for name, weight in TRUE_COEFFICIENTS.items())
    repaired = pd.Series(
        (rng.random(len(averages)) < _sigmoid(logit)).astype(int),
        index=averages.index, name="needs_repair")

    frame = frame.merge(repaired, left_on="turbine", right_index=True)
    frame[["vibration_rms", "oil_temperature_c", "power_output_kw",
           "ambient_humidity"]] = frame[
        ["vibration_rms", "oil_temperature_c", "power_output_kw",
         "ambient_humidity"]].round(2)
    frame["particle_count"] = frame["particle_count"].round(0).astype(int)
    return frame


if __name__ == "__main__":
    df = load_turbines()
    per_turbine = df.groupby("turbine")["needs_repair"].first()
    print(df.head())
    print(f"\n{len(df)} readings from {df['turbine'].nunique()} turbines")
    print(f"{int(per_turbine.sum())} turbines needed a repair "
          f"({per_turbine.mean():.0%} of the fleet)")
    print(f"rows labelled 1: {df['needs_repair'].mean():.0%}")

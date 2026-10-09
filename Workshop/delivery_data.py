"""The parcels for the workshop: which deliveries arrive late.

Students load the data with ``load_deliveries()`` and should not need to open
this file. **Do not read further before the end of the workshop**: the rule
that decides which parcels are late is written out below, and the instructor
reveals it in the closing discussion.

Every column is generated from a fixed seed, so every laptop sees exactly the
same 1,200 parcels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 7
N_PARCELS = 1_200

#: Fraction of rain readings lost to a gauge that drops out at random.
RAIN_MISSING = 0.10


def lateness_logit(df: pd.DataFrame) -> np.ndarray:
    """The rule. Log-odds of a parcel arriving late.

    Longer routes and fuller vans are later; experienced drivers are earlier;
    rain slows everything down. Departure time matters through rush hour: a
    van leaving far from the quiet middle of the day (13:00) meets traffic in
    both directions, so the effect grows with (hour - 13) squared - a curve no
    straight line in `departure_hour` can follow. `vehicle_age_years` has no
    effect at all.
    """
    return (
        -9.6
        + 0.1575 * (df["distance_km"] - 30)
        + 0.07 * (df["parcels_on_van"] - 90)
        - 0.315 * (df["driver_years"] - 8)
        + 0.44 * (df["departure_hour"] - 13) ** 2
        + 0.70 * df["rain_mm"]
    ).to_numpy()


def load_deliveries(n_parcels: int = N_PARCELS,
                    random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """Return the features and the target `late` (1 = arrived late)."""
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({
        "distance_km": rng.uniform(2, 60, n_parcels).round(1),
        "parcels_on_van": rng.integers(20, 161, n_parcels),
        "driver_years": rng.integers(0, 26, n_parcels),
        "departure_hour": rng.uniform(7, 19, n_parcels).round(2),
        "rain_mm": rng.exponential(3.0, n_parcels).round(1),
        "vehicle_age_years": rng.integers(0, 16, n_parcels),
    })
    p_late = 1 / (1 + np.exp(-lateness_logit(df)))
    late = pd.Series((rng.random(n_parcels) < p_late).astype(int), name="late")
    gone = rng.random(n_parcels) < RAIN_MISSING
    df.loc[gone, "rain_mm"] = np.nan
    return df, late

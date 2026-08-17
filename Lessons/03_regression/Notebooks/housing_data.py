"""Synthetic housing data for lesson 3.

Real datasets hide their answer. This one does not: the coefficients that
generated the prices are written down below, so every estimate in the lesson
can be checked against the truth. That is what makes it possible to say
"ridge shrank this coefficient by 12%" rather than "ridge shrinks
coefficients".

    from housing_data import load_housing, TRUE_COEFFICIENTS

    df = load_housing()

Everything is offline and reproducible: same seed, same numbers, on any
machine.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

#: Euros of price attributable to one unit of each feature. The lesson checks
#: its estimates against these.
TRUE_COEFFICIENTS = {
    "area_sqm": 2_400.0,        # per square metre
    "bedrooms": 9_000.0,        # per bedroom, on top of the area it adds
    "bathrooms": 14_000.0,      # per bathroom
    "age_years": -1_800.0,      # depreciation per year
    "distance_km": -6_500.0,    # per km from the centre
    "garage": 12_000.0,         # binary
}
TRUE_INTERCEPT = 45_000.0

#: Standard deviation of the price noise, in euros. Large enough that the
#: estimates are visibly uncertain on small samples, small enough that the
#: signal is unmistakable.
NOISE_SD = 18_000.0


def load_housing(n_samples: int = 600, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a housing table whose price is a known linear function plus noise.

    The features are correlated the way they are in reality - bigger houses
    have more bedrooms, houses far from the centre are newer - which is what
    makes the coefficient estimates interesting rather than trivial.
    """
    rng = np.random.default_rng(random_state)

    area = rng.normal(110, 35, n_samples).clip(35, 260)
    # Bedrooms follow area, with slack: correlation without determination.
    bedrooms = np.clip(np.round(area / 38 + rng.normal(0, 0.6, n_samples)), 1, 6)
    bathrooms = np.clip(np.round(bedrooms / 2 + rng.normal(0, 0.45, n_samples)), 1, 4)
    distance = rng.gamma(2.4, 2.6, n_samples).clip(0.3, 22)
    # Suburbs were built later, so age falls as distance rises.
    age = np.clip(rng.normal(45 - 1.4 * distance, 14, n_samples), 0, 95)
    garage = (rng.random(n_samples) < 0.35 + 0.02 * distance).astype(int)

    frame = pd.DataFrame({
        "area_sqm": area.round(1),
        "bedrooms": bedrooms.astype(int),
        "bathrooms": bathrooms.astype(int),
        "age_years": age.round(0).astype(int),
        "distance_km": distance.round(2),
        "garage": garage,
    })

    price = TRUE_INTERCEPT + sum(
        coefficient * frame[name] for name, coefficient in TRUE_COEFFICIENTS.items()
    )
    price = price + rng.normal(0, NOISE_SD, n_samples)
    # A floor, so noise cannot produce a house that costs nothing. It bites on
    # roughly one row in five hundred and keeps the target plausible.
    price = price.clip(lower=25_000)
    frame["price"] = price.round(-2).astype(int)  # to the nearest 100 euros

    return frame


def load_with_collinearity(n_samples: int = 600,
                           random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """The same data plus `area_sqft`, which is `area_sqm` in other units.

    An exactly redundant column, of the kind that arrives in real extracts when
    two systems were merged. Used in the regularisation section to show what
    happens to ordinary least squares when two columns carry one fact.
    """
    frame = load_housing(n_samples, random_state)
    rng = np.random.default_rng(random_state + 1)
    # Not an exact multiple: a rounding error, as a real conversion would have.
    frame.insert(1, "area_sqft", (frame["area_sqm"] * 10.7639
                                  + rng.normal(0, 0.4, len(frame))).round(1))
    return frame


if __name__ == "__main__":
    df = load_housing()
    print(df.head())
    print(f"\n{len(df)} houses")
    print(f"price: {df['price'].min():,} to {df['price'].max():,} euros")

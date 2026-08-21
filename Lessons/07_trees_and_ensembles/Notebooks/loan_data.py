"""Synthetic loan-default data for lesson 7.

A bank's real underwriting rules are rarely a single straight line through
income and debt. They are closer to a checklist: an income floor below which
almost nobody is approved, a debt ceiling above which almost nobody is,  and
one or two "stressed" combinations in between — income and debt levels that
are each unremarkable alone but risky together. That is exactly the shape a
decision tree represents natively (a handful of axis-aligned cuts) and a
linear model represents badly (the risky region is not one half-plane, it is
several disconnected patches).

Two loaders, two jobs.

``load_loans``       two features, the rule-based boundary, drawable.
``load_with_noise``  the same two real features plus ``n_noise`` columns of
                      pure noise, for testing whether feature importance and
                      cross-validated accuracy survive irrelevant columns.

Everything is offline and reproducible: same seed, same numbers, anywhere.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

#: Below this income, in thousands of euros per year, default regardless of
#: debt: there is no buffer left for anything to go wrong.
TRUE_INCOME_FLOOR = 28.0

#: Above this debt-to-income ratio, default regardless of income: the
#: repayment no longer fits the budget no matter how much comes in.
TRUE_DEBT_CEIL = 0.82

#: Two "stressed" rectangles: income and debt that are each unremarkable on
#: their own but risky in that specific combination. Island A sits at
#: low-to-middle income with moderate-to-high debt; island B sits at
#: middle-to-high income with a debt ratio that would be fine at island A's
#: income but is stretched relative to the higher repayments income implies.
TRUE_ISLAND_A_INCOME = (35.0, 55.0)
TRUE_ISLAND_A_DEBT = (0.40, 0.70)
TRUE_ISLAND_B_INCOME = (65.0, 85.0)
TRUE_ISLAND_B_DEBT = (0.55, 0.75)

#: Fraction of labels flipped after the rule is applied: real repayment
#: outcomes depend on things not in this dataset (a job loss, a medical
#: bill), so a rule with zero noise would teach the wrong lesson about
#: what accuracy of 1.000 should mean.
LABEL_NOISE = 0.07


def _true_default(income, debt):
    """The noise-free rule. Vectorised over NumPy arrays."""
    island_a = (
        (income >= TRUE_ISLAND_A_INCOME[0]) & (income <= TRUE_ISLAND_A_INCOME[1])
        & (debt >= TRUE_ISLAND_A_DEBT[0]) & (debt <= TRUE_ISLAND_A_DEBT[1])
    )
    island_b = (
        (income >= TRUE_ISLAND_B_INCOME[0]) & (income <= TRUE_ISLAND_B_INCOME[1])
        & (debt >= TRUE_ISLAND_B_DEBT[0]) & (debt <= TRUE_ISLAND_B_DEBT[1])
    )
    return (
        (income < TRUE_INCOME_FLOOR) | (debt > TRUE_DEBT_CEIL) | island_a | island_b
    ).astype(int)


def load_loans(n_samples: int = 1_200, random_state: int = RANDOM_STATE):
    """Two features per applicant, and whether the loan later defaulted.

    Returns (X, y) with X a DataFrame of ``income_k`` (thousands of euros a
    year, 20 to 100) and ``debt_ratio`` (0 to 1). About 39% of applicants
    default; the healthy region is most of the plane and the risky region is
    four disconnected patches — a floor, a ceiling, and two stressed islands
    in between — so no single straight line separates the classes.
    """
    rng = np.random.default_rng(random_state)
    income = rng.uniform(20.0, 100.0, n_samples)
    debt = rng.uniform(0.0, 1.0, n_samples)

    default = _true_default(income, debt)
    flip = rng.random(n_samples) < LABEL_NOISE
    default = np.where(flip, 1 - default, default)

    X = pd.DataFrame({"income_k": income.round(2), "debt_ratio": debt.round(4)})
    return X, pd.Series(default, name="default")


def load_with_noise(n_noise: int, n_samples: int = 1_200,
                     random_state: int = RANDOM_STATE):
    """The same problem, with `n_noise` columns that carry nothing.

    The two real features are unchanged, so any importance a noise column
    receives, or any accuracy lost, is the effect of irrelevant columns
    alone rather than a harder underlying problem.
    """
    X, y = load_loans(n_samples, random_state)
    if n_noise == 0:
        return X, y
    rng = np.random.default_rng(random_state + 1)
    noise = pd.DataFrame(
        rng.normal(size=(len(X), n_noise)),
        columns=[f"noise_{i:02d}" for i in range(n_noise)], index=X.index)
    return pd.concat([X, noise], axis=1), y


if __name__ == "__main__":
    X, y = load_loans()
    print(X.head())
    print(f"\n{len(X)} applicants, {int(y.sum())} defaulted ({y.mean():.1%})")
    print(f"income  {X['income_k'].min():.1f} to {X['income_k'].max():.1f} k EUR")
    print(f"debt    {X['debt_ratio'].min():.2f} to {X['debt_ratio'].max():.2f}")

    Xn, yn = load_with_noise(20)
    print(f"\nwith noise: {Xn.shape[1]} columns, {Xn.shape[1] - 2} of them pure noise")

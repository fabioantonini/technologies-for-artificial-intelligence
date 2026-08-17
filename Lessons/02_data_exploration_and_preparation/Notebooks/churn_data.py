"""Shared synthetic dataset for lesson 2's three notebooks.

    from churn_data import make_churn_data

Every real, publicly hosted "messy" teaching dataset (Titanic, Ames housing,
Adult census) sits behind a network call this course cannot depend on being
available in every room. This dataset is generated instead, with numpy and a
fixed seed, so every notebook that imports it sees byte-identical data
offline and forever - and, usefully for a lesson about mess, we know the
ground truth: which columns are genuinely predictive, which are pure noise,
and which missing values are MCAR versus MAR. Real projects rarely afford
that luxury; use it here to see the mechanisms clearly before meeting them
in the wild.

The story: whether a telecom customer churns (cancels) in the next month.
"""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "age", "num_support_calls"]
LOW_CARD_CATEGORICAL = ["contract_type", "region"]
HIGH_CARD_CATEGORICAL = ["zip_code"]
TARGET = "churned"


def make_churn_data(n=2000, seed=7):
    """Return a DataFrame of n synthetic telecom customers with realistic mess."""
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 73, n).astype(float)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20]
    )
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    age = rng.normal(42, 14, n).round().clip(18, 90)
    num_support_calls = rng.poisson(1.2, n).astype(float)

    base_charge = {"month-to-month": 70.0, "one-year": 60.0, "two-year": 50.0}
    monthly_charges = np.array([base_charge[c] for c in contract_type])
    monthly_charges = monthly_charges + rng.normal(0, 15, n)
    monthly_charges = monthly_charges.clip(15, None)

    # 500 zip codes over 2000 customers - about 4 per code, several with only
    # one. High cardinality is what makes target encoding dangerous.
    zip_code = np.array([f"Z{z:03d}" for z in rng.integers(0, 500, n)])

    # Ground truth: churn depends on tenure, contract, charges and support
    # calls. It does NOT depend on zip_code or region - both are noise with
    # respect to the label, which is exactly what makes them useful later for
    # showing what a leak manufactures out of nothing.
    contract_effect = np.select(
        [contract_type == "month-to-month", contract_type == "one-year"],
        [1.1, 0.1],
        default=-0.6,
    )
    logits = (
        -0.045 * tenure_months
        + contract_effect
        + 0.012 * (monthly_charges - 70)
        + 0.28 * num_support_calls
        - 1.0
    )
    p_churn = 1 / (1 + np.exp(-logits))
    churned = rng.binomial(1, p_churn)

    # Billing errors: ~1% of charges are entered with an extra digit or two.
    billing_error_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    monthly_charges[billing_error_idx] *= rng.uniform(15, 40, len(billing_error_idx))

    # Data entry errors: ~0.5% of tenures are physically impossible.
    bad_tenure_idx = rng.choice(n, size=int(0.005 * n), replace=False)
    tenure_months[bad_tenure_idx] = rng.choice([-3, -1, 999], len(bad_tenure_idx))

    # MCAR: age goes missing for reasons unrelated to anything (~8%) - a
    # field a customer simply chose not to fill in on the sign-up form.
    mcar_idx = rng.choice(n, size=int(0.08 * n), replace=False)
    age[mcar_idx] = np.nan

    # MAR: support-call counts go missing more often for long-tenure
    # customers, whose early call history predates the current CRM system.
    # The chance of being missing depends on an OBSERVED column (tenure),
    # not on the (unrecorded) call count itself.
    tenure_z = (tenure_months - tenure_months.mean()) / tenure_months.std()
    mar_prob = np.clip(0.05 + 0.05 * tenure_z, 0.01, 0.55)
    mar_mask = rng.random(n) < mar_prob
    num_support_calls[mar_mask] = np.nan

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "age": age,
            "num_support_calls": num_support_calls,
            "contract_type": contract_type,
            "region": region,
            "zip_code": zip_code,
            "churned": churned,
        }
    )

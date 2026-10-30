"""Two account books from the same bank, for exercise 8.

Meridian is a fictional retail-and-commercial bank. Its fraud-analytics team
monitors every account for signs of takeover — an attacker who has gained
control of an account and is quietly testing how much they can move before
someone notices. Nobody has labelled a single account "compromised" or
"clean" ahead of time; that is exactly the situation unsupervised learning
is for. This module gives you **two account books from the same bank**,
generated the same underlying way, so that any difference in how well a
detector works comes from how each book's anomalies were built, not from
the two books being different kinds of data.

``load_retail_accounts``    six features, personal current accounts.
``load_business_accounts``  the same six features, business current accounts.

Both books share one genuine-account generator: every ordinary account,
retail or business, is driven by the same two latent factors and the same
noise. Two independent standard-normal factors, ``cash_flow_scale`` and
``activity_level``, generate all six observed columns — so the honest
dimensionality of either book is 2, not 6. ``cash_flow_scale`` mostly
governs how much money moves through the account (inflow, outflow, average
transaction size); ``activity_level`` mostly governs how often the account
is touched (transaction count, night-time share, new payees added). Neither
factor is ever shown to any method below.

A small fraction of accounts in each book (``TRUE_ANOMALY_FRACTION``) are
overwritten with an account-takeover pattern — but the *pattern differs
between the two books*, and that difference is the entire exercise.

Retail — a credential-stuffing takeover
----------------------------------------

Someone has phished a retail customer's login. They cannot move large sums
immediately (the bank's own transfer limits see to that), so what they
actually do, in the account's activity for the week under review, is add
several new payees and probe the account at night, when a legitimate
retail customer rarely logs in. Everything about how much money moves
through the account — inflow, outflow, average transaction size, overall
transaction count — is left untouched, still drawn from the same genuine
generating process as every honest retail account. Only two of the six
columns move, and they move a long way: ``night_transaction_ratio`` and
``new_payee_count`` are pushed well outside the range any genuine account
occupies.

Business — a mule account
----------------------------------------

A shell company is opened purely to launder funds through Meridian's
business book. Whoever set the account up did not have access to any real
business's books, so instead they picked numbers that *sound* plausible for
some small business — an inflow here, a transaction count there — each one
independently, from roughly the range genuine business accounts occupy.
What they could not fake, because they did not know it existed, is the
*relationship between the six numbers*: in every genuine business account,
inflow, outflow, transaction count, average transaction size, night ratio
and new-payee count all trace back to the same two underlying factors, so
they move together in a fixed way. The mule account's numbers are each
individually unremarkable and never move together at all.

**Try this, before reading any further:** for the retail book, would you
expect the takeover accounts to sit near the middle of a two-dimensional
scatter of the data, or out towards an edge? Now ask the same question
about the business book's mule accounts, given that their construction is
different. Section 6.2 of the handout describes exactly this kind of
mismatch for Aurora's account table — the mechanism here is the same idea,
built into a different pair of datasets so that the answer cannot be
recalled, only worked out.

The ``TRUE_*`` constants below record exactly how each book was built,
feature by feature. **Do not read past the loader functions until Part 4.**

Both loaders are offline, seeded with NumPy's ``default_rng``, and return
the same numbers on any machine. Neither needs a network connection or an
API key.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared genuine-account generator
# ---------------------------------------------------------------------------

#: The six columns on file for every account, retail or business.
FEATURES = [
    "monthly_inflow_eur",
    "monthly_outflow_eur",
    "transaction_count",
    "avg_transaction_size_eur",
    "night_transaction_ratio",
    "new_payee_count",
]

#: Number of accounts in each book.
TRUE_N_ACCOUNTS = 1600

#: Fraction of each book that is a planted anomaly.
TRUE_ANOMALY_FRACTION = 0.03

#: Two independent standard-normal latent factors drive every genuine
#: account, in both books. This is the number a scree plot is trying to
#: recover, and the number Part 4 checks your chosen number of components
#: against.
TRUE_N_LATENT_FACTORS = 2
TRUE_LATENT_NAMES = ["cash_flow_scale", "activity_level"]

#: Each feature's mean, its loading on (cash_flow_scale, activity_level),
#: and the standard deviation of its own independent noise. A feature with
#: a large loading on a factor is strongly explained by it; a feature with
#: loadings near zero on both is close to pure noise (none of the six here
#: are — every column carries genuine signal from at least one factor).
TRUE_FEATURE_MODEL = {
    #                    mean,   (loading on cash_flow_scale, loading on activity_level), noise_std
    "monthly_inflow_eur":        (2200.0, (900.0,   0.0), 120.0),
    "monthly_outflow_eur":       (1900.0, (850.0,   0.0), 110.0),
    "transaction_count":         (38.0,   (2.0,    11.0), 3.0),
    "avg_transaction_size_eur":  (46.0,   (9.0,    -5.0), 3.5),
    "night_transaction_ratio":   (0.09,   (-0.015,  0.045), 0.02),
    "new_payee_count":           (3.0,    (0.0,     1.4), 0.9),
}


def _make_genuine(n, rng):
    """``n`` genuine accounts, both latent factors standard-normal."""
    z1 = rng.normal(0, 1, n)  # cash_flow_scale
    z2 = rng.normal(0, 1, n)  # activity_level

    columns = {}
    for name, (mean, (load1, load2), noise_std) in TRUE_FEATURE_MODEL.items():
        signal = mean + load1 * z1 + load2 * z2
        columns[name] = signal + rng.normal(0, noise_std, n)

    df = pd.DataFrame(columns)
    df["night_transaction_ratio"] = df["night_transaction_ratio"].clip(0, 1)
    df["new_payee_count"] = df["new_payee_count"].clip(lower=0)
    df["transaction_count"] = df["transaction_count"].clip(lower=1)
    df["monthly_inflow_eur"] = df["monthly_inflow_eur"].clip(lower=50.0)
    df["monthly_outflow_eur"] = df["monthly_outflow_eur"].clip(lower=50.0)
    df["avg_transaction_size_eur"] = df["avg_transaction_size_eur"].clip(lower=5.0)
    return df


# ---------------------------------------------------------------------------
# Retail book — spike anomalies
# ---------------------------------------------------------------------------

#: The two columns a credential-stuffing takeover disturbs. The other four
#: are left at the genuine generating process.
TRUE_RETAIL_SPIKE_FEATURES = ("night_transaction_ratio", "new_payee_count")

#: (mean, std) each spiked column is redrawn from, replacing the genuine
#: value. Both are far outside the genuine range: genuine
#: night_transaction_ratio has mean 0.09 and clips at [0, 1]; genuine
#: new_payee_count has mean 3.0. A spiked account's two probed columns sit
#: roughly 6-10 genuine standard deviations out (Part 4 asks you to compute
#: the exact figure from the genuine accounts' own mean and standard
#: deviation, not from this comment).
TRUE_RETAIL_SPIKE_PARAMS = {
    "night_transaction_ratio": (0.62, 0.08),
    "new_payee_count": (14.0, 2.5),
}


def load_retail_accounts(n_samples=TRUE_N_ACCOUNTS, anomaly_fraction=TRUE_ANOMALY_FRACTION,
                          random_state=8301):
    """Retail current accounts, with credential-stuffing takeovers planted.

    Returns a DataFrame of the six ``FEATURES`` plus a boolean
    ``is_anomaly`` column. The anomalies move only two of the six features,
    a long way outside the genuine range; the other four are indistinguishable
    from a genuine account's.
    """
    rng = np.random.default_rng(random_state)
    n_anomaly = int(round(n_samples * anomaly_fraction))
    n_genuine = n_samples - n_anomaly

    genuine_df = _make_genuine(n_genuine, rng)
    genuine_df["is_anomaly"] = False

    # Anomalous accounts keep the genuine generating process for the four
    # untouched columns, and have the two probed columns overwritten.
    anomaly_df = _make_genuine(n_anomaly, rng)
    for feature, (mean, std) in TRUE_RETAIL_SPIKE_PARAMS.items():
        anomaly_df[feature] = rng.normal(mean, std, n_anomaly)
    anomaly_df["night_transaction_ratio"] = anomaly_df["night_transaction_ratio"].clip(0, 1)
    anomaly_df["is_anomaly"] = True

    df = pd.concat([genuine_df, anomaly_df], ignore_index=True)
    df[FEATURES] = df[FEATURES].round(3)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Business book — combination anomalies
# ---------------------------------------------------------------------------


def load_business_accounts(n_samples=TRUE_N_ACCOUNTS, anomaly_fraction=TRUE_ANOMALY_FRACTION,
                            random_state=8302):
    """Business current accounts, with mule accounts planted.

    Returns a DataFrame of the six ``FEATURES`` plus a boolean
    ``is_anomaly`` column. Each anomalous account's six values are drawn
    **independently**, one column at a time, from the interquartile-ish
    (5th-95th percentile) range genuine business accounts occupy for that
    column — so no single column is extreme, but the relationship between
    columns that every genuine account follows is broken.
    """
    rng = np.random.default_rng(random_state)
    n_anomaly = int(round(n_samples * anomaly_fraction))
    n_genuine = n_samples - n_anomaly

    genuine_df = _make_genuine(n_genuine, rng)
    genuine_df["is_anomaly"] = False

    anomaly_df = pd.DataFrame({
        feature: rng.uniform(
            genuine_df[feature].quantile(0.05),
            genuine_df[feature].quantile(0.95),
            n_anomaly,
        )
        for feature in FEATURES
    })
    anomaly_df["is_anomaly"] = True

    df = pd.concat([genuine_df, anomaly_df], ignore_index=True)
    df[FEATURES] = df[FEATURES].round(3)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retail = load_retail_accounts()
    business = load_business_accounts()
    for name, df in (("retail", retail), ("business", business)):
        n_anom = int(df["is_anomaly"].sum())
        print(f"{name}: {len(df)} accounts, {n_anom} anomalies ({n_anom / len(df):.1%})")
        print(df[FEATURES].describe().loc[["mean", "std", "min", "max"]].round(2))
        print()

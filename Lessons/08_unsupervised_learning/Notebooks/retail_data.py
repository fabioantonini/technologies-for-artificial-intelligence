"""Synthetic data for Lesson 8 — Unsupervised learning.

One narrative runs through all three notebooks: Aurora, a fictional online
retailer, wants to understand its customers without ever showing a model a
label, because none exists. Every function below returns data together with
the ``TRUE_*`` constants that generated it, so a clustering or a
dimensionality reduction can be checked against a ground truth that is
normally never available for unsupervised problems.

No network access, no API keys — every array is generated locally with numpy.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Customer segments — Notebook 01, k-means
# ---------------------------------------------------------------------------

TRUE_N_SEGMENTS = 4

TRUE_SEGMENT_NAMES = [
    "window shoppers",
    "bargain hunters",
    "occasional big spenders",
    "loyal high spenders",
]

# (annual_spend_eur mean, visits_per_month mean, n_customers)
TRUE_SEGMENT_PARAMS = [
    (80.0, 1.4, 500),
    (260.0, 6.2, 500),
    (900.0, 1.3, 400),
    (960.0, 5.6, 600),
]

# Standard deviation of spend and visit frequency, shared across segments
# except that spend variability scales with the segment's own mean (bigger
# spenders vary more in euro terms) — this is what keeps the two big-spend
# segments from being trivially separable on the spend axis alone.
TRUE_SPEND_STD_FRACTION = 0.16
TRUE_VISIT_STD = 0.85


def make_customer_segments(seed=8001):
    """2,000 Aurora customers, two features: annual spend and visit frequency.

    Each of the four segments is an independent Gaussian blob. Segment
    identity is returned only as ``true_segment`` for later validation — no
    clustering method below is allowed to see it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for label, (spend_mean, visit_mean, n) in zip(TRUE_SEGMENT_NAMES, TRUE_SEGMENT_PARAMS):
        spend = rng.normal(spend_mean, spend_mean * TRUE_SPEND_STD_FRACTION, n)
        visits = rng.normal(visit_mean, TRUE_VISIT_STD, n)
        rows.append(pd.DataFrame({
            "annual_spend_eur": spend,
            "visits_per_month": visits,
            "true_segment": label,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["annual_spend_eur"] = df["annual_spend_eur"].clip(lower=5.0)
    df["visits_per_month"] = df["visits_per_month"].clip(lower=0.1)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Session data with a bot core — Notebook 02, hierarchical clustering & DBSCAN
# ---------------------------------------------------------------------------

TRUE_BOT_FRACTION = 0.12
TRUE_N_SESSIONS = 1500

# Bots: a tight, low-variance cluster — automated scripts behave the same
# way every time.
TRUE_BOT_CENTER = (4.0, 12.0)     # (session_duration_min, pages_viewed)
TRUE_BOT_STD = (0.30, 0.55)

# Genuine users: centred on the *same* mean duration and page count — so the
# two groups are indistinguishable on either axis' mean alone — but spread
# into a ring around it, because human browsing is far more variable.
TRUE_RING_CENTER = TRUE_BOT_CENTER
TRUE_RING_RADIUS = (7.0, 1.6)     # (mean radius, std of radius)


def make_bot_sessions(seed=8002):
    """1,500 weekly session summaries: a dense bot core inside a ring of humans.

    Built specifically so that a k-means model with k=2, which assumes
    convex, roughly equal-variance clusters, cannot recover the two groups:
    the honest split is "close to the centre" versus "far from it", not a
    linear boundary.
    """
    rng = np.random.default_rng(seed)
    n_bots = int(round(TRUE_N_SESSIONS * TRUE_BOT_FRACTION))
    n_humans = TRUE_N_SESSIONS - n_bots

    bot_duration = rng.normal(TRUE_BOT_CENTER[0], TRUE_BOT_STD[0], n_bots)
    bot_pages = rng.normal(TRUE_BOT_CENTER[1], TRUE_BOT_STD[1], n_bots)

    angle = rng.uniform(0, 2 * np.pi, n_humans)
    radius = rng.normal(TRUE_RING_RADIUS[0], TRUE_RING_RADIUS[1], n_humans)
    radius = np.clip(radius, 1.5, None)
    human_duration = TRUE_RING_CENTER[0] + radius * np.cos(angle) * 0.45
    human_pages = TRUE_RING_CENTER[1] + radius * np.sin(angle) * 3.6

    df = pd.DataFrame({
        "session_duration_min": np.concatenate([bot_duration, human_duration]),
        "pages_viewed": np.concatenate([bot_pages, human_pages]),
        "true_group": ["bot"] * n_bots + ["human"] * n_humans,
    })
    df["session_duration_min"] = df["session_duration_min"].clip(lower=0.2)
    df["pages_viewed"] = df["pages_viewed"].clip(lower=1.0)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. High-dimensional customer table — Notebook 03, PCA, anomaly detection, t-SNE
# ---------------------------------------------------------------------------

TRUE_N_LATENT_FACTORS = 3
TRUE_N_CUSTOMERS = 2000
TRUE_ANOMALY_FRACTION = 0.02  # fraction of accounts replaced by account-takeover pattern

TRUE_LATENT_NAMES = ["spending_propensity", "engagement", "price_sensitivity"]


def make_customer_features(seed=8003):
    """2,000 Aurora accounts, 8 observed features driven by 3 latent factors.

    Every observed column is a linear combination of three independent
    standard-normal latent factors plus its own noise term — so the honest
    dimensionality of this table is 3, not 8, and PCA is being asked to
    rediscover a number that is known here and nowhere else in the course.

    A final ``TRUE_ANOMALY_FRACTION`` of accounts are overwritten with an
    account-takeover pattern: independent, unstructured feature draws that
    ignore the latent factors entirely. Individually each value sits inside
    the normal range for *some* legitimate customer — the anomaly is in the
    combination, not any single column.
    """
    rng = np.random.default_rng(seed)
    n = TRUE_N_CUSTOMERS
    n_anomaly = int(round(n * TRUE_ANOMALY_FRACTION))
    n_normal = n - n_anomaly

    z1 = rng.normal(0, 1, n_normal)  # spending propensity
    z2 = rng.normal(0, 1, n_normal)  # engagement
    z3 = rng.normal(0, 1, n_normal)  # price sensitivity

    def noisy(signal, noise_std):
        return signal + rng.normal(0, noise_std, n_normal)

    annual_spend_eur = noisy(500 + 300 * z1, 30)
    avg_basket_value_eur = noisy(60 + 20 * z1 - 10 * z3, 4)
    purchase_frequency = noisy(5 + 2.0 * z2, 0.5)
    discount_usage_rate = np.clip(noisy(0.30 + 0.15 * z3 - 0.05 * z1, 0.03), 0, 1)
    days_since_last_purchase = np.clip(noisy(30 - 5 * z2, 2.5), 0, None)
    weekend_purchase_ratio = np.clip(noisy(0.40 + 0.10 * z2, 0.03), 0, 1)
    mobile_session_count = np.clip(noisy(10 + 4 * z2 + 2 * z1, 1.3), 0, None)
    return_rate = np.clip(noisy(0.05 + 0.03 * z3, 0.012), 0, 1)

    normal_df = pd.DataFrame({
        "annual_spend_eur": annual_spend_eur,
        "avg_basket_value_eur": avg_basket_value_eur,
        "purchase_frequency": purchase_frequency,
        "discount_usage_rate": discount_usage_rate,
        "days_since_last_purchase": days_since_last_purchase,
        "weekend_purchase_ratio": weekend_purchase_ratio,
        "mobile_session_count": mobile_session_count,
        "return_rate": return_rate,
    })
    normal_df["is_anomaly"] = False

    # Account takeover: each column drawn independently from the *marginal*
    # range seen across legitimate accounts, so no single feature is extreme,
    # but the combination violates the covariance structure every genuine
    # account follows.
    anomaly_df = pd.DataFrame({
        col: rng.uniform(normal_df[col].quantile(0.05), normal_df[col].quantile(0.95), n_anomaly)
        for col in normal_df.columns if col != "is_anomaly"
    })
    anomaly_df["is_anomaly"] = True

    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

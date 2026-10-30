#!/usr/bin/env python3
"""Recompute every number the lesson 8 handout works out by hand.

The rule this file lives by: **start from the raw inputs, never from the
handout's own intermediate values.** So the six-customer example is redone with
plain loops, the silhouette and the adjusted Rand index are recomputed from
their definitions and compared with scikit-learn, the merge heights are rebuilt
from Ward's formula, PCA is reached by three routes (the angle formula, the
eigenvectors, the singular value decomposition), and the reconstruction-error
detector is rebuilt from the decomposition rather than from `PCA`. Claims that
could be one lucky draw are rechecked at another seed.

Run it directly, or let `tools/verify_lesson.py 08` discover it.
"""

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))

from retail_data import (                                        # noqa: E402
    make_customer_segments, make_bot_sessions, make_customer_features,
    TRUE_N_SESSIONS, TRUE_BOT_FRACTION, TRUE_ANOMALY_FRACTION, TRUE_N_CUSTOMERS,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples  # noqa: E402
from sklearn.neighbors import NearestNeighbors                   # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402
from sklearn.decomposition import PCA                            # noqa: E402
from scipy.cluster.hierarchy import linkage                      # noqa: E402

CHECKS = 0
FAILURES = []


def check(label, computed, expected, tol, note=""):
    """Assert one handout number, and say how it was reached."""
    global CHECKS
    CHECKS += 1
    ok = abs(computed - expected) <= tol
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {label:<62s} {computed:>12.6f}  handout {expected:<10.6g}"
          f"{'  ' + note if note else ''}")
    if not ok:
        FAILURES.append(f"{label}: computed {computed!r}, handout says {expected!r}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def sq(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def mean_point(points):
    return tuple(sum(c) / len(points) for c in zip(*points))


def ari_by_pairs(a, b):
    """Adjusted Rand index from its definition: pair counts, then the chance correction."""
    a, b = np.asarray(a), np.asarray(b)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    together_both = sum(comb(int(n), 2) for n in table.ravel())
    together_a = sum(comb(int(n), 2) for n in table.sum(axis=1))
    together_b = sum(comb(int(n), 2) for n in table.sum(axis=0))
    pairs = comb(len(a), 2)
    expected = together_a * together_b / pairs
    return (together_both - expected) / ((together_a + together_b) / 2 - expected)


# =====================================================================
section("2.3 Six customers by hand")
A, B, C, D, E, F = (1, 1), (1, 2), (2, 1), (8, 6), (9, 6), (8, 7)
pts = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

c1, c2 = A, B
check("C to A, squared", sq(C, c1), 1, 0)
check("C to B, squared", sq(C, c2), 2, 0)
check("D to B, squared", sq(D, c2), 65, 0)
check("D to A, squared", sq(D, c1), 74, 0)
check("E to B, squared", sq(E, c2), 80, 0)
check("F to B, squared", sq(F, c2), 74, 0)
groups = {n: (0 if sq(p, c1) <= sq(p, c2) else 1) for n, p in pts.items()}
check("round 1 assign gives AC / BDEF", float("".join(n for n in pts if groups[n] == 0) == "AC"), 1, 0)
wcss = lambda cs: sum(sq(p, cs[groups[n]]) for n, p in pts.items())
check("round 1 assign, WCSS", wcss((c1, c2)), 220, 0)
c1 = mean_point([pts[n] for n in pts if groups[n] == 0])
c2 = mean_point([pts[n] for n in pts if groups[n] == 1])
check("round 1 move, AC centre x", c1[0], 1.5, 1e-12)
check("round 1 move, BDEF centre x", c2[0], 6.5, 1e-12)
check("round 1 move, BDEF centre y", c2[1], 5.25, 1e-12)
check("round 1 move, B's line", sq(B, c2), 40.8125, 1e-12)
check("round 1 move, D's line", sq(D, c2), 2.8125, 1e-12)
check("round 1 move, E's line", sq(E, c2), 6.8125, 1e-12)
check("round 1 move, F's line", sq(F, c2), 5.3125, 1e-12)
check("round 1 move, WCSS", wcss((c1, c2)), 56.25, 1e-12)
check("round 2, B to the first centre", sq(B, c1), 1.25, 1e-12)
groups = {n: (0 if sq(p, c1) <= sq(p, c2) else 1) for n, p in pts.items()}
check("round 2 assign, WCSS", wcss((c1, c2)), 16.6875, 1e-12)
c1 = mean_point([pts[n] for n in pts if groups[n] == 0])
c2 = mean_point([pts[n] for n in pts if groups[n] == 1])
check("round 2 move, ABC centre", c1[0], 4 / 3, 1e-12)
check("round 2 move, DEF centre x", c2[0], 25 / 3, 1e-12)
check("round 2 move, DEF centre y", c2[1], 19 / 3, 1e-12)
check("round 2 move, WCSS = 8/3", wcss((c1, c2)), 2.6667, 5e-5)
again = {n: (0 if sq(p, c1) <= sq(p, c2) else 1) for n, p in pts.items()}
check("round 3 changes nothing", float(again == groups), 1, 0)

section("2.5 Silhouette of customer B")
dist = lambda p, q: np.sqrt(sq(p, q))
a = (dist(B, A) + dist(B, C)) / 2
b = (dist(B, D) + dist(B, E) + dist(B, F)) / 3
check("sqrt 2", dist(B, C), 1.414, 5e-4)
check("sqrt 65", dist(B, D), 8.062, 5e-4)
check("sqrt 80", dist(B, E), 8.944, 5e-4)
check("sqrt 74", dist(B, F), 8.602, 5e-4)
check("a", a, 1.207, 5e-4)
check("b", b, 8.536, 5e-4)
check("s = (b - a) / max(a, b)", (b - a) / max(a, b), 0.859, 5e-4)
six = np.array(list(pts.values()), dtype=float)
check("same s from scikit-learn silhouette_samples", silhouette_samples(six, [0, 0, 0, 1, 1, 1])[1], 0.859, 5e-4)

# =====================================================================
section("2.4-2.6 The 2,000 customers")
customers = make_customer_segments()
Xs = StandardScaler().fit_transform(customers[["annual_spend_eur", "visits_per_month"]].values)


def lloyd(X, centres):
    """Another implementation of the two moves, written independently of the notebook."""
    while True:
        labels = np.linalg.norm(X[:, None] - centres[None], axis=2).argmin(axis=1)
        new = np.vstack([X[labels == j].mean(axis=0) for j in range(len(centres))])
        if np.allclose(new, centres):
            return labels, ((X - new[labels]) ** 2).sum()
        centres = new


finals = np.array([lloyd(Xs, Xs[np.random.default_rng(s).choice(len(Xs), 4, replace=False)])[1]
                   for s in range(30)])
check("best WCSS over 30 random starts", finals.min(), 374.1, 0.05)
check("starts reaching it", np.isclose(finals, finals.min(), atol=0.05).sum(), 29, 0)
check("worst WCSS over 30 random starts", finals.max(), 1390.4, 0.05)
check("worst / best, 'more than three times'", float(finals.max() / finals.min() > 3), 1, 0)
best_sk = KMeans(n_clusters=4, n_init=10, random_state=7).fit(Xs)   # another seed than the notebook
check("scikit-learn KMeans, another random_state", best_sk.inertia_, 374.0919, 5e-5)
check("silhouette at k = 4", silhouette_score(Xs, best_sk.labels_), 0.690, 5e-4)
sil = {k: silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=7).fit(Xs).labels_)
       for k in range(2, 9)}
check("silhouette is highest at k = 4", max(sil, key=sil.get), 4, 0)
check("ARI against the truth, by pair counting", ari_by_pairs(customers["true_segment"], best_sk.labels_), 0.985, 5e-4)
check("ARI against the truth, scikit-learn", adjusted_rand_score(customers["true_segment"], best_sk.labels_), 0.985, 5e-4)

# =====================================================================
section("3. The sessions")
sessions = make_bot_sessions()
S = StandardScaler().fit_transform(sessions[["session_duration_min", "pages_viewed"]].values)
bot = (sessions["true_group"] == "bot").values
check("bots", bot.sum(), 180, 0)
check("12% of 1,500", TRUE_BOT_FRACTION * TRUE_N_SESSIONS, 180, 1e-9)
km = KMeans(n_clusters=2, n_init=10, random_state=3).fit(S)
check("k-means k = 2, ARI (pair counting)", ari_by_pairs(bot, km.labels_), -0.046, 5e-4)
bot_cluster = km.labels_[bot][0]
check("k-means: every bot in one cluster", float((km.labels_[bot] == bot_cluster).all()), 1, 0)
check("k-means: humans sharing the bots' cluster", ((km.labels_ == bot_cluster) & ~bot).sum(), 780, 0)
ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(S)
check("Ward cut at 2, ARI (pair counting)", ari_by_pairs(bot, ward.labels_), -0.062, 5e-4)
for name in ("complete", "average", "single"):
    other = AgglomerativeClustering(n_clusters=2, linkage=name).fit(S)
    check(f"{name} linkage does not separate the bots either (ARI < 0.1)",
          float(adjusted_rand_score(bot, other.labels_) < 0.1), 1, 0)

section("3.3 The merge tree")
eight = np.array([[1.0, 1.0], [1.6, 1.3], [1.2, 2.3], [4.2, 4.0], [5.0, 4.4], [4.5, 5.3],
                  [8.2, 1.6], [8.8, 2.6]])


def ward_height(p, q):
    """Ward merge height of two groups, from their sizes and means (scipy's convention)."""
    na, nb = len(p), len(q)
    return np.sqrt(2 * na * nb / (na + nb) * sq(eight[p].mean(axis=0), eight[q].mean(axis=0)))


check("A + B, as the plain distance sqrt(0.45)", np.sqrt(0.6 ** 2 + 0.3 ** 2), 0.671, 5e-4)
check("A + B, Ward", ward_height([0], [1]), 0.671, 5e-4)
check("D + E, Ward", ward_height([3], [4]), 0.894, 5e-4)
check("G + H, Ward", ward_height([6], [7]), 1.166, 5e-4)
check("F + DE, Ward", ward_height([5], [3, 4]), 1.275, 5e-4)
check("C + AB, Ward", ward_height([2], [0, 1]), 1.333, 5e-4)
check("GH + DEF, Ward", ward_height([6, 7], [3, 4, 5]), 7.193, 5e-4)
check("the same heights from scipy", linkage(eight, "ward")[:6, 2].round(3).tolist()
      == [0.671, 0.894, 1.166, 1.275, 1.333, 7.193], 1, 0)

section("3.4 DBSCAN on twelve points, counted by hand")
twelve = np.array([[2.0, 2.0], [2.6, 2.3], [2.2, 2.9], [3.0, 2.8], [2.7, 3.5], [3.5, 3.4],
                   [1.6, 2.7], [3.3, 2.2], [4.4, 3.9], [5.3, 4.5], [6.2, 1.4], [1.0, 5.6]])
within = np.linalg.norm(twelve[:, None] - twelve[None], axis=2) <= 0.9
core = within.sum(axis=1) >= 4                       # itself included
border = ~core & (within & core[None, :]).any(axis=1)
noise = ~core & ~border
check("core points", core.sum(), 4, 0)
check("border points", border.sum(), 4, 0)
check("noise points", noise.sum(), 4, 0)
check("core points have four to six in their disc", float(set(within.sum(axis=1)[core]) <= {4, 5, 6}), 1, 0)
check("border points have three", float(set(within.sum(axis=1)[border]) == {3}), 1, 0)
toy = DBSCAN(eps=0.9, min_samples=4).fit(twelve)
check("scikit-learn agrees on the noise", float(((toy.labels_ == -1) == noise).all()), 1, 0)
check("one group", len(set(toy.labels_) - {-1}), 1, 0)

section("3.5-3.7 DBSCAN on the sessions")
kth = np.sort(np.linalg.norm(S[:, None] - S[None], axis=2), axis=1)[:, 9]   # 10th, itself first
check("sessions with 10th neighbour beyond 0.30", (kth > 0.30).sum(), 65, 0)
db = DBSCAN(eps=0.30, min_samples=10).fit(S)
check("DBSCAN ARI (pair counting)", ari_by_pairs(bot, db.labels_), 0.941, 5e-4)
bot_group = db.labels_[bot][0]
check("all 180 bots in one group", (db.labels_[bot] == bot_group).sum(), 180, 0)
check("humans in the other group", ((db.labels_ != bot_group) & (db.labels_ != -1) & ~bot).sum(), 1303, 0)
check("humans of 1,320", (~bot).sum(), 1320, 0)
check("humans placed with the bots", ((db.labels_ == bot_group) & ~bot).sum(), 4, 0)
check("noise sessions", (db.labels_ == -1).sum(), 13, 0)
check("noise sessions that are humans", ((db.labels_ == -1) & ~bot).sum(), 13, 0)
radius = np.linalg.norm(S - S[bot].mean(axis=0), axis=1)
outer_start = np.percentile(radius[~bot], 95)
check("noise in the gap: beyond the core, short of the ring",
      ((db.labels_ == -1) & (radius > radius[bot].max()) & (radius < np.percentile(radius[~bot], 5))).sum(), 2, 0)
check("noise on the outer edge (beyond 2.1)", ((db.labels_ == -1) & (radius > 2.1)).sum(), 11, 0)
kept = db.labels_ != -1
check("silhouette, k-means k = 2", silhouette_score(S, km.labels_), 0.401, 5e-4)
check("silhouette, DBSCAN", silhouette_score(S[kept], db.labels_[kept]), -0.106, 5e-4)
per_point = silhouette_samples(S[kept], db.labels_[kept])
ring = db.labels_[kept] != bot_group
check("ring points with a negative silhouette, %", 100 * (per_point[ring] < 0).mean(), 99.2, 0.05)

# =====================================================================
section("4. PCA")
accounts = make_customer_features()
cols = [c for c in accounts.columns if c != "is_anomaly"]
Z = StandardScaler().fit_transform(accounts[cols].values)
anomaly = accounts["is_anomaly"].values
check("anomalies, 2% of 2,000", TRUE_ANOMALY_FRACTION * TRUE_N_CUSTOMERS, 40, 1e-9)
check("anomalies in the table", anomaly.sum(), 40, 0)
pair = Z[:, :2]
r = (pair[:, 0] * pair[:, 1]).mean()                 # standardised: the mean product is r
check("r, spend and basket value", r, 0.859, 5e-4)
theta = np.radians(np.arange(0, 180.5, 0.5))
formula = 1 + r * np.sin(2 * theta)
check("1 + r sin 2theta is widest at 45 degrees", np.degrees(theta[formula.argmax()]), 45, 0)
check("1 + r sin 2theta is narrowest at 135 degrees", np.degrees(theta[formula.argmin()]), 135, 0)
measured = [(pair @ np.array([np.cos(t), np.sin(t)])).var() for t in (0, np.pi / 4, 3 * np.pi / 4)]
check("variance on the spend axis, measured", measured[0], 1.0, 1e-9)
check("variance at 45 degrees, measured = 1 + r", measured[1], 1.859, 5e-4)
check("variance at 135 degrees, measured = 1 - r", measured[2], 0.141, 5e-4)
values = np.linalg.eigvalsh(np.array([[1, r], [r, 1]]))[::-1]
check("eigenvalue 1 of [[1, r], [r, 1]]", values[0], 1.859, 5e-4)
check("eigenvalue 2 of [[1, r], [r, 1]]", values[1], 0.141, 5e-4)
singular = np.linalg.svd(pair, compute_uv=False)
check("first share, from the SVD", singular[0] ** 2 / (singular ** 2).sum(), 0.9297, 5e-5)
check("first share, 1.859 / 2, as a percentage", 100 * (1 + r) / 2, 92.97, 5e-3)
check("second share, from the SVD", singular[1] ** 2 / (singular ** 2).sum(), 0.0703, 5e-5)
s8 = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False) ** 2
share = s8 / s8.sum()
check("first three components, % (SVD)", 100 * share[:3].sum(), 93.5, 0.05)
check("the fourth, % (SVD)", 100 * share[3], 2.4, 0.05)

# =====================================================================
section("5. Anomalies")
genuine = accounts.loc[~anomaly, cols]
fraud = accounts.loc[anomaly, cols]
inside = ((fraud >= genuine.quantile(0.05)) & (fraud <= genuine.quantile(0.95))).values.mean()
check("anomaly values inside the genuine 5th-95th range", inside, 1.0, 0)

# 5.2 two columns, rebuilt from the diagonal directly rather than from PCA
centred = pair - pair.mean(axis=0)
u = np.array([1, 1]) / np.sqrt(2)
shadow = centred @ u
error2 = ((centred - np.outer(shadow, u)) ** 2).sum(axis=1)
g = np.flatnonzero(~anomaly)[np.abs(shadow[~anomaly]).argmax()]
f = np.flatnonzero(anomaly)[error2[anomaly].argmax()]
check("genuine account, distance from the middle", np.linalg.norm(pair[g]), 4.70, 5e-3)
check("genuine account, reconstruction error", error2[g], 0.01, 5e-3)
check("anomaly, distance from the middle", np.linalg.norm(pair[f]), 1.96, 5e-3)
check("anomaly, reconstruction error", error2[f], 3.75, 5e-3)


def detector(table):
    """Reconstruction error from three components, via the SVD, and the 2% flagged."""
    X = StandardScaler().fit_transform(table[cols].values)
    centred = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    kept = vt[:3]
    rebuilt = centred @ kept.T @ kept
    e = ((centred - rebuilt) ** 2).sum(axis=1)
    return e, e > np.quantile(e, 0.98)


e, flagged = detector(accounts)
check("flagged accounts", flagged.sum(), 40, 0)
check("37 of 40 caught", (flagged & anomaly).sum(), 37, 0)
check("mean error, anomalies", e[anomaly].mean(), 4.676, 5e-4)
check("mean error, genuine", e[~anomaly].mean(), 0.431, 5e-4)
caught = []
for seed in (8003, 1, 2, 3, 4, 5, 6, 7):
    table = make_customer_features(seed=seed)
    _, hit = detector(table)
    caught.append(int((hit & table["is_anomaly"].values).sum()))
check("lowest count over eight batches", min(caught), 32, 0)
check("highest count over eight batches", max(caught), 38, 0)
check("median count over eight batches", float(np.median(caught)), 36, 0)

section("5.4 The 2-D pictures")
p2 = PCA(n_components=2).fit_transform(Z)
middle = p2[~anomaly].mean(axis=0)
check("PCA picture: anomalies from the middle", np.linalg.norm(p2[anomaly] - middle, axis=1).mean(), 1.30, 5e-3)
check("PCA picture: genuine from the middle", np.linalg.norm(p2[~anomaly] - middle, axis=1).mean(), 2.20, 5e-3)
for seed in (1, 2):                                   # not one lucky batch
    table = make_customer_features(seed=seed)
    lab = table["is_anomaly"].values
    q = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(table[cols].values))
    mid = q[~lab].mean(axis=0)
    check(f"PCA picture, batch {seed}: anomalies closer to the middle",
          float(np.linalg.norm(q[lab] - mid, axis=1).mean() < np.linalg.norm(q[~lab] - mid, axis=1).mean()), 1, 0)
from sklearn.manifold import TSNE                                # noqa: E402
t2 = TSNE(n_components=2, perplexity=30, random_state=1, init="pca").fit_transform(Z)  # another seed
mid = t2[~anomaly].mean(axis=0)
check("t-SNE, another seed: anomalies closer to the middle",
      float(np.linalg.norm(t2[anomaly] - mid, axis=1).mean() < np.linalg.norm(t2[~anomaly] - mid, axis=1).mean()), 1, 0)

# =====================================================================
section("Result")
print(f"\n  {CHECKS} hand-worked numbers recomputed from the raw inputs.")
if FAILURES:
    print(f"  {len(FAILURES)} DISAGREE with the handout:\n")
    for failure in FAILURES:
        print(f"    - {failure}")
    sys.exit(1)
print("  All agree with the handout.\n")

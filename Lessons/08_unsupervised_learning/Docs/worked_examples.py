"""Recompute every number lesson 8 works out by hand.

    python Lessons/08_unsupervised_learning/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. Every check reaches the handout's figure
from ``retail_data.py``'s generators directly, or from first principles
(the 2x2 eigenvalue formula, the SVD identity), never from the handout's
own intermediate numbers.

Tolerances are tighter for pure arithmetic and closed-form identities, and
looser for anything that depends on scikit-learn's internals (KMeans'
random restarts, DBSCAN, t-SNE), since this script may run under a
different scikit-learn version than the teaching container.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))
from retail_data import (
    make_customer_segments, make_bot_sessions, make_customer_features,
    TRUE_N_SEGMENTS, TRUE_N_LATENT_FACTORS, TRUE_BOT_FRACTION, TRUE_N_SESSIONS,
)

HANDOUT = "Lessons/08_unsupervised_learning/Docs/unsupervised_learning.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


def at_least(name: str, computed, floor) -> None:
    global checks
    checks += 1
    if computed < floor:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    expected at least {floor}, recomputing gives {computed}")


def at_most(name: str, computed, ceiling) -> None:
    global checks
    checks += 1
    if computed > ceiling:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    expected at most {ceiling}, recomputing gives {computed}")


# ============================================================ Section 1.1

segments = make_customer_segments()
sessions = make_bot_sessions()
accounts = make_customer_features()

same("1.1 number of customers", len(segments), 2000, tolerance=0)
same("1.1 number of true segments", TRUE_N_SEGMENTS, 4, tolerance=0)
same("1.1 number of sessions", len(sessions), TRUE_N_SESSIONS, tolerance=0)
same("1.1 bot fraction", TRUE_BOT_FRACTION, 0.12, tolerance=0)
n_bots_expected = int(round(TRUE_N_SESSIONS * TRUE_BOT_FRACTION))
same("1.1 number of bot sessions", (sessions["true_group"] == "bot").sum(),
     n_bots_expected, tolerance=0)
same("1.1 number of accounts", len(accounts), 2000, tolerance=0)
same("1.1 number of latent factors", TRUE_N_LATENT_FACTORS, 3, tolerance=0)
same("1.1 number of planted anomalies", accounts["is_anomaly"].sum(), 40, tolerance=0)

# ============================================================ Section 2.1
# The 6-point Lloyd's-algorithm worked example, recomputed independently.

X_seg = segments[["annual_spend_eur", "visits_per_month"]].values
Xs_seg = StandardScaler().fit_transform(X_seg)
scaler_seg = StandardScaler().fit(X_seg)
pts = Xs_seg[:6]


def assign(pts, centers):
    d = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(d, axis=1)


def wcss(pts, labels, centers):
    return float(np.sum((pts - centers[labels]) ** 2))


centers = np.array([pts[0], pts[1]])
expected_wcss = [(15.965, 7.281), (5.491, 4.674), (4.674, 4.674)]
for it, (assign_expected, update_expected) in enumerate(expected_wcss):
    labels = assign(pts, centers)
    w_assign = wcss(pts, labels, centers)
    same(f"2.1 iteration {it} WCSS after assign step", w_assign, assign_expected, tolerance=5e-3)
    new_centers = np.array([
        pts[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
        for j in range(2)
    ])
    w_update = wcss(pts, labels, new_centers)
    same(f"2.1 iteration {it} WCSS after update step", w_update, update_expected, tolerance=5e-3)
    centers = new_centers

if not (assign(pts, centers) == labels).all():
    raise SystemExit(f"{HANDOUT}: 2.1 the algorithm should have converged by iteration 2")
checks += 1

# ============================================================ Section 2.2
# Naive-random vs k-means++ initialisation, a second sweep independent of
# the notebook's (same recipe, run again here from raw data).


def kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(X[rng.choice(n, p=d2 / d2.sum())])
    return np.array(centers)


def lloyd(X, centers0, max_iter=200, tol=1e-8):
    centers = centers0.copy()
    for _ in range(max_iter):
        labels = assign(X, centers)
        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(len(centers))
        ])
        if np.sum((new_centers - centers) ** 2) < tol:
            centers = new_centers
            break
        centers = new_centers
    labels = assign(X, centers)
    return wcss(X, labels, centers)


naive_results, pp_results = [], []
for seed in range(30):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs_seg), TRUE_N_SEGMENTS, replace=False)
    naive_results.append(lloyd(Xs_seg, Xs_seg[idx]))
    rng = np.random.default_rng(seed)
    pp_results.append(lloyd(Xs_seg, kmeans_pp_init(Xs_seg, TRUE_N_SEGMENTS, rng)))

naive_results, pp_results = np.array(naive_results), np.array(pp_results)
same("2.2 best of 30 naive starts", naive_results.min(), 374.1, tolerance=1.0)
same("2.2 worst of 30 naive starts", naive_results.max(), 1390.4, tolerance=5.0)
same("2.2 best of 30 k-means++ starts", pp_results.min(), 374.1, tolerance=1.0)
same("2.2 worst of 30 k-means++ starts", pp_results.max(), 1088.3, tolerance=15.0)
if not pp_results.max() < naive_results.max():
    raise SystemExit(f"{HANDOUT}: 2.2 k-means++'s worst case should beat naive's worst case")
checks += 1

# ============================================================ Section 2.3
# The elbow / silhouette table and the final k=4 ARI.

wcss_curve, sil_curve = [], []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs_seg)
    wcss_curve.append(km.inertia_)
    sil_curve.append(silhouette_score(Xs_seg, km.labels_) if k > 1 else None)

expected_wcss_curve = [4000.0, 2128.5, 1125.8, 374.1, 320.6, 286.5, 244.5, 210.4]
expected_sil_curve = [None, 0.472, 0.564, 0.690, 0.593, 0.525, 0.528, 0.474]
for k, (w, w_exp, s, s_exp) in enumerate(
        zip(wcss_curve, expected_wcss_curve, sil_curve, expected_sil_curve), start=1):
    same(f"2.3 WCSS at k={k}", w, w_exp, tolerance=max(1.0, 0.01 * w_exp))
    if s is not None:
        same(f"2.3 silhouette at k={k}", s, s_exp, tolerance=0.01)

if not sil_curve.index(max(sil_curve[1:])) + 1 == 4:
    raise SystemExit(f"{HANDOUT}: 2.3 silhouette should peak at k=4")
checks += 1

final_km = KMeans(n_clusters=TRUE_N_SEGMENTS, n_init=10, random_state=0).fit(Xs_seg)
ari_segments = adjusted_rand_score(segments["true_segment"], final_km.labels_)
same("2.3 final ARI, customer segments", ari_segments, 0.985, tolerance=0.01)

# ============================================================ Section 3
# k-means, four linkage rules, and DBSCAN on the bot/human session data.

X_sess = sessions[["session_duration_min", "pages_viewed"]].values
Xs_sess = StandardScaler().fit_transform(X_sess)

km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs_sess)
ari_km2 = adjusted_rand_score(sessions["true_group"], km2.labels_)
same("3.1 k-means (k=2) ARI on session data", ari_km2, -0.046, tolerance=0.03)

expected_linkage_ari = {"ward": -0.062, "complete": -0.101, "average": -0.001, "single": -0.001}
for method, expected in expected_linkage_ari.items():
    agg = AgglomerativeClustering(n_clusters=2, linkage=method).fit(Xs_sess)
    ari = adjusted_rand_score(sessions["true_group"], agg.labels_)
    same(f"3.2 {method}-linkage ARI", ari, expected, tolerance=0.03)

db = DBSCAN(eps=0.30, min_samples=10).fit(Xs_sess)
ari_db = adjusted_rand_score(sessions["true_group"], db.labels_)
same("3.3 DBSCAN ARI", ari_db, 0.941, tolerance=0.02)
same("3.3 DBSCAN core points", len(db.core_sample_indices_), 1435, tolerance=15)
same("3.3 DBSCAN noise points", int((db.labels_ == -1).sum()), 13, tolerance=5)

noise_true = sessions.loc[db.labels_ == -1, "true_group"]
at_least("3.3 every noise point is human, not a misplaced bot",
         (noise_true == "human").mean(), 1.0)

# ============================================================ Section 5
# The 2x2 hand-solvable eigenvalue example, and the 8-feature PCA table.

cov2 = np.cov(Xs_seg, rowvar=False)
r = cov2[0, 1]
eigvals2, eigvecs2 = np.linalg.eigh(cov2)
order2 = np.argsort(eigvals2)[::-1]
eigvals2 = eigvals2[order2]

same("5.2 correlation r between spend and visit frequency", r, 0.171, tolerance=0.01)
# The 1+r / 1-r formula assumes a covariance matrix with an exact unit
# diagonal; numpy.cov's m-1 denominator leaves the diagonal near but not
# exactly 1, so the formula is an approximation, not an identity here.
same("5.2 largest eigenvalue near 1+r", eigvals2[0], 1 + r, tolerance=1e-3)
same("5.2 smallest eigenvalue near 1-r", eigvals2[1], 1 - r, tolerance=1e-3)
same("5.2 handout's rounded largest eigenvalue", eigvals2[0], 1.171, tolerance=0.01)
same("5.2 handout's rounded smallest eigenvalue", eigvals2[1], 0.829, tolerance=0.01)

feature_names = [c for c in accounts.columns if c != "is_anomaly"]
X_acc = accounts[feature_names].values
Xs_acc = StandardScaler().fit_transform(X_acc)
n_acc = len(Xs_acc)

cov8 = (Xs_acc.T @ Xs_acc) / (n_acc - 1)
eigvals8, eigvecs8 = np.linalg.eigh(cov8)
order8 = np.argsort(eigvals8)[::-1]
eigvals8 = eigvals8[order8]

Xc = Xs_acc - Xs_acc.mean(axis=0)
_, S, _ = np.linalg.svd(Xc, full_matrices=False)
eigvals_svd = (S ** 2) / (n_acc - 1)

sk_pca_full = PCA().fit(Xs_acc)

max_disagreement = max(
    np.max(np.abs(eigvals8 - eigvals_svd)),
    np.max(np.abs(eigvals8 - sk_pca_full.explained_variance_)),
)
at_most("5.3 eigh vs SVD vs sklearn PCA, largest disagreement", max_disagreement, 1e-8)

expected_eigvals8 = [3.539, 2.568, 1.380, 0.192, 0.104, 0.099, 0.077, 0.043]
for i, exp in enumerate(expected_eigvals8):
    same(f"5.4 PC{i+1} eigenvalue", eigvals8[i], exp, tolerance=0.01)

explained_ratio = eigvals8 / eigvals8.sum()
same("5.4 cumulative variance, first 3 components", explained_ratio[:3].sum(), 0.935, tolerance=0.005)
same("5.4 variance added by 4th component", explained_ratio[3], 0.024, tolerance=0.005)

# ============================================================ Section 6.1
# Reconstruction error and the fraud-detection recall/precision.

pca3 = PCA(n_components=TRUE_N_LATENT_FACTORS).fit(Xs_acc)
reconstructed = pca3.inverse_transform(pca3.transform(Xs_acc))
recon_error = np.sum((Xs_acc - reconstructed) ** 2, axis=1)
threshold = np.quantile(recon_error, 0.98)
flagged = recon_error > threshold
true_anomaly = accounts["is_anomaly"].values
caught = int(np.sum(flagged & true_anomaly))

same("6.1 reconstruction-error threshold", threshold, 2.019, tolerance=0.05)
same("6.1 accounts flagged", int(flagged.sum()), 40, tolerance=3)
same("6.1 anomalies caught", caught, 37, tolerance=3)
same("6.1 mean reconstruction error, anomalies", recon_error[true_anomaly].mean(), 4.676, tolerance=0.3)
same("6.1 mean reconstruction error, genuine accounts", recon_error[~true_anomaly].mean(), 0.431, tolerance=0.05)

# ============================================================ Section 6.2
# "Anomalies sit closer to the centre" — checked at the notebook's seed AND
# at two further seeds, since a claim that two quantities differ in a
# consistent direction needs re-running, not just computing once.


def centroid_distances(seed):
    df = make_customer_features(seed=seed)
    feats = [c for c in df.columns if c != "is_anomaly"]
    Xs = StandardScaler().fit_transform(df[feats].values)
    is_anom = df["is_anomaly"].values
    pca2 = PCA(n_components=2).fit_transform(Xs)
    centroid = pca2[~is_anom].mean(axis=0)
    d_anom = np.linalg.norm(pca2[is_anom] - centroid, axis=1).mean()
    d_genuine = np.linalg.norm(pca2[~is_anom] - centroid, axis=1).mean()
    return d_anom, d_genuine


d_anom_0, d_genuine_0 = centroid_distances(8003)
same("6.2 PCA-2D mean distance, anomalies (original seed)", d_anom_0, 1.30, tolerance=0.1)
same("6.2 PCA-2D mean distance, genuine (original seed)", d_genuine_0, 2.20, tolerance=0.1)

for other_seed in (9001, 12345):
    d_anom, d_genuine = centroid_distances(other_seed)
    if not d_anom < d_genuine:
        raise SystemExit(
            f"{HANDOUT}: 6.2 claim 'anomalies sit closer to the centre' failed "
            f"at seed {other_seed}: anomaly dist {d_anom:.3f} vs genuine dist {d_genuine:.3f}")
    checks += 1

# t-SNE ordering, checked once at the notebook's own seed (t-SNE is
# expensive and stochastic; the PCA check above already re-runs the
# generator at two further seeds for the underlying claim about the
# anomalies themselves).
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Xs_acc)
centroid_tsne = tsne_2d[~true_anomaly].mean(axis=0)
d_anom_tsne = np.linalg.norm(tsne_2d[true_anomaly] - centroid_tsne, axis=1).mean()
d_genuine_tsne = np.linalg.norm(tsne_2d[~true_anomaly] - centroid_tsne, axis=1).mean()
if not d_anom_tsne < d_genuine_tsne:
    raise SystemExit(
        f"{HANDOUT}: 6.2 t-SNE anomalies-closer-to-centre claim failed: "
        f"anomaly dist {d_anom_tsne:.2f} vs genuine dist {d_genuine_tsne:.2f}")
checks += 1


# ------------------------------ Section 3.4, whether eps transfers to a new week
#
# The section's claim is that eps = 0.30 does NOT transfer while the k-distance
# recipe does. Checking only that 0.30 works on the lesson's own week would
# confirm the sentence this section was written to replace, so both halves are
# checked: some week must fail under the constant, and every week must succeed
# under the recipe.


def _knee_eps(Z, min_samples=10):
    d = np.sort(NearestNeighbors(n_neighbors=min_samples)
                .fit(Z).kneighbors(Z)[0][:, -1])
    x = np.linspace(0, 1, len(d))
    y = (d - d.min()) / (d.max() - d.min())
    return d[np.argmax(np.abs(y - x))]


fixed_aris, knee_aris, collapsed = [], [], 0
for week_seed in (8002, 1, 2, 3, 4, 5, 6):
    week = make_bot_sessions(seed=week_seed)
    week_truth = (week["true_group"] == "bot").astype(int)
    Zw = StandardScaler().fit_transform(
        week[["session_duration_min", "pages_viewed"]].values)

    fixed_labels = DBSCAN(eps=0.30, min_samples=10).fit_predict(Zw)
    knee_labels = DBSCAN(eps=_knee_eps(Zw), min_samples=10).fit_predict(Zw)

    fixed_aris.append(adjusted_rand_score(week_truth, fixed_labels))
    knee_aris.append(adjusted_rand_score(week_truth, knee_labels))
    collapsed += len(set(fixed_labels) - {-1}) < 2

at_least("3.4 a fixed eps=0.30 collapses on at least one regenerated week",
         collapsed, 1)
at_most("3.4 ... and the worst such week scores essentially zero",
        min(fixed_aris), 0.05)
at_least("3.4 the k-distance recipe recovers the split on every week",
         min(knee_aris), 0.70)
at_least("3.4 on the lesson's own week the tuned constant beats the recipe",
         fixed_aris[0] - knee_aris[0], 0.01)

# -------------------------- Section 6.1, the range behind "37 of 40"
#
# The handout carries 37 as the memorable figure and 32-38 as what it means.
# The range is the part worth checking: a detector quoted as a specification
# would be quoting one quarter's luck.
caught_counts = []
for batch_seed in (8003, 1, 2, 3, 4, 5, 6, 7):
    batch = make_customer_features(seed=batch_seed)
    batch_label = batch["is_anomaly"].values
    Zb = StandardScaler().fit_transform(batch[feature_names].values)
    pb = PCA(n_components=TRUE_N_LATENT_FACTORS).fit(Zb)
    err = np.sum((Zb - pb.inverse_transform(pb.transform(Zb))) ** 2, axis=1)
    hit = err > np.quantile(err, 0.98)
    caught_counts.append(int((hit & batch_label).sum()))

same("6.1 this batch catches the handout's 37", caught_counts[0], 37, tolerance=0)
at_most("6.1 the worst batch catches materially fewer than 37",
        min(caught_counts), 34)
at_least("6.1 ... and the best catches at least 37", max(caught_counts), 37)

print(f"{checks} numbers recomputed from raw inputs and confirmed against {HANDOUT}.")

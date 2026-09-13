---
title: "Unsupervised Learning — Key Concepts"
subtitle: "Lesson 8 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "13 November 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly. Use this to find your way back into the text, or to check
before an exercise that there is no word here you could not define.

---

## What changes without a target

**Unsupervised learning.** Learning from a table with no answer key: the task is to
describe structure rather than predict a label. → § 1

**Why it usually comes first.** Most data an organisation collects was never
labelled, and for a question like "which of these are alike" the labels are not even
well defined until someone clusters and looks. → § 1

---

## k-means

**Within-cluster sum of squares (WCSS).** The total squared distance from each point
to its own cluster's centre — the quantity k-means minimises. → § 2.1

**Lloyd's algorithm.** Alternate two exact steps: assign every point to its nearest
centre, then move every centre to the mean of its points. → § 2.1

**Why it converges.** Neither step can increase the objective, and there are only
finitely many ways to partition the points — so the sequence must stop. → § 2.1

**Local minimum.** What it stops at: a partition no single reassignment improves, not
necessarily the best one. Where it lands depends entirely on where the centres
started. → § 2.1

**k-means++.** An initialisation that picks each new centre with probability
proportional to its squared distance from the nearest centre already chosen, so a
cluster with no representative is favoured. → § 2.2

**The elbow.** Where the WCSS curve stops falling steeply. Past the true number of
clusters, another one only subdivides a group that was already coherent. → § 2.3

**Silhouette score.** Per point, how much closer it is to its own cluster than to the
nearest other one. Negative means the assignment is actively wrong, not merely
ambiguous. → § 2.3

**Adjusted Rand index (ARI).** Agreement between two labellings of the same points,
corrected so that a random assignment scores zero. An external metric: it needs a
ground truth. → § 2.3

---

## When round is the wrong shape

**Agglomerative clustering.** Start with every point its own cluster and repeatedly
merge the two closest, recording every merge as a branch. → § 3.2

**Dendrogram.** That record of merges. Cutting it at a height is the same as choosing
a number of clusters. → § 3.2

**Linkage criterion.** How the distance between two *clusters* is defined — closest
pair, farthest pair, average, or least increase in within-cluster variance. It
decides what shapes the method can find. → § 3.2

**Chaining.** Single linkage's failure: one close pair is enough to merge, so the
method strings together anything that touches, absorbing the dataset along its
densest path. → § 3.2

**DBSCAN.** Clustering by density: a cluster is a connected region of crowded points,
whatever its shape. → § 3.3

**Core, border and noise points.** Crowded enough to grow a cluster; close to one but
not able to extend it; neither. Being able to answer *neither* is something no method
in § 2 can do. → § 3.3

**`eps` and `min_samples`.** The neighbourhood radius and the count that makes a point
a core. Choosing `eps` is the decision that matters. → § 3.3

**k-distance plot.** Every point's distance to its `min_samples`-th neighbour, sorted.
The bend between the dense and sparse regimes is where `eps` belongs. → § 3.3

**A constant does not transfer; the recipe does.** A value read off one week's curve
can fail outright on the next week's data, while re-reading it each time does not.
→ § 3.4

---

## Judging a clustering

**Internal metrics.** Computed from the clustering and the data alone — silhouette,
WCSS. They measure self-consistency, and can be confidently wrong about whether the
algorithm's assumption fitted at all. → § 4

**External metrics.** Compared against a separate labelling assumed correct. Real
problems rarely have one as cleanly as synthetic data does. → § 4

---

## Principal component analysis

**Principal component analysis (PCA).** Find the direction along which the data varies
most, then the next-most perpendicular to it, and so on. → § 5.1

**Why variance stands in for information.** A direction along which every point looks
alike cannot help tell points apart, so it is safe to discard. → § 5.1

**Covariance matrix.** The matrix whose eigenvectors are the principal components and
whose eigenvalues are the variance each one captures. → § 5.2

**Singular value decomposition (SVD).** The numerically preferred route to the same
components, because it never forms the covariance matrix — and squaring a matrix
squares its conditioning problem. → § 5.3

**Scree plot.** Explained variance per component, read for where it stops falling
steeply. The judgement PCA does not make for you. → § 5.4

---

## Using what PCA leaves out

**Reconstruction.** Mapping a point onto the kept components and back: the closest
point to it that lies on the subspace PCA selected. → § 6.1

**Reconstruction error.** How far a point is from that subspace. Large for a point
whose columns do not co-vary the way genuine points' do, even when every single value
is ordinary. → § 6.1

**Why a two-dimensional plot is the wrong tool here.** It answers "how far from the
middle", and these anomalies are identified by "how badly do the kept components
predict the dropped ones" — a comparison the picture has already thrown away. → § 6.2

**t-distributed stochastic neighbour embedding (t-SNE).** A method that arranges
points so near ones stay near, with no commitment to distances between far ones. → § 7

**What t-SNE is not for.** Distances between its clusters are not meaningful, and
there is no fixed transform to place a new point into an existing embedding. → § 7

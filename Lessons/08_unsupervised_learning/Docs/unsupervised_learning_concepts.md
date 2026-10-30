---
title: "Unsupervised Learning — Key Concepts"
subtitle: "Lesson 8 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "27 November 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly. Use this to find your way back into the text, or to check
before an exercise that there is no word here you could not define.


## What changes without a target

**Unsupervised learning.** Learning from a table with no answer key: the task is to
describe structure (which records are alike, which few numbers summarise many,
which records do not belong) rather than predict a label. → § 1

**Ground truth, and why this lesson has one.** The rules that generated the data,
published in `retail_data.py`. Real data has none, which is what makes checking an
unsupervised result hard. → § 1.1


## k-means

**Centre.** One representative point per group, its "typical member"; k-means
gives every point to the nearest centre. → § 2.1

**Standardising before clustering.** Putting every column in units of its own
spread, so that distance is not decided by whichever column has the largest
numbers. → § 2.1

**Assign and move.** The two moves k-means repeats: give each point to its nearest
centre, then put each centre at the mean of its points. → § 2.2

**Within-cluster sum of squares (WCSS).** The squared distance from each point to
its own centre, added up: how tight the groups are. `inertia_` in scikit-learn.
→ § 2.2

**Why k-means always stops.** Neither move can raise the WCSS, and there are only
finitely many groupings, so a round eventually changes nothing. → § 2.4

**Local minimum, and several starts.** Stopping is not finding the best grouping:
a bad start can stop at a much worse one. Run from several starts and keep the
lowest WCSS (`n_init`). → § 2.4

**k-means++.** A way of choosing the starting centres that prefers points far from
those already chosen. → § 2.4

**Elbow.** The point where WCSS, plotted against $k$, stops falling steeply. → § 2.5

**Silhouette.** For one point, $(b - a)/\max(a, b)$, comparing its mean distance
to its own group with its mean distance to the nearest other group; averaged over
all points to score a grouping. → § 2.5

**Adjusted Rand index (ARI).** Agreement between two groupings of the same
points, counted over pairs and rescaled so that a random grouping scores about zero.
→ § 2.6

**Internal and external measures.** Internal ones (silhouette) need only the data
and the grouping; external ones (ARI) need the true grouping. → § 2.6


## When groups are not round

**Why k-means draws straight boundaries.** With two centres, the points equally
far from both form a straight line, so the two groups are always the two sides of
one. → § 3.2

**Hierarchical clustering.** Start with every point alone and repeatedly merge
the two closest groups. → § 3.3

**Dendrogram, and cutting it.** The tree of merges, with each join drawn at the
height of its merge; a horizontal cut chooses the number of groups. → § 3.3

**Linkage, and Ward linkage.** The rule for the distance between two groups;
Ward merges the pair whose union adds least to the WCSS. → § 3.3

**DBSCAN.** Density-based spatial clustering of applications with noise: groups
are crowded connected regions, found without centres and without $k$. → § 3.4

**Core, border and noise points.** A core point has at least `min_samples` points,
itself included, within `eps`; a border point lies within `eps` of a core point;
noise is neither. → § 3.4

**k-distance plot.** Every point's distance to its $k$-th nearest neighbour,
sorted; its upward bend suggests `eps`. → § 3.5

**Noise as an answer.** A point DBSCAN will not place is "not confidently either",
which can be more useful than a forced label. → § 3.6

**What the silhouette cannot see.** It rewards compact groups, so on a ring it
prefers a wrong answer. → § 3.7


## Principal component analysis

**Principal component analysis (PCA).** Finding a few new columns, directions in
the data, that carry almost all of its variation. → § 4.1

**Shadow (projection).** A point dropped at a right angle onto a direction; its
position along the direction is a one-number summary. → § 4.2

**First and second principal components.** The direction along which the shadows
spread most, and the next one at right angles to it. → § 4.2

**Eigenvectors of the covariance matrix.** What the principal components are, in
any number of columns; each eigenvalue is the variance along its component. → § 4.3

**Explained variance ratio.** Each component's share of the total variance. → § 4.3

**Scree plot.** The shares drawn component by component; keep the components
before the drop. → § 4.4


## Anomalies

**An anomaly in the combination.** A record ordinary on every column that breaks
how the columns move together. → § 5.1

**Reconstruction.** A record rebuilt from its shadow on the kept components
(`inverse_transform` of `transform`). → § 5.2

**Reconstruction error.** The squared distance between a record and its
reconstruction; large when the kept components cannot describe it. → § 5.2

**One batch, and the range.** A detection count is one draw; repeating on other
batches from the same process gives the range it belongs to. → § 5.3

**t-SNE.** t-distributed stochastic neighbour embedding: a two-dimensional picture that keeps
close neighbours close and does not keep large distances; for looking, not
measuring. → § 5.4

**Far from the middle, far from the pattern.** Why a two-dimensional picture can hide
anomalies that reconstruction error finds. → § 5.4

**Choosing a method.** Which of today's tools fits which task. → § 6

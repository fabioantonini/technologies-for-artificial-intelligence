---
title: "Lesson 8: Unsupervised Learning"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "13 November 2026"
---

# Agenda

- A fourth question this course asks of data: no target at all
- k-means: the objective, Lloyd's algorithm, choosing $k$
- Hierarchical clustering and DBSCAN: shape versus density
- Validating a clustering with no labels
- Principal component analysis (PCA) and t-SNE

::: notes
Frame the whole lesson before the first slide of content. Every method met
so far in this course - regression, logistic regression, trees, ensembles  - 
learned from pairs: an input and a target. Today removes the target
entirely. The task changes from "predict the label" to "describe the
structure": which points look alike, which few numbers summarise the many,
which points do not belong.

Say the running example once, up front: a fictional retailer, Aurora, and
three of its actual questions, one per notebook - segmenting customers,
telling bots from humans, and catching disguised fraud. None of the three
has a $y$ supplied by anyone.

Handout section 1.
:::

# Exercise 7, before we start

- What counts is the **defence of the chosen family**, not the winning score
- The gap to watch for: a feature-importance ranking quoted without checking how
  many candidate columns competed for attention at each split
- Today's methods come with a version of the same obligation: a clustering
  or a projection is a claim, and it needs its own check

::: notes
Nothing to collect and nothing to hand back - discuss it. Ask what the
importance ranking was taken to mean, and listen for the gap: reading it as
settled fact rather than as something section 7 of that lesson showed could be
dominated by noise columns.

Link forward: today has no labels at all, so there is no accuracy score to
fall back on even provisionally. The obligation to check a result does not
go away just because there is no $y$ to check it against - it changes
shape, and section 4 of today's handout is entirely about that change.
:::

# No $y$ at all

- Every method so far learned from pairs: input $x$, target $y$
- **Unsupervised learning** removes $y$: measurements, no answer key
- The question: not "predict the label" but "describe the structure"
- Most organisational data was never labelled: expensive, slow, often
  not well-defined
- Often the *first* pass on a dataset

::: notes
Make the shift concrete rather than abstract. Ask the room: if you were
handed 2,000 customers' spending records with no group name attached to
any of them, what would "a good answer" even mean, before any algorithm
runs? There is no accuracy to check against - only whether the grouping
looks and behaves like something real.

That is the conceptual pivot for the whole three hours: every diagnostic
today either measures internal consistency (does this clustering agree
with itself) or borrows an outside signal when one happens to exist
(section 4). Neither is the clean "score against the test set" this course
has used since lesson 1.

Handout section 1.
:::

# Aurora: three problems, no answer key

- **Segment 2,000 customers**, from spend and visits (**k-means**, nb. 01)
- **Tell bots from humans** in 1,500 sessions, same *average* behaviour
  (**hierarchical clustering, DBSCAN**, nb. 02)
- **Catch disguised fraud** in an 8-column table, ordinary on every column
  (**PCA**, nb. 03)
- Every dataset is synthetic: its truth is normally hidden

::: notes
Walk the three problems as three genuinely different shapes of "no
labels" - not three demonstrations of the same idea. Segmenting customers
is close to what students probably picture when they hear "clustering."
Bots-versus-humans is deliberately built to break that picture: same mean,
different spread. The fraud problem breaks it again: individually
unremarkable values, anomalous only in combination.

Say explicitly that one notebook is not "the more advanced version" of
another - each is chosen because a different method is the right tool for
it, and section 8 of the handout is the summary table of when to reach for
which.

Handout section 1.1.
:::

# A ground truth Aurora will never have

- `Notebooks/retail_data.py` publishes the **true** structure as `TRUE_*`
  constants
- This lesson can check every method against a real answer: Aurora's own
  analysts never could
- Hold onto that asymmetry: it makes today teachable, and it is exactly
  what is missing the first time this runs on real data

::: notes
This is a methodological point, not a throwaway remark - flag it so it
does not get lost under the numbers that follow. Every "matches the truth"
claim this lesson makes (ARI scores, the scree plot recovering exactly
3 latent factors) is only checkable because the data was built for
teaching. On a real, unlabelled dataset, none of those checks are
available in the same form, which is precisely why section 4 exists: what
is left to check when there genuinely is no answer key.

Handout section 1.1.
:::

# 2,000 customers, no labels

![](customers_unlabelled.png)

::: notes
Let the picture sit for a moment before saying anything. Four clouds are
visible to the eye before any algorithm runs - ask the room to count them
before you say the number.

That is not an accident of this dataset: it is exactly the structure
k-means is built to exploit, and it is why this figure opens the section
rather than any equation. The "try this" in the handout asks students to
predict, from the four segment means alone, where each cloud should sit  - 
worth mentioning as a pre-read they can still do retroactively.

Handout section 2.
:::

# Turning "alike" into a number

- A good grouping puts every point close to one representative "typical
  customer," and far from every other group's
- **k-means** turns "close" into a number and minimises the total
- Two things are searched for jointly: every point's **assignment**, and
  the $k$ cluster **centres**
- Both at once is hard; the trick is minimising each separately, holding
  the other fixed

::: notes
Give the intuition with no symbols first, exactly as the handout does:
picture each cluster as having one representative point, and ask what
"good" means for such a picture. Then say that k-means turns "close" into
a squared distance and adds it up.

Flag the two-kinds-of-variable structure before the objective appears  - 
it is the reason the algorithm on the next few slides is two alternating,
each individually easy, steps rather than one hard search.

Handout section 2.1.
:::

# The objective: within-cluster sum of squares

- Minimise, jointly, over the cluster centres $\mu_j$ **and** every
  assignment $r_{ij}$
- $r_{ij} = 1$ if point $i$ is assigned to cluster $j$, else $0$: exactly
  one $1$ per row
- This quantity has a name: **within-cluster sum of squares (WCSS)**, or
  *inertia* in scikit-learn

::: notes
Set the two unknowns up before showing the formula on the next slide:
k-means is searching over the cluster centres *and* the assignment of
every point at once, and $r_{ij}$ is just bookkeeping for which point goes
where - 1 if point $i$ is in cluster $j$, 0 otherwise, exactly one 1 per
row.

Handout section 2.1.
:::

# Minimise $J$

$$J = \sum_{i=1}^{m}\sum_{j=1}^{k} r_{ij}\,\Vert x_i - \mu_j \Vert^2$$

::: notes
Read the sum out loud in words: for every point, add the squared distance
to the centre of the cluster it belongs to; add that up over all points.

This cannot be minimised in one shot: choosing the best assignment needs
the centres, and choosing the best centres needs the assignment. Do not
derive the two closed-form minimisations here - state that they exist and
move to what they say on the next slide. The full derivation, including
the gradient that gives the mean as the exact minimiser, is handout
section 2.1, for after the lecture.

Handout section 2.1.
:::

# Lloyd's algorithm: assign, then update, repeat

- **Assign step**: centres fixed, send every point to its **nearest**
  centre: exact minimiser over assignments
- **Update step**: assignment fixed, move every centre to the **mean** of
  its points, where the algorithm's name comes from
- Alternate the two until nothing changes
- Each step is an exact minimisation, no approximation, no step size

::: notes
Both steps are genuinely exact, not heuristic approximations - worth
stressing, since students have just spent two lessons on gradient descent,
where every step is approximate. Here, each half of the alternation solves
its half of the problem exactly: nearest-centre assignment is provably
optimal given fixed centres, and the mean is provably optimal given a
fixed assignment.

Ask the room: if each step is exactly optimal, what could possibly go
wrong? The answer is the next slide.

Handout section 2.1.
:::

# Every step can only lower $J$

- Assign and update each solve an exact minimisation, so neither step can
  ever **increase** $J$
- $J$ is bounded below by 0, with only finitely many ways to partition $m$
  points into $k$ groups
- Never increasing, cannot fall forever: the sequence must repeat, and that
  is convergence
- It converges to a **local** minimum, not necessarily the **global** one

::: notes
This is the argument for why Lloyd's algorithm terminates at all, and it
is worth giving in full even though it is not a derivation of a formula  - 
it is three short facts (non-increasing, bounded below, finitely many
partitions) chained together, and it is the kind of argument students will
meet again.

The sting is in the last bullet. Converged does not mean correct - it
means stuck, possibly at a bad local minimum. That is exactly what the
next few slides measure.

Handout section 2.1.
:::

# Worked example: six customers, badly-chosen starts

| Iteration | Assignment | WCSS after assign | WCSS after update |
|---|---|---|---|
| 0 | `[0 1 1 1 0 1]` | 15.965 | 7.281 |
| 1 | `[0 0 1 1 0 1]` | 5.491 | 4.674 |
| 2 | `[0 0 1 1 0 1]` | 4.674 | 4.674 |

::: notes
Six standardised customers, two deliberately bad initial centres - the
first two points in the dataset. Walk the WCSS column: 15.965, 7.281,
5.491, 4.674 - falling at every single step, exactly as the previous slide
argued it must.

By iteration 2 the assignment repeats and both steps leave $J$ unchanged:
converged. In this particular six-point case it happens to be the global
optimum - say clearly that this is a property of this small example, not
a guarantee, since section 2.2 is about exactly the case where it is not.

`Docs/worked_examples.py` reproduces this table directly from
`retail_data.py`. Handout section 2.1.
:::

# A bad start is a real risk

- 30 independent naive random starts of Lloyd's algorithm on the
  2,000-customer data
- Best final WCSS: **374.1**. Worst: **1,390.4**, **3.7 times worse**,
  same data, same update rule
- This happened on the very **first** attempt (worst of 30), not a rare
  one-in-a-thousand pathology
- The naive fix ($k$ points picked uniformly at random) gives no way to
  tell in advance which kind of run you got

::: notes
Land the number, then the framing. Nothing about the update rule changed
between the best run and the worst - only where the search started. That
is the entire content of "local, not global minimum" made concrete.

Ask the room: if you ran this once, on real data, with no truth to check
against, how would you know which kind of run you got? You would not - the
WCSS value alone does not distinguish a global optimum from a mediocre
local one, because you have no reference point.

Handout section 2.2.
:::

# k-means++: favour centres far from what's chosen

$$P(x_i \text{ chosen next}) = \frac{d(x_i)^2}{\sum_{i'} d(x_{i'})^2}, \qquad d(x_i) = \min_{j \text{ chosen}} \Vert x_i - \mu_j \Vert$$

::: notes
Say the rule in words first: the first centre is a uniformly random point;
every centre after that is picked with probability proportional to its
squared distance from the nearest centre already chosen. A point already
close to an existing centre is unlikely to be picked again; a point far
from everything picked so far - plausibly the seed of an as-yet-unrepresented
cluster - is favoured.

This is initialisation only - Lloyd's algorithm afterward is completely
unchanged. It addresses exactly the risk on the previous slide, by making
a bad draw of starting centres much less likely.

Handout section 2.2.
:::

# k-means++, measured

- Same 30 seeds, k-means++ instead of naive random
- Range narrows to **374.1–1,088.3**: the worst case gains ~300 units
- Reaches the **global optimum on the first attempt**
- `sklearn.cluster.KMeans` uses k-means++ plus 10 restarts by default
- Notebook 01 matches its WCSS to $10^{-4}$

::: notes
Read the range change against the previous slide's 374.1–1,390.4  - 
narrower, and reaching the optimum immediately rather than only sometimes.
This is why k-means++ is the library default rather than an optional
extra.

The $10^{-4}$ agreement is worth naming explicitly as a genuine check, not
an assumption - the from-scratch and scikit-learn implementations are
searching for the same optimum from starts drawn the same way, and the
notebook confirms it rather than takes it on faith.

Handout section 2.2.
:::

# Choosing $k$

- $k$ is an **input** to k-means, not something the algorithm discovers
- WCSS is monotonically non-increasing in $k$: at $k=m$, $J=0$ exactly
- So the raw value of $J$ at the "best" $k$ is not the signal
- **The elbow**: where the WCSS curve stops falling steeply, past the
  true count, an extra centre only subdivides an already-coherent group

::: notes
Make the trap explicit: a student who does not think about this will be
tempted to pick the $k$ with the lowest WCSS, and the lowest WCSS is
always the largest $k$ tried, which is a useless answer.

The elbow logic is a shape argument, not a threshold - ask the room how
they would describe "where the curve stops falling steeply" precisely
enough to automate it. The honest answer is that it is somewhat
subjective, which motivates a second, more quantitative diagnostic on the
next slide.

Handout section 2.3.
:::

# The silhouette score

- For point $i$: $a_i$ is its mean distance to every other point in its
  **own** cluster
- $b_i$ is its mean distance to the points of the **nearest other**
  cluster
- Combine the two into a single number per point

::: notes
Set up the two ingredients before showing the formula on the next slide:
$a_i$ measures how well a point fits its own cluster, $b_i$ measures the
best alternative it could have had instead.

Handout section 2.3.
:::

# Between $-1$ and $1$

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$$

::: notes
Walk the three regimes: $s_i \to 1$ when a point sits deep inside its own
cluster and far from every other; $s_i \approx 0$ on a boundary between
two clusters; $s_i < 0$ when a point is, on average, closer to a different
cluster than its own - an assignment the silhouette treats as actively
wrong.

Averaged over all points, this gives one number per $k$ that is a genuine
quantity to search over, not only a shape to eyeball - a sharper diagnostic
than the elbow, and one that measures something related but not identical:
whether every point sits closer to its own group than to any rival.

Handout section 2.3.
:::

# Elbow and silhouette, on the same data

![](elbow_silhouette.png)

::: notes
WCSS falls sharply through $k=4$ and flattens after; silhouette peaks at
$k=4$ and then declines. Point out that the two curves are answering
different questions - WCSS asks only "does adding a cluster help," which
is agnostic to what a cluster means; silhouette asks whether every point
is closer to its own group than to any rival, a stronger, more geometric
condition.

Say plainly that the two diagnostics agreeing here is a property of this
dataset's four well-separated, comparably sized blobs - not a guarantee.
Section 4 of the handout returns to a dataset where a diagnostic can be
confidently wrong.

Handout section 2.3.
:::

# Both point to $k=4$

- WCSS at $k=4$: **374.1**, the sharpest bend in the elbow curve
- Silhouette at $k=4$: **0.690**, the peak across $k=1$ to $8$
- Both match the number of segments the data was actually built with
- Agreement here is worth noting precisely **because** it does not always
  happen, and is not required to

::: notes
State both numbers before the punchline. The agreement is genuinely nice
for teaching, but resist letting the class walk away thinking the two
diagnostics always agree - that is exactly the kind of overclaim the
handout's "predictable mistake" boxes exist to head off, and this lesson's
own section 4 will show a case where a diagnostic looks fine and is not.

Handout section 2.3.
:::

# k-means against a ground truth Aurora doesn't have

![](kmeans_vs_truth.png)

::: notes
Left: the true segments, generated by `retail_data.py`, never shown to
k-means. Right: k-means' own $k=4$ clustering, fitted centres marked.
Visually near-identical - say the number on the next slide rather than
here, since this slide is the figure alone.

Repeat the asymmetry from earlier: this comparison exists only because the
data is synthetic. Aurora's own analysts would have the right panel and
nothing else.

Handout section 2.3.
:::

# The adjusted Rand index

- Compares two labellings: a **random** assignment scores 0, **identical**
  labellings score 1
- Cluster "0" need not mean the same thing on both sides: relabelling is
  handled
- k-means vs. the true segments: **0.985**, near-perfect
- An **external** metric: needs a ground truth, which section 4 asks about
  when there isn't one

::: notes
Give the definition before the number, since ARI will recur for the rest
of the lesson as the standard way to score a clustering against a truth.
0.985 is about as good as this metric gets - worth contrasting later
against the bot/human numbers in section 3, which come out negative.

Bridge forward explicitly: everything scored today with ARI is only
checkable because this lesson knows the answer. Section 4 is about what
happens when it does not.

Handout section 2.3.
:::

# Notebook 1, live

- k-means from scratch, checked against scikit-learn's k-means++ and
  10-restart defaults
- The 30-seed initialisation comparison, reproduced live
- The elbow and silhouette curves, computed directly from the data

::: notes
Run Notebooks/01_kmeans_from_scratch.ipynb. Twenty minutes.

Have them watch the from-scratch assign/update loop converge on the
six-point worked example before scikit-learn's version ever runs - the
agreement is checked, not assumed.

The cell worth protecting if time is short is the 30-seed initialisation
sweep: printing the 374.1-versus-1,390.4 gap themselves lands harder than
reading it off a slide.
:::

# A dataset built to defeat k-means

- Security suspects some of 1,500 sessions were automated
- Two features: duration in minutes, pages viewed
- **12%** are, by construction, **bots**: scripted and consistent
- Bots and humans share the **same mean**: a tight cloud inside a diffuse
  ring
- No straight cut separates them: there are no two offset clouds

::: notes
Set this up as a deliberate provocation, not a natural dataset. Section 2
just showed k-means recovering four clusters almost perfectly - this
section is built specifically to break the assumption that made that
possible.

Ask the room to predict, before the figure: if the two groups share a
mean, what happens when k-means tries to place two centres? The honest
answer, which the next slide shows, is that there is no separating line
for it to find at all.

Handout section 3.1.
:::

# 1,500 sessions, unlabelled

![](sessions_unlabelled.png)

::: notes
A dense smudge sitting inside a much larger, looser cloud, both centred in
roughly the same place. Let the room look for a gap a linear or round
boundary could exploit - there isn't one, and that is the point.

Contrast directly with the four visible clouds from section 2's opening
figure: same task (cluster with no labels), a completely different
picture, and it should not be obvious yet which algorithm handles it.

Handout section 3.1.
:::

# k-means on this shape

- Run k-means with $k=2$ on the session data
- ARI: **$-0.046$**, indistinguishable from a random split
- Not a tuning or initialisation failure, no starting point fixes it
- WCSS wants round, compact clusters; the only cut through this cloud
  slices **both** the bot core and the human ring in half

::: notes
Stress "no starting point fixes it" - this is not the local-minimum
problem from section 2.2, which k-means++ addresses. This is a mismatch
between what k-means optimises (compactness) and what the data actually
looks like (a core and a surrounding ring, same centre). Better
initialisation cannot fix an objective that is the wrong shape for the
data.

Ask the room: what would a "cluster" need to mean for this shape to be
findable at all? The answer - density, not compactness - is where the
next two sections go.

Handout section 3.1.
:::

# Merge the two closest

- Start with every point as its **own** cluster
- Repeatedly merge the two **closest** clusters, recording every merge as
  a branch in a tree: a **dendrogram**
- Cutting the tree at a height is equivalent to choosing $k$: branches
  crossed = cluster count
- "Closest" needs a rule for distance **between clusters**: the
  **linkage criterion**

::: notes
Introduce the dendrogram as an object before any linkage rule - the tree
itself is the useful idea, and any linkage rule can build one. Say
explicitly that choosing $k$ after the fact, by cutting the tree at a
height, is a genuinely different workflow from k-means, where $k$ has to
be fixed before the algorithm starts.

The linkage criterion is where the real content is, and it is the next
slide.

Handout section 3.2.
:::

# Four notions of "closest"

- **Single**: closest pair, which can chain a winding path, or bridge two
  groups at one touching point
- **Complete**: farthest pair, which favours compact groups
- **Average**: mean distance over every cross-pair, a compromise
- **Ward**: least added variance, the *same* objective k-means minimises

::: notes
Four rules, four different biases - worth naming what each one is built to
find, since the next slide is going to show all four failing on this
particular dataset for two different reasons.

Ward is worth pausing on: it is not a fourth, unrelated idea, it is
k-means' own objective expressed as a merge rule, so predict out loud that
it should fail on this data for the same reason k-means just did.

Handout section 3.2.
:::

# Ward-linkage merges on 60 sessions

![](dendrogram.png)

::: notes
A branch peeling off single points one at a time, rather than splitting
into two comparably sized halves, is the visual signature of chaining  - 
point it out directly on the figure, even at this small 60-session sample.

This is a preview of what the full 1,500-session numbers on the next slide
make precise: chaining is not a rare edge case here, it is what several of
the four linkage rules do on the whole dataset.

Handout section 3.2.
:::

# All four linkage rules, scored

| Linkage | ARI | Cluster sizes |
|---|---|---|
| Ward | $-0.062$ | 1,007 / 493 |
| Complete | $-0.101$ | 1,253 / 247 |
| Average | $-0.001$ | 1,499 / 1 |
| Single | $-0.001$ | 1,499 / 1 |

::: notes
Read the two failure modes separately, they are not the same failure.
Ward and complete fail for the same reason k-means did - both still search
for compact, comparably sized pieces, which this shape does not have.
Single and average instead produce one cluster of 1,499 points and a
cluster of size 1 - not a discovery, but chaining: the algorithm absorbed
almost the entire dataset along its densest path before running out of
points to merge.

Land the summary line: **all four linkage rules score at or below zero
ARI**. Hierarchical clustering, by itself, is not a fix for the shape
problem - it needs a rule that looks for density rather than compactness
or chained proximity.

Handout section 3.2.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. What comes back after the break is the method that
actually solves the bot/human problem - worth saying, so the room returns
with the open question fresh: what would a cluster need to mean for that
ring-around-a-core shape to be findable?
:::

# DBSCAN: density instead of shape

- **DBSCAN**: density-based spatial clustering of applications with noise
- Classifies points by neighbourhood crowding: radius `eps`, minimum count
  `min_samples`
- **Core point**: at least `min_samples` others within `eps`
- **Border point**: within `eps` of a core, but not core itself
- Everything else: **noise**

::: notes
Say the name in full once, then use the acronym. Walk the three point
types slowly - core, border, noise - since the next slide's definition of
a cluster depends on all three.

Nothing here mentions a centre or a mean - flag that explicitly as the
structural difference from everything in sections 2 and 3.2.

Handout section 3.3.
:::

# What counts as a cluster

- A **maximal set of core points**, chained by mutual `eps`-closeness,
  plus every attached border point
- No centre, no mean, no shape assumption in the definition
- A compact core and a diffuse ring both count, if dense enough at the
  scale `eps` measures
- Points can belong to **nothing**: noise is first-class, not an error

::: notes
This is the slide that answers the question left open before the break:
a cluster defined by density, rather than compactness, can be a ring
around a core, because nothing in the definition forces round or convex
shapes.

The "noise is first-class" point matters on its own - every method in
sections 2 and 3.2 must assign every point to exactly one cluster. DBSCAN
does not, and section 3.3's numbers later show exactly what that buys.

Handout section 3.3.
:::

# Choosing eps: the parameter that matters most

- Too small: even the dense bot core fractures into isolated noise points
- Too large: the ring bridges back into the core, everything merges, as
  Ward linkage did
- A **$k$-distance plot**: distance from every point to its
  `min_samples`-th nearest neighbour, sorted
- Dense regions give a small distance, sparse ones a large one: the plot
  typically bends sharply between the two

::: notes
Two failure directions, both bad, and `eps` sits between them - motivate
the $k$-distance plot as a principled way to find that middle point rather
than guessing.

Ask the room what shape they expect the sorted-distance curve to have
before showing it. The intuition: flat while you are still inside dense
regions, then a sharp climb once points start being genuinely far from
their neighbours.

Handout section 3.3.
:::

# The bend in the curve

![](k_distance_plot.png)

::: notes
10th-nearest-neighbour distance, sorted, for all 1,500 sessions. The curve
stays low and flat through most of the range and bends sharply upward near
the end.

`eps = 0.30` sits just past the bend, in the flat region - point at
exactly where on the curve that is. This is the value used on the next
slide's results.

Handout section 3.3.
:::

# DBSCAN, measured

| | Result |
|---|---|
| ARI against true bot/human split | **0.941** |
| Core points | 1,435 of 1,500 |
| Noise points | 13 |

::: notes
0.941 against k-means' $-0.046$ and Ward's $-0.062$ on the identical
dataset - same data, a method built for the actual shape.

Worth reading the noise points individually rather than as a rounding
error: checking their true label shows all 13 are genuine human sessions,
not misplaced bots - sessions whose one week of browsing happened, by
chance, to look almost as mechanically regular as a script's. DBSCAN's
answer for them - not confidently either group - is more honest than a
forced binary label, and it is an answer no method in section 2 or 3.2 can
even give.

Handout section 3.3.
:::

# Three views, side by side

![](clustering_comparison.png)

::: notes
True group, k-means, and DBSCAN on the same axes. DBSCAN's cluster
boundary follows the actual density gap; k-means' straight cut does not
exist in this data at all - point directly at the contrast rather than
describing it.

This is the single figure to leave up longest in this segment - it makes
the whole section's argument visible in one look.

Handout section 3.3.
:::

# Does `eps` transfer?

| Week | ARI at `eps = 0.30` | Knee `eps` | ARI at knee |
|---|---|---|---|
| this one | **0.941** | 0.258 | 0.890 |
| week 4 | **−0.012**, *one cluster* | 0.226 | 0.858 |
| week 5 | **−0.009**, *one cluster* | 0.232 | 0.887 |
| the other four | 0.93 – 0.96 | 0.21 – 0.23 | 0.79 – 0.85 |

::: notes
Seven weeks regenerated from the same generator - same site, same bot
fraction, same shapes, different individual sessions. Read the two ARI
columns against each other before saying anything.

**The constant does not transfer.** On two of the seven, DBSCAN returns a
single cluster and an ARI of zero. Not a worse answer - no answer at all.
The reason is in the knee column: 0.30 is larger than any of these weeks'
own curves suggest, and on those two draws it is large enough to bridge the
density gap and swallow the ring into the core.

**The recipe does transfer.** Re-read `eps` off each week's own k-distance
curve and the split comes back every time, at 0.79 to 0.89.

Now the row to sit on, and it is the first one. On the week we studied, the
hand-picked 0.30 *beats* the recipe - 0.941 against 0.890. Say the
consequence out loud: the tuned constant is better where it works and absent
where it does not.

That is not a fact about DBSCAN. It is what fitting a constant to one sample
buys and costs, and lesson 5 said the same thing about trusting a single
train/test split. Ask the room which of the two they would ship to Aurora's
security team, and why.

Handout section 3.4.
:::

# Choosing among the three

| Situation | Reach for |
|---|---|
| Clusters plausibly round, similar size, $k$ known or searchable | k-means |
| Want the hierarchy itself: nested groupings at every scale | Agglomerative clustering |
| Clusters irregular, nested, or of very different densities; outliers should be flagged | DBSCAN |
| Dataset is very large, every point must be assigned to something | k-means (DBSCAN's noise needs a policy) |

::: notes
This table is worth reading as a decision the data makes, not the analyst.
The predictable mistake, from the handout: having watched k-means recover
the customer segments almost perfectly in section 2, the reasonable
conclusion is that clustering as a category works well. The instinct is
not wrong about k-means specifically - it is wrong about clustering being
one method with one assumption.

k-means and Ward both optimise "compact and round"; DBSCAN optimises
"densely connected" - a nested structure like today's bot core is exactly
where the two notions give opposite answers, and which definition is
right is a property of the data, not the algorithm.

Handout section 3.5.
:::

# Notebook 2, live

- Agglomerative clustering, all four linkage rules, and the dendrogram
- DBSCAN, the $k$-distance plot, and the `eps` sweep
- Every result checked against the true bot/human labels

::: notes
Run Notebooks/02_hierarchical_and_dbscan.ipynb. Twenty minutes.

Protect the `eps` sweep if time is short - watching ARI rise and then fall
again as `eps` moves past the bend makes the "too small fractures, too
large merges" argument concrete rather than asserted.

Have them check the 13 noise points' true labels themselves rather than
reading the number off a slide.
:::

# No answer key this time

- Every check so far, ARI in sections 2 and 3, used a ground truth this
  lesson happens to have because the data is synthetic
- Aurora's actual analysts, on real customer or session data, will never
  have that
- Two families of metric fill the gap, and it matters which one is
  available in a given situation

::: notes
Return explicitly to the asymmetry flagged in section 1: everything scored
with ARI today is checkable only because the truth is published. This
section is about what is left when it is not.

Set up the two-way split before naming either family - internal (only the
clustering and the data) versus external (an outside answer key) - the
next two slides take one each.

Handout section 4.
:::

# Internal metrics: only the clustering and the data

- Use **no** outside answer key: computed from the clustering and the
  points alone
- **Silhouette**: are points closer to their own cluster than to any
  other? Answerable from the clustering alone
- **WCSS**: usable, but weaker, only comparable across different $k$ on
  the *same* algorithm, not across different methods
- Always available, even when nothing else is

::: notes
Silhouette already appeared in section 2.3 as a way to choose $k$ - here
it is being recast as one member of a broader category, "internal
metrics," which is the useful generalisation.

The WCSS caveat is worth spelling out: it cannot be used to compare
k-means against DBSCAN, because DBSCAN does not even define centres the
way WCSS needs. That limitation is part of why a family of alternatives
exists at all.

Handout section 4.
:::

# External metrics: an outside answer key

- Compare a clustering against a **separate** labelling assumed correct,
  ARI and its relatives
- Here: the data generator's `true_segment` / `true_group`
- In practice: a hand-labelled sample, a business rule, or a downstream
  outcome: did the "bot" cluster get blocked and never return?
- When such a signal exists, even partial or noisy, **use it**

::: notes
The examples of a real-world external signal matter more than the
definition here - a downstream outcome (blocked and never returned) is
the kind of proxy label that is available far more often than a clean,
hand-verified truth, and it is worth spending a moment on that it counts
even when partial or noisy.

Set up the punchline explicitly: internal metrics can be confidently
wrong, because they have no way to know the clustering's underlying
assumption is the wrong one for this data - only whether it is
self-consistent. That is the next slide.

Handout section 4.
:::

# What silhouette can't see

- A silhouette of 0.69, reported alone, sounds like strong endorsement
- It endorses only **what the metric can see**: the shape, not the
  assumption behind it
- k-means on the bot/human data (ARI $\approx -0.046$) still scores
  positive
- Silhouette measures **roundness**, not correctness, and k-means always
  looks round by construction

::: notes
This is the handout's named predictable mistake, and it deserves to be
said slowly. Ask the room directly: if you only had the silhouette score
and not the ARI, would you have any hint that k-means' clustering on the
session data is worthless? The honest answer is no - the internal metric
alone gives no warning.

That is the whole reason external validation matters whenever any signal
for it exists, even a rough one, and it is why this section sits between
the clustering methods and PCA rather than being a footnote to section 2.

Handout section 4.
:::

# Eight columns, not eight independent numbers

![](feature_correlation.png)

::: notes
Aurora's account table has 8 columns. The block of warm cells among
spend-related columns, and the separate block among engagement-related
columns, is the correlation structure PCA is about to compress - point at
both blocks directly.

Say plainly what this figure is doing here: it is the motivation for
everything that follows, not a result. Spend, basket value and mobile
session count all move together because they partly reflect the same
underlying tendency to spend - PCA is about to find that underlying
tendency directly.

Handout section 5.1.
:::

# PCA: the direction that varies most

- Finds the direction in feature space along which the data varies
  **most**: the first principal component
- Then the next-most-varying direction, **perpendicular** to the first,
  and so on
- "Varies the most" stands in for "carries the most information": a
  direction where every point looks nearly identical tells points apart
  hardly at all
- A direction like that is safe to discard

::: notes
Give the intuition before any algebra, exactly as the handout does: PCA is
looking for the direction that best tells points apart, not a mysterious
transform. "Safe to discard" is the key phrase - it is what makes
dimensionality reduction a defensible operation rather than just throwing
information away.

The next few slides give the two derivations the handout works through in
full - state results here, point to the handout for the steps.

Handout section 5.1.
:::

# Maximise variance, subject to unit length

- The variance of standardised data projected onto a direction $v$, with
  ‖v‖ = 1, is what PCA maximises
- $\Sigma$ is the sample covariance matrix: diagonal 1, off-diagonal
  entries the correlations the previous figure showed as colours
- The unit-length constraint matters: without it, variance grows without
  bound simply by scaling $v$ up

::: notes
"Maximise variance" is meaningless until direction is separated from
length - that is what the unit-length constraint buys. The handout works
the constrained optimisation through a Lagrange multiplier in full  - 
worth saying that word exists, not deriving it here.

Handout section 5.2.
:::

# Variance along a direction

$$\text{Var}(Xv) = v^\top \Sigma v, \qquad \Sigma = \frac{1}{m-1}X^\top X$$

::: notes
$\Sigma$ is the sample covariance matrix - for standardised columns, its
diagonal is 1 and its off-diagonal entries are correlations, the same
numbers the previous figure showed as colours.

Handout section 5.2.
:::

# The stationary points are eigenvectors of $\Sigma$

$$\Sigma v = \lambda v$$

::: notes
State the result, not the Lagrangian steps that get there - those are
handout section 5.2, in full. Say what this equation means in words:
directions that solve the constrained maximisation are exactly the
eigenvectors of the covariance matrix, with $\lambda$ the corresponding
eigenvalue.

Substituting back gives $v^\top \Sigma v = \lambda$ - the variance a
direction captures **is** its eigenvalue. That single fact is the bridge
from "an eigenvector" to "a principal component," and it is the fact
worth the class actually remembering.

Handout section 5.2.
:::

# The eigenvalue is the variance that direction captures

- Substituting $\Sigma v = \lambda v$ back: $v^\top \Sigma v = \lambda$
- The direction of **largest** variance is the eigenvector with the
  **largest** eigenvalue; the second is next-largest, orthogonal to the
  first
- Orthogonality is automatic: $\Sigma$ is symmetric, by the spectral
  theorem
- Order all $n$ eigenvalues largest to smallest: every component, at once

::: notes
This slide turns the previous slide's single equation into the actual
recipe: sort eigenvalues, keep eigenvectors in that order, and that
ordered list *is* PCA. Worth saying explicitly that this is the entire
algorithm once the eigendecomposition is in hand - everything before this
was justifying why eigenvectors are the right thing to compute at all.

The automatic-orthogonality point is worth a beat: it is not a separate
constraint PCA has to enforce, it falls out of $\Sigma$ being symmetric.

Handout section 5.2.
:::

# Worked example: spend and visits

| | Value |
|---|---|
| Correlation (off-diagonal of $\Sigma$) | 0.171 |
| Eigenvalues | $1.171$ and $0.829$ |
| First component direction | $(1,1)/\sqrt{2}$ |

::: notes
Two standardised features from section 2 - spend and visit frequency. For
this symmetric $2\times2$ form, the eigenvalues are exactly $1+r$ and
$1-r$ for correlation $r$, checkable directly from $\Sigma v = \lambda v$
with no numerical solver - with $r = 0.171$: 1.171 and 0.829, matching
`numpy.linalg.eigh` to three decimal places.

The direction $(1,1)/\sqrt{2}$ is the "spend and visit often together"
diagonal, not either axis alone - say why that makes sense: customers who
spend more also tend to visit more, so the greatest joint variation runs
along the diagonal between the two original features.

Handout section 5.2.
:::

# The same answer, more stably: the SVD

- The **singular value decomposition (SVD)** of the centred data is the
  numerically preferred route to the same components
- Squaring a matrix squares its condition number, never form $X^\top X$
  explicitly

::: notes
Callback to lesson 3 deliberately - condition number squaring is the same
argument that motivated preferring gradient-based or QR-based solvers over
the raw normal equation there, and it is worth naming that connection out
loud so the pattern generalises for students.

The equation is on the next slide.

Handout section 5.3.
:::

# $\lambda_i$ from singular values

$$X = USV^\top, \qquad \lambda_i = \frac{s_i^2}{m-1}$$

::: notes
Say what the equation means: $U$ and $V$ orthogonal, $S$ diagonal with
non-negative singular values $s_1 \geq s_2 \geq \dots$; the columns of $V$
are exactly the eigenvectors of $X^\top X$, and $\lambda_i = s_i^2/(m-1)$
recovers the eigenvalues without ever forming $\Sigma$.

Handout section 5.3.
:::

# Three routes, one answer

- `numpy.linalg.eigh` on the covariance matrix
- `numpy.linalg.svd` on the centred data, with $\lambda_i = s_i^2/(m-1)$
- `sklearn.decomposition.PCA`
- On the 8-feature account table, all three agree to within
  $5 \times 10^{-15}$: floating-point rounding, and nothing more

::: notes
This is a genuine cross-check, in the same spirit as the k-means++
notebook agreeing with scikit-learn to $10^{-4}$ earlier - three
independent computational routes to the same mathematical object,
converging to sixteen significant figures.

Worth a sentence on why this matters pedagogically: it is one thing to
prove eigendecomposition and SVD give the same eigenvalues on paper, and
another to watch three separate library calls confirm it to
floating-point precision on real data.

Handout section 5.3.
:::

# How many components: the scree plot

![](pca_scree.png)

::: notes
Per-component and cumulative explained variance for the 8 account
features. The first 3 components explain 93.5% of total variance; the
4th adds only 2.4% more - point at exactly where the sharp elbow sits.

Ask the room to predict the number before you say it - most will read the
bend correctly from the shape alone, which is the whole point of a scree
plot as a diagnostic.

Handout section 5.4.
:::

# The elbow lands on 3

| Component | Eigenvalue | Variance | Cumulative |
|---|---|---|---|
| PC1 | 3.539 | 44.2% | 44.2% |
| PC2 | 2.568 | 32.1% | 76.3% |
| PC3 | 1.380 | 17.2% | 93.5% |
| PC4 | 0.192 | 2.4% | 96.0% |
| PC5–PC8 | 0.043–0.104 | 0.5–1.3% each | 100.0% |

::: notes
3 is not an arbitrary read of the elbow - it is the exact number of latent
factors (`retail_data.py`'s spending propensity, engagement, price
sensitivity) that generated all 8 observed columns. The scree plot
recovered, from the data alone, a number this lesson deliberately withheld
from every method until now.

Say clearly that this clean an elbow will not appear on every dataset
students meet later - it exists here because the generator built 8 columns
as noisy linear combinations of exactly 3 signals. The next slide is what
to do when it is not this clean.

Handout section 5.4.
:::

# What to do when the elbow is softer

- This dataset's elbow is unusually clean; most real data is not
- Common alternative: keep enough components for a chosen fraction of the
  variance, typically 90–95%
- Or treat the number of components as a hyperparameter, and let
  cross-validation decide
- The scree plot starts the judgement; it does not automate it

::: notes
Close the PCA segment by generalising beyond this lesson's convenient
dataset - students will not always get a number this clean, and this
slide is explicitly about what changes when they don't.

Frame the cross-validation option as a callback to lesson 5: choosing the
number of components is a model-selection decision like any other, and
the same honest-validation discipline that lesson used for hyperparameters
applies here too.

Handout section 5.4.
:::

# Notebook 3, live

- PCA on the 8-feature account table, checked three ways, to
  $5\times10^{-15}$
- Reconstruction error flags disguised fraud: **37 of 40** planted
  anomalies caught at the 98th-percentile threshold
- A 2-D PCA and t-SNE scatter of the same accounts, and why neither shows
  the anomalies as outliers

::: notes
Run Notebooks/03_pca_and_anomaly_detection.ipynb. Twenty minutes.

This is the lesson's headline number and it is delivered live in the
notebook rather than pre-built on a slide - reconstructing accounts from
the top 3 components and watching genuine accounts snap back almost
exactly, while planted anomalies do not, is more convincing run than
described.

Protect the reconstruction-error histogram and the 2-D scatter comparison
if time is short: the point that anomalies sit **closer** to the centre
than genuine accounts, on average, in both the PCA and t-SNE projections,
is the section's real content, and it needs to be seen, not just quoted.

Handout sections 6.1, 6.2.
:::

# t-SNE: a tool built for looking, not for measuring

- **t-SNE** (t-distributed stochastic neighbour embedding) arranges points
  so those close together **stay** close
- Distances between points that started far apart are not preserved
- Excellent at making separate clusters visually obvious, more so than
  PCA's first two components
- **Distances between clusters mean nothing**

::: notes
Position t-SNE precisely: a visualisation tool, not a measurement tool,
and the two limitations on this slide are the reason. Two clusters drawn
far apart in a t-SNE plot are not necessarily more different than two
drawn close together - resist any temptation to read distance off the
picture the way a PCA plot's axes can be read.

Tie back directly to notebook 03: t-SNE inherited the exact same blind
spot PCA's 2-D projection had for the fraud anomalies - good for looking,
not a substitute for reconstruction error. The full KL-divergence
derivation is deliberately out of scope; point to the further reading.

Handout section 7.
:::

# What to take away

- k-means converges only to a **local** minimum: 30 starts ranged WCSS
  374.1 to 1,390.4
- A core inside a ring defeats k-means **and** Ward; DBSCAN scored
  **0.941**
- Internal metrics measure self-consistency, external (ARI) a truth real
  problems rarely have
- **37 of 40 disguised accounts** caught by reconstruction error, invisible
  to a 2-D scatter

::: notes
This is the number to carry out of the room: 37 of 40, using nothing but
the relationship between eight individually ordinary-looking columns, on a
detection problem neither of the standard 2-D visualisation tools could
solve.

Ask them to write it down - the fourth lesson this course has produced one
of these carry-home numbers: 77% accuracy on coin-flip labels, 94 of 128
imputed rows that borrowed from the test set, a coefficient of 365 where an
unpenalised fit wanted billions, and now this.

If one habit should survive from today: before trusting an unlabelled
result, ask what the metric or the picture in front of you is actually
capable of seeing - a silhouette score cannot see a wrong shape
assumption, and a 2-D scatter cannot see what the discarded dimensions
would have shown.

Handout summary and notation table.
:::

# Homework

- **Exercise 8**, discussed at the start of **Friday 20 November**
- `Exercises/08_unsupervised_learning.md`

::: notes
Set it explicitly and say out loud when it comes back: the first ten
minutes of lesson 9, Friday 20 November.

Remind them of the standing rule from lesson 5, still in force with no
labels at all: every reported number needs a check against what the
method could and could not see - an ARI needs a ground truth to be
meaningful, a silhouette score needs to be read as roundness rather than
correctness, and a reconstruction-error threshold needs its false-positive
rate reported alongside its catch rate, not instead of it.

Next lesson moves to a different question this course has not yet
asked. Close the loop on today: four lessons in a row have each produced
one number worth remembering - encourage them to notice that pattern
themselves before it is pointed out again.
:::

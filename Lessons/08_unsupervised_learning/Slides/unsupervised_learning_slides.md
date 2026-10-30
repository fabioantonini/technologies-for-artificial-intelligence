---
title: "Lesson 8: Unsupervised Learning"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "27 November 2026"
---

# Agenda

- No target at all: what changes
- k-means: one centre per group, two moves, how many groups
- When groups are not round: a tree of merges, and DBSCAN
- Principal component analysis (PCA): the widest shadow
- Fraud that looks ordinary on every column, and how to catch it

::: notes
Frame the three hours before any content. Every method in this course so far
learned from pairs: an input and a target. Today removes the target entirely.
The task changes from "predict the label" to "describe the structure": which
records look alike, which few numbers summarise many columns, which records do
not belong.

Say what this version of the lesson leaves out: the proofs, and some extra
methods. They are in the advanced version of the lesson, in
Advanced/08_unsupervised_learning_advanced/, for anyone curious, and nothing
there is examined. Every method today is built from a picture and a small
worked example.

Handout section 1.
:::

# Exercise 7, before we start

- Nothing is collected: we discuss it, ten minutes
- What counts is the **defence of the chosen model family**, not the winning score
- The common gap: an importance ranking quoted as fact, without checking how
  much of it landed on noise columns
- Today the same duty, with no labels at all: every grouping is a claim to check

::: notes
Nothing to collect and nothing to hand back: discuss it. Ask what the
feature-importance ranking was taken to mean, and listen for the gap of lesson
7: reading the ranking as settled fact, when lesson 7 showed that a random
forest put 54% of its importance on twenty columns of pure noise.

Link forward. Today there is no accuracy to fall back on, because there is no
target. The duty to check a result does not disappear; it changes shape, and
the lesson returns to it three times: the check against a truth only synthetic
data has (slide 20), the silhouette that prefers a wrong answer (slide 35), and
the 2-D picture that points at the wrong accounts (slide 50).
:::

# No target at all

- So far: pairs, an input $x$ and a target $y$
- **Unsupervised learning** has only $x$: no answer key
- The task: "describe the structure", not "predict the label"
- Most data was never labelled: too slow, costly, or ill defined
- So these methods often run **first**

::: notes
Make the shift concrete. Ask the room: if you were handed 2,000 customers'
spending records with no group name attached to any of them, what would a good
answer even look like, before any algorithm runs? There is no accuracy to
compute, because there is nothing to compare the answer with.

"Not even well defined" deserves an example: "which of these customers are
alike?" has no answer until somebody groups them and looks. That is why
unsupervised methods often come first, and supervised learning second, once
the first look has suggested what a useful target would be.

Handout section 1.
:::

# Aurora: three problems, no answer key

- **Group 2,000 customers**, from annual spend and visits per month:
  **k-means**, notebook 1
- **Tell bots from humans** in 1,500 website sessions with the same average
  behaviour: **a tree of merges and DBSCAN**, notebook 2
- **Catch fraud** in an 8-column account table, where every value looks
  ordinary: **PCA**, notebook 3

::: notes
Aurora is a fictional online retailer, and the three problems are three
different shapes of "no labels", not three demonstrations of one idea.

Grouping customers is close to what most people picture when they hear
"clustering". Bots against humans is built to break that picture: the two kinds
of session have the same average duration and page count, and differ only in
how much they vary. The fraud problem breaks it again: every single value of a
fraudulent account looks ordinary, and only the combination is wrong.

DBSCAN and PCA are named here and defined on their own slides later (slides 29
and 39). Handout section 1.1.
:::

# A truth Aurora will never have

- `Notebooks/retail_data.py` makes all three datasets, and publishes how in
  `TRUE_*` constants
- So today every method can be checked against the real answer
- Aurora's own analysts could never do that check
- Remember the difference: it makes the lesson teachable, and it is exactly
  what is missing on real data

::: notes
This is a point about method, not a remark in passing, so say it slowly. Every
"matches the truth" result today, the agreement scores and the scree plot
finding exactly three hidden tendencies, can be checked only because the data
was built for teaching, with its generating rules written down in
retail_data.py.

On a real, unlabelled dataset none of those checks exists. Ask the room to keep
asking, at every result today: could Aurora have computed this number? The
answer separates the internal measures, computed from the data alone, from the
external ones, which need the truth.

Handout section 1.1.
:::

# 2,000 customers, no labels

![](customers_unlabelled.png)

::: notes
This is all Aurora's marketing team has: two numbers per customer, annual spend
in euros and visits per month, and no group name on either.

Ask the room how many groups they see. Most will say four, and they are right:
retail_data.py built four segments, with names like "window shoppers" and
"loyal high spenders". Four round clouds, each gathered around a middle point,
is exactly the situation the next method is built for.

Handout section 2, the opening figure.
:::

# One centre per group

- One point per group, its **centre**: the "typical customer"
- Every customer goes to the **nearest** centre
- **k-means** finds $k$ centres that keep customers close to their own
- $k$, the number of groups, is **your choice**
- "Nearest" is straight-line distance: **standardise** first (lesson 2)

::: notes
State the idea in words before any symbol. If customers really fall into a few
groups, one representative point per group describes them, and a good grouping
puts every customer close to their own representative.

Dwell on the last bullet, because it is the first mistake people make. Spend
runs into hundreds of euros, visits per month into single figures. Measured
raw, a difference of 50 euros outweighs a difference of 5 visits a month, and
the grouping would be decided by spend alone. Standardising, as in lesson 2,
measures each column in units of its own spread.

Handout section 2.1.
:::

# Two moves, repeated

- Start: $k$ centres, e.g. $k$ random customers
- **Assign**: every point to its nearest centre
- **Move**: every centre to the **mean** of its points
- Repeat until a round changes nothing
- In code: `KMeans(n_clusters=4)`, then `labels_`, `cluster_centers_`

::: notes
This is the whole algorithm: two moves and a stopping rule. The move step puts
each centre at the mean of its points, which is where the "means" in the name
comes from. Notebook 1 writes
it in about ten lines of NumPy before using scikit-learn, precisely so that
nobody thinks there is more to it.

In code the full call notebook 1 uses is KMeans(n_clusters=4, n_init=10). The
parameter n_init comes back on slide 15: it is the number of different starts
scikit-learn tries. labels_ holds one group number per customer, and
cluster_centers_ the coordinates of the centres.

Handout section 2.2.
:::

# One number for how tight the groups are

- Draw a line from every point to the centre of its group
- **Within-cluster sum of squares** (WCSS): the squared lengths of those lines,
  added up
- Short lines, small WCSS: every point is close to its centre
- scikit-learn reports it as `inertia_`

::: notes
Make the picture before the formula, which comes on the next slide. A dashed
line from each customer to its centre; WCSS adds up the squared lengths. A
grouping whose lines are all short is a tight grouping.

Two things to say about it. First, there is no target in it: only the data and
the centres. Second, it is the same kind of quantity as the cost J of lessons 3
and 4, the number a method makes small, which is why the formula on the next
slide uses the letter J.

Handout section 2.2.
:::

# The within-cluster sum of squares, as a formula

$$J = \sum_{i=1}^{m} \| x_i - \mu_{c(i)} \|^2$$

::: notes
Read it in words, symbol by symbol. c(i) is the group of point i, and mu_j the
centre of group j. For every point x_i, from the first to the m-th, take its
group c(i) and that group's centre; measure the distance from
the point to the centre, square it, and add up.

The double bars mean ordinary straight-line length. Nothing else is hidden in
the formula: it is the sum of the squared dashed lines of the previous slide,
written down.

Handout section 2.2.
:::

# Six customers, by hand

![](kmeans_by_hand.png)

::: notes
Six customers with spend in hundreds of euros and visits per month, so the two
columns are already on comparable scales. A, B and C are small occasional
buyers; D, E and F big regular ones. The two centres start on A and B, a
deliberately poor choice, since both are in the same group.

Walk the four panels left to right and read the WCSS in each title.

Round 1, assign: C goes to A; D, E and F go to B, which is one visit higher.
D is at 7 squared plus 4 squared, 65, from B, against 74 from A. WCSS =
0 + 1 + 0 + 65 + 80 + 74 = 220.

Round 1, move: the BDEF centre goes to the mean of four points, (6.5, 5.25),
and the WCSS falls to 56.25. But B is now left far behind.

Round 2, assign: B is nearer the first centre and switches, WCSS 16.69. Round
2, move: the centres go to the middle of each triangle, WCSS 8/3 = 2.67. A third
round changes nothing, so it stops, at ABC and DEF.

Handout section 2.3 carries every distance.
:::

# Why it always stops

- **Assign** cannot raise WCSS: points only move to a nearer centre
- **Move** cannot raise it: the mean is the closest point to a group,
  in squared distance
- Finitely many ways to split $m$ points into $k$ groups
- So WCSS falls or stays until nothing changes: it **stops**
- Six customers: 220, 56.25, 16.69, 2.67

::: notes
This is the argument for why k-means always finishes, in four lines.

The second bullet needs one sentence. Take a single coordinate: the sum of
(x_i - mu) squared is a parabola in mu, and its lowest point is where its
derivative, minus 2 times the sum of (x_i - mu), is zero, which is exactly when
mu is the mean. So moving a centre to the mean can only shorten its lines.

The last bullet is the six-customer example again: the score never went up.
Then ask the room the question the next slide answers: stopping means nothing
changes any more. Does it mean the answer is the best grouping?

Handout section 2.4.
:::

# Same data, two starts

![](kmeans_two_starts.png)

::: notes
The same ten lines of code on all 2,000 customers, standardised, started from
two different random picks of four customers. White circles are where the
centres began, black paths where they travelled, crosses where they stopped.

Left: the centres travel to the four clouds, WCSS 374.1. Right: two centres end
up sharing the group at the bottom right, while a single centre covers both
groups on the left; WCSS 1,390.4, more than three times worse. And it stays
there: from that grouping, assigning and moving change nothing, so the
algorithm stops, although a better grouping exists.

That is the answer to the previous slide's question. Stopping is not finding.
Handout section 2.4.
:::

# The remedy: several starts, keep the best

- 30 random starts: 29 reach WCSS **374.1**, one stops at **1,390.4**
- Rare, but one run cannot tell you it was the unlucky one
- So: several starts, keep the lowest WCSS
- scikit-learn: `n_init=10`, plus smarter starts (**k-means++**)
- `KMeans` and the ten lines agree: 374.0919

::: notes
The two numbers come from notebook 1, which runs the hand-written k-means from
30 random starts. One in thirty is rare, but over a year of reruns it will
happen, and the WCSS of one run alone does not tell you which kind of run you
had.

The remedy costs only time: several starts, keep the best. scikit-learn does it
by default with n_init=10, and it also chooses the starting centres with
k-means++, which picks each new starting centre preferring points far from the
ones already picked, so that two centres rarely start inside one group.

The last bullet is a check, not a claim: notebook 1 prints the same WCSS to
four decimals from both. Handout section 2.4.
:::

# How many groups?

- $k$ is an **input**: the algorithm does not choose it
- WCSS always falls as $k$ grows: one group per point gives 0
- So "lowest WCSS" always picks the largest $k$ tried
- Instead, **the elbow**: where WCSS stops falling steeply
- And **the silhouette**: is each point nearer its own group than the next?

::: notes
Start from the third bullet, because it is the predictable mistake: choosing k
by the lowest WCSS. It sounds like what every lesson so far has done, picking
the lowest cost. But with one group per customer every line has length zero,
so WCSS would always choose the largest k you tried.

The elbow asks where adding a group stops paying: past the right number, an
extra centre only splits a group that was already fine. The silhouette asks a
different question, about each point; the next two slides build it on one
customer.

Handout section 2.5.
:::

# The silhouette of one point

![](silhouette_one_point.png)

::: notes
Back to the six customers, after k-means grouped them as ABC and DEF. Take
customer B.

The solid blue lines go to the other members of B's own group: lengths 1 to A
and the square root of 2, 1.414, to C. Their mean is a = 1.207.

The dotted red lines go to the members of the other group: the square roots of
65, 80 and 74, that is 8.062, 8.944 and 8.602. Their mean is b = 8.536.

When the solid lines are much shorter than the dotted ones, B sits well inside
its own group. The next slide turns that comparison into one number: here,
(8.536 - 1.207) / 8.536 = 0.859.

Handout section 2.5, worked example.
:::

# The silhouette, as a formula

$$s = \frac{b - a}{\max(a, b)}$$

::: notes
a is the point's mean distance to the other members of its own group, b its
mean distance to the members of the nearest other group.

Read the three cases off the formula. If a is much smaller than b, s is close
to 1: the point is well inside its group. If a and b are about equal, s is
close to 0: the point sits on a border. If a is larger than b, s is negative:
the point is closer, on average, to another group than to its own.

Dividing by the larger of the two keeps s between -1 and 1 whatever the units.
The silhouette of a whole grouping is the average over all points, and in code
it is silhouette_score(X, labels). Customer B: 0.859.

Handout section 2.5.
:::

# Elbow and silhouette, on 2,000 customers

![](elbow_silhouette.png)

::: notes
k-means on the 2,000 standardised customers for k from 1 to 8, from notebook 1.

Left, the elbow: WCSS falls steeply up to k = 4, the dashed line, and slowly
after. Right, the mean silhouette: highest at the same k = 4, with 0.690.

Point out why the silhouette is easier to act on: reading a bend takes
judgement, and two people can disagree about where it is; reading a maximum
does not. Both agree here because the data really is four round, well
separated clouds. Slide 35 shows a case where the silhouette points the wrong
way.

Handout section 2.5.
:::

# Checking against the truth: the adjusted Rand index

- Possible only here: the generator knows each customer's segment
- For every **pair**: do both groupings agree (together, or apart)?
- Share of pairs that agree: the Rand index
- **Adjusted Rand index** (ARI): random grouping about 0, identical 1
- In code: `adjusted_rand_score(truth, labels)`

::: notes
The ARI compares two groupings of the same points without caring what the
groups are called: k-means' cluster 0 does not have to be "window shoppers".
That is why it works on pairs, and pairs do not carry names.

Why adjust? Even a random grouping agrees about many pairs, because most pairs
are apart in both groupings. The adjustment subtracts what chance alone would
give, so 0 means "no better than random", and a grouping worse than random can
even score below 0.

Then the distinction to carry through the lesson. The silhouette is internal:
it needs only the data and the grouping, so Aurora could compute it. The ARI is
external: it needs the truth, which Aurora does not have.

Handout section 2.6.
:::

# k-means against the truth

![](kmeans_vs_truth.png)

::: notes
Left, the true segments, from the generator and never shown to k-means. Right,
k-means with k = 4, centres marked with crosses.

The ARI between the two is 0.985: k-means found the four segments almost
exactly. Ask the room to find the disagreements: a handful of customers where
two segments touch, coloured differently from their neighbours.

Then the predictable mistake. Having seen this, most conclude that k-means is a
reliable way to find groups. The reasoning is sound as far as it goes: on a few
round clouds of similar spread it is fast and close to the best possible. What
fails is the assumption under it, that groups are round, and the next part of
the lesson builds data where they are not.

Handout section 2.6.
:::

# Notebook 1, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| look first, 7 | `StandardScaler` → 2,000 customers, 2 columns |
| k-means by hand, 9-13 | NumPy only → WCSS 220, 56.25, 16.69, 2.67 |
| two starts, 14-15 | NumPy, then `KMeans` → 374.1 against 1,390.4 |
| how many groups, 16-19 | `KMeans`, `silhouette_score` → 0.690 at k = 4 |
| against the truth, 20-21 | `adjusted_rand_score` → 0.985 |

::: notes
k-means by hand, on Aurora's customers. Run
Notebooks/01_kmeans_by_hand.ipynb. 25 minutes.

Before section 3.1 runs, have them predict where the two centres will end up
from the start on A and B, and check after each round against the figure.

If time runs short, keep section 3.2, the two starts drawn side by side:
watching one start stop at a grouping more than three times worse is what turns
"k-means finds a local minimum" from a phrase into a fact.

If somebody finishes early: run k-means with k = 3 on the customers and see
which two segments are merged into one, then explain why those two.
:::

# 1,500 sessions, no labels

![](sessions_unlabelled.png)

::: notes
Aurora's security team logs two numbers per website session: how long it
lasted and how many pages it viewed. Some sessions are bots: scripts, not
shoppers.

Ask the room what they see. Not four separate clouds this time: a dense smudge
sitting inside a much larger, looser ring, both centred on the same place. A
script does the same thing every time, so the bots pile up in a tight spot;
people vary, so they spread out around it. retail_data.py makes 180 of the
1,500 sessions bots, 12%, with the same average as the humans.

The natural split is "close to the middle" against "far from it". Handout
section 3.1.
:::

# Why k-means must fail here

- With $k = 2$, the points equally far from the two centres form a
  **straight line**
- So k-means' two groups are always the two sides of a straight line
- No straight line separates a core from the ring around it
- Result: ARI **−0.046**, no better than chance
- Not bad luck with the start: more starts cannot fix it

::: notes
Draw it on the board: two centres, and the line of points equally far from
both, halfway between them and at right angles to the segment joining them.
Everything on one side goes to one centre, everything on the other to the
other.

So whatever k-means does, it cuts the picture with a straight line. The best it
can do here is cut it through the middle, which cuts the ring in half and puts
the bots with one half of the humans. Notebook 2 measures it: all 180 bots in
one cluster, together with 780 humans.

The ARI of -0.046 is from notebook 2. The point of the last bullet: this is a
wrong assumption, not a bad run, so the remedy of slide 15 does not apply.

Handout section 3.2.
:::

# Hierarchical clustering: a tree of merges

- Start: every point is a group of its own
- Merge the two **closest** groups; repeat until one is left
- The record is a tree, a **dendrogram**: a join's height = how far apart
- **Cut** it with a horizontal line: the cut chooses $k$
- In code: `AgglomerativeClustering(n_clusters=2)`

::: notes
A different idea from k-means: no centres to place, just repeated merging of
whatever is closest. One tree serves every k at once: cutting higher gives
fewer groups, cutting lower gives more.

"Closest" needs a rule once a group has more than one point, called the
linkage. scikit-learn's default, Ward linkage, merges the two groups whose union
adds the least to the within-cluster sum of squares of slide 10: the merge that
keeps the groups most compact. For two single points that is simply the two
nearest. Ward is what AgglomerativeClustering uses unless told otherwise
(linkage="ward").

Handout section 3.3.
:::

# Eight points, five merges, one cut

![](merge_tree.png)

::: notes
Eight points in three loose clumps. Left, the first five merges, numbered.
Right, the dendrogram of the same merges.

The first three merges join the three closest pairs: A with B at height 0.671,
which is simply their distance, the square root of 0.6 squared plus 0.3
squared; then D with E, then G with H. Then F and C join their neighbours.
After the fifth merge, at 1.333, three groups are left.

Now point at the long vertical stretch between 1.3 and 7.2 where nothing
merges. The next merge has to bridge a real gap, and the tree says so with a
jump in height. A long stretch with no merge is the tree's way of saying "here
is a natural number of groups". The dashed cut at 4 crosses three branches.

Handout section 3.3, worked example.
:::

# Ward on the sessions: the same failure

![](round_methods_fail.png)

::: notes
Hierarchical clustering with Ward linkage on the 1,500 sessions, cut at two
groups, next to k-means and the truth.

ARI -0.062: no better than k-means, for the same reason. Ward merges whatever
keeps the groups compact, which is what k-means rewards too, so both end with
two compact halves of the picture. Point at where each puts the bots: k-means
with the bottom of the ring, Ward with its right-hand side.

A different procedure, the same idea of what a group is. The other linkage
rules, merging by the closest pair or the farthest pair of points, do not
separate the bots either; the advanced lesson compares all four.

Handout section 3.3.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. The second half opens with a method that asks a different
question about each point: not "which centre is nearest?" but "is this point in
a crowded place?"
:::

# DBSCAN: is this point in a crowded place?

- Look at the disc of radius `eps` around each point
- **Core** point: at least `min_samples` points in its disc, **itself included**
- **Border** point: not core, but inside a core point's disc; it joins that group
- **Noise**: neither; it belongs to no group (label $-1$)
- A group is a chain of core points plus their border points: **no centres,
  no $k$**

::: notes
DBSCAN stands for density-based spatial clustering of applications with noise.
The name says what it does: it groups by density, how crowded a place is, and
it is allowed to call a point noise.

Two details to say aloud. scikit-learn counts the point itself in min_samples,
so min_samples = 4 means the point and three others. And nothing here mentions a
centre or a shape: a group can be a ring, a band, or any connected crowded
region, which is exactly what the sessions need. DBSCAN does not ask for k
either: the number of groups is however many chains there turn out to be.

In code: DBSCAN(eps=0.30, min_samples=10), with the noise labelled -1 in
labels_. Handout section 3.4.
:::

# Core, border and noise, drawn

![](dbscan_points.png)

::: notes
Twelve points, eps = 0.9 drawn as the disc radius, min_samples = 4.

The three shaded discs are one of each kind. The blue disc belongs to a core
point: at least four points inside it, counting itself. The teal disc belongs
to a border point: only three points in its disc, too few to be core, but it
lies inside a core point's disc, so it joins that group. The rust disc belongs
to a noise point: nothing but itself within 0.9.

Ask the room: the rust point is barely farther than 0.9 from the group. Why is
it not a border point? Because nothing but itself lies within 0.9 of it, so it
is inside no core point's disc. Its nearest neighbour, at 1.03, is only a
border point, and border points cannot extend a group anyway.

Four core, four border, four noise: one group. Handout section 3.4.
:::

# Choosing `eps`: the k-distance plot

- Too small: even the bot core breaks into noise
- Too large: the ring reaches the core, all one group
- Sort every point's distance to its 10th neighbour (10 = `min_samples`)
- Put `eps` where the sorted curve **bends upward**
- In code: `NearestNeighbors(n_neighbors=10)`, then `kneighbors`

::: notes
eps is the parameter that matters most, and these two failure modes are why:
too small and nothing is crowded enough, too large and everything is.

The k-distance plot turns the choice into something you can read. Sorting the
10th-neighbour distances puts the crowded points on the left, with short
distances, and the sparse ones on the right. The bend is where crowded gives
way to sparse.

One detail for the code: kneighbors counts each point as its own first
neighbour, so the 10th column is the 9th other point. That matches DBSCAN,
which also counts the point itself in min_samples.

Handout section 3.5.
:::

# The bend in the curve

![](k_distance_plot.png)

::: notes
The 1,500 sessions sorted by the distance to their 10th nearest neighbour, in
standardised units, from notebook 2.

The curve stays low and flat for most sessions and turns sharply upward at the
right-hand end. The dashed line at 0.30 sits just above the flat part: only 65
of the 1,500 sessions have their 10th neighbour farther away than that, and
notebook 2 prints the count.

Say honestly that reading a bend takes judgement, exactly like the elbow of
slide 19. The plot narrows the choice; it does not make it for you.

Handout section 3.5.
:::

# DBSCAN on the sessions

![](dbscan_result.png)

::: notes
DBSCAN with eps = 0.30 and min_samples = 10, next to the truth.

ARI 0.941, from notebook 2. All 180 bots form one group. 1,303 of the 1,320
humans form the other. 4 humans are placed with the bots, and 13 sessions, the
black points, are noise.

Compare with slide 27, where k-means and Ward scored below zero on the same
data. The difference is not tuning: it is a different definition of what a
group is, crowded and connected rather than compact. Handout section 3.6.
:::

# The 13 noise sessions

- All 13 are **humans**
- They sit where the data is thin: 2 in the gap around the bot core,
  11 on the ring's sparse outer edge
- DBSCAN's answer for them: **not confidently either**
- For a security analyst, more useful than a forced "bot" or "human"

::: notes
Notebook 2 checks each noise session against the truth, which only synthetic
data allows: all 13 are humans. It also measures where they are, by their
distance from the middle of the bot core: two lie between the core and the
ring, eleven on the outer edge, beyond where 95% of humans are.

The point to make: noise is an output, not a failure. A k-means model would
have forced each of those 13 into a group. DBSCAN says it does not know, and in
a security pipeline "I do not know" can be sent to a person instead of to an
automatic block. The supplementary reading of this lesson is about what such a
label costs a real person.

Handout section 3.6.
:::

# The silhouette prefers the wrong answer

| grouping | silhouette (no truth needed) | ARI (needs the truth) |
|---|---|---|
| k-means, $k = 2$ | 0.401 | −0.046 |
| DBSCAN | −0.106 | 0.941 |

::: notes
Aurora has no truth, so it could not compute the ARI column. It could compute
the silhouette, and the silhouette prefers k-means' useless split.

Why. Two compact halves are exactly what the silhouette rewards, as slide 18's
formula says: points close to their own group, far from the next. A ring is not
compact. Notebook 2 measures it: 99.2% of the ring's points have a negative
silhouette, because on average they are closer to the bot core in the middle
than to the rest of their own ring, most of which lies across the circle.

The predictable mistake: with no labels, the silhouette is the only score
available, so choosing the grouping that scores highest feels natural. But it
measures compactness, whatever the shape of the data. Only looking at the
data, or knowing what the groups should mean, can tell you compactness is the
wrong question. Handout section 3.7.
:::

# Choosing among the three

| method | a group is | needs $k$? | fails when |
|---|---|---|---|
| k-means | the points nearest one centre | yes | groups are not round |
| tree of merges, Ward | a branch of the tree | the cut chooses it | as k-means |
| DBSCAN | a crowded connected region | no | groups differ a lot in density |

::: notes
The last column is the one to remember. k-means and Ward share a failure,
because they share a definition of a group. DBSCAN has its own: one eps has to
suit every group, so a dense group next to a sparse one cannot both be found
with a single disc size.

When is the tree worth it? When the tree itself is informative: it shows the
groups at every scale at once, and on a few hundred points it can be read by
eye.

Handout section 3.8.
:::

# Notebook 2, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| look first, 23 | `StandardScaler` → 1,500 sessions, 180 bots |
| k-means, 24 | `KMeans`, `adjusted_rand_score` → ARI −0.046 |
| tree of merges, 25-27 | `AgglomerativeClustering` → ARI −0.062 |
| DBSCAN drawn, 29-30 | `DBSCAN` → 4 core, 4 border, 4 noise |
| `eps`, 31-32 | `NearestNeighbors` → 65 sessions beyond 0.30 |
| DBSCAN, 33-34 | `DBSCAN` → ARI 0.941, 13 noise |
| silhouette, 35 | `silhouette_score`, `silhouette_samples` → 0.401 against −0.106 |

::: notes
Merge trees and DBSCAN, on Aurora's sessions. Run
Notebooks/02_merge_trees_and_dbscan.ipynb. 20 minutes.

Before section 3, have them predict, from the scatter alone, what k-means with
k = 2 will do. Before section 5.2, have them predict whether DBSCAN will put
any bot among the noise.

If time runs short, keep section 6: the silhouette preferring the wrong answer
is the lesson's second warning, after the bad start, and the one easiest to
forget.

If somebody finishes early: rerun DBSCAN with eps = 0.20 and with eps = 0.50 and
explain each result with the two failure modes of slide 31.
:::

# Eight columns, not eight independent numbers

![](feature_correlation.png)

::: notes
Aurora's account table: 2,000 accounts, 8 columns, from annual spend to return
rate. The figure is the correlation between every pair of columns, standardised;
dark red is strongly positive, dark blue strongly negative.

Point at annual spend and average basket value: 0.859. Several columns move
together, which means fewer than eight numbers are really varying.
retail_data.py knows why: three hidden tendencies, how much a customer likes to
spend, how engaged they are, how sensitive to price, drive all eight columns.
Nobody tells PCA that.

Handout section 4.1.
:::

# The direction with the widest shadow

- Spend and basket value, standardised: correlation 0.859
- Drop each point onto a line at a right angle: its **shadow**
- One number per account sums up two columns
- Best line: shadows **spread most** (largest variance)
- That is the **first principal component**

::: notes
Principal component analysis, PCA, built on two columns, where it can be drawn.
The second principal component is the direction at right angles to the first.

The key sentence is the fourth bullet. A summary that keeps the accounts apart
keeps the information that tells them apart, so we want the direction where the
shadows are most spread out. A direction where all shadows land in the same
place would say nothing about anyone.

Standardised means mean 0 and standard deviation 1 per column, so the variance
of each column on its own is 1. The next slide shows the shadows on two
directions, and their spread for every direction.

Handout section 4.2.
:::

# The spread of the shadows

![](pca_shadow.png)

::: notes
Left: sixty accounts and their shadows on the spend axis, 0 degrees. Middle:
the same accounts on the diagonal, 45 degrees. The shadows on the diagonal are
visibly more spread out: variance 1.86 against 1.00.

Right: the variance of the shadows for every direction from 0 to 180 degrees.
The peak, at 45 degrees along the band, is the first component. The lowest
point, at 135 degrees across the band, is the second. On the spend axis the
variance is exactly 1, as it must be for a standardised column.

The next slide gives the formula of that curve. Handout section 4.2.
:::

# Why the peak is $1 + r$

$$\operatorname{Var}(z_\theta) = 1 + r \sin 2\theta$$

::: notes
z_theta is the shadow at angle theta. This is the variance of a sum, from
probability, nothing more; write the middle step on the board. The shadow is
z_theta = x1 cos(theta) + x2 sin(theta). Its variance is cos squared times
the variance of x1, which is 1, plus sin squared times the variance of x2, also
1, plus twice sin cos times their covariance, which for standardised columns is
the correlation r.

cos squared plus sin squared is 1, and 2 sin cos is sin(2 theta). So the
largest value is 1 + r, at 45 degrees, and the smallest 1 - r, at 135 degrees:
with r = 0.859, that is 1.859 and 0.141, the curve of the previous slide.

The two components share the total variance, 1 + 1 = 2: the first carries
1.859 / 2, 93% of it. One number per account, its shadow on the diagonal,
describes both columns almost completely. Handout section 4.2.
:::

# Any number of columns: stated, not derived

- 8 columns: no angle to sweep
- **The components are the eigenvectors of the covariance matrix**
- Each **eigenvalue**: the variance of the shadows along it
- Two columns: $1 + r$ and $1 - r$, on the diagonals, as swept
- In code: `PCA().fit(X)`, shares in `explained_variance_ratio_`

::: notes
State it, do not derive it: the derivation is in the advanced lesson. What
matters is the meaning: sorted by eigenvalue, largest first, the eigenvectors
are the first, second, third component, each at right angles to the ones
before, and each eigenvalue says how much of the spread its component carries.

The fourth bullet is the check that the statement agrees with what we just did
by hand. The covariance matrix of two standardised columns has 1 on the
diagonal and r off it; its eigenvalues are 1 + r and 1 - r. Notebook 3 checks
it a third way with scikit-learn: shares 0.9297 and 0.0703, which are 1.859 / 2
and 0.141 / 2.

Handout section 4.3.
:::

# How many components: the scree plot

![](pca_scree.png)

::: notes
PCA on all eight standardised columns. Bars: each component's share of the
variance. Line: the share of the first one, two, three components together.

Three components carry 93.5% of the variance; the fourth adds only 2.4%. The
scree plot has found, from the data alone, the number retail_data.py built in:
three hidden tendencies. Read it like the elbow of slide 19: keep the
components before the drop.

The predictable mistake: keeping components until a round share, 95% or 99%,
regardless of the plot. Keeping more variance sounds like losing less. But past
the drop, the extra components carry each column's own noise, not structure,
and the next part of the lesson needs exactly that separation.

Handout section 4.4.
:::

# Fraud that looks ordinary on every column

- 40 of 2,000 accounts (2%) taken over by fraudsters
- Each value drawn **inside** the genuine 5th-95th percentile range,
  column by column: notebook 3 finds 100% inside
- So no one-column rule ("spend above X") finds them
- What fraud cannot copy: **how the columns move together**

::: notes
This is how retail_data.py built the anomalies, and it is the whole difficulty.
Every single value of a fraudulent account is ordinary. A single-column check,
the kind a business rule would use, has nothing to find.

But a genuine account with a high spend also has a large basket, because one
tendency drives both. A fraudster drawing spend and basket separately breaks
that link. The next slide shows it on the two columns we already know.

Handout section 5.1.
:::

# Off the band

![](anomalies_off_the_band.png)

::: notes
The two columns of slide 39, annual spend and basket value, with the 40 anomalies
in red.

Look first at the margins, top and right: each column on its own. The red bars
sit inside the blue ones: on one column at a time, the anomalies look ordinary.

Then look at the scatter. Genuine accounts follow the band; many anomalies fall
off it, with a high spend and a small basket, or the other way round. The
anomaly is in the pair, not in either column.

Handout section 5.1.
:::

# Rebuild each account from its shadow

![](reconstruction_two_columns.png)

::: notes
Keep only the first component, and rebuild each account from its shadow alone:
the reconstruction, the gold square, is the point on the line where the shadow
fell. In code, pca.inverse_transform(pca.transform(X)): transform computes the
shadows, inverse_transform puts them back in the original columns.

Two accounts. The genuine one is far from the middle, 4.70, yet its
reconstruction lands almost on it: error 0.01, because it lies on the band. The
anomaly is closer to the middle, 1.96, and its reconstruction is far away:
error 3.75, because the line cannot describe it.

Read the title aloud: far from the middle is not the same as far from the
pattern. Handout section 5.2.
:::

# The reconstruction error, as a formula

$$e = \| x - \hat{x} \|^2$$

::: notes
x is an account, x-hat its reconstruction from the kept components. The
formula is the squared length of the dashed line on the previous slide.

In eight columns the line becomes the space spanned by the three components the
scree plot kept. An account is suspicious when those three components cannot
rebuild it: the detector flags the accounts with the largest e.

Handout section 5.2.
:::

# 37 of 40

![](reconstruction_error.png)

::: notes
Notebook 3 fits PCA with three components to the eight standardised columns,
rebuilds every account, and flags the 2% with the largest reconstruction error:
40 accounts, because Aurora's fraud team estimates that about 2% of accounts are
compromised.

37 of the 40 anomalies are among the 40 flagged. Point at how few red bars lie
left of the dashed threshold: the mean error is 4.676 for anomalies against
0.431 for genuine accounts.

This is the number of the lesson: 37 of 40, using only how eight ordinary-looking
columns relate to each other. Ask them to write it down. Handout section 5.3.
:::

# 37 is one batch

- Regenerate the table at 7 other seeds: same process, different accounts
- Same procedure: 3 components, flag the worst-rebuilt 2%
- Caught: from **32 to 38** of 40, median 36
- Carry 37 as the memorable figure, and the range as what it means: about
  nine in ten
- Lesson 5's point, with no $y$: one run is one draw

::: notes
37 is this quarter's batch of accounts. Notebook 3 regenerates the table with
seven other seeds and repeats the identical procedure. The count caught runs
from 32 to 38 of 40, with a median of 36.

So the detector finds about nine anomalies in ten, and which nine depends on
where this quarter's fraudsters happened to land. Quoting "37 of 40" as a
property of the method would be quoting one quarter's luck: lesson 5 made the
same point about a single train/test split, and it holds with no target at all.

Handout section 5.3.
:::

# Why not just plot it and look?

![](pca_tsne_anomalies.png)

::: notes
The natural move: squash the eight columns into two and look for the anomalies
at the edge of the picture. Left, PCA's first two components. Right, t-SNE,
defined on the next slide. Anomalies in red.

Ask the room to find red points at the edges. There are hardly any. On the PCA
picture the anomalies sit, on average, 1.30 from the middle, against 2.20 for
genuine accounts: closer, not farther. Flagging the edge would catch the wrong
accounts.

Why: a fraudulent account's values were drawn independently and never extreme,
so along the directions where genuine accounts spread out together, its values
partly cancel and its shadow lands near the middle. What gives it away is how
badly three components rebuild it, and a 2-D picture has thrown that away.

This is the predictable mistake the exercise builds on. Handout section 5.4.
:::

# t-SNE: for looking, not for measuring

- **t-SNE** (t-distributed stochastic neighbour embedding): made for pictures
- In 2-D, **close neighbours stay close**
- Large distances are not kept: gaps **mean little**
- No `transform`: a new account means running it all again
- In code: `TSNE(n_components=2)`

::: notes
t-SNE is often more striking than PCA at making separate groups visible,
because that is what it is built for. The price is in the third and fourth
bullets. Two groups drawn far apart in a t-SNE picture are not necessarily more
different than two drawn close together. And PCA's transform can place a new
account into an existing picture, while t-SNE has to be run again on everything.

On the fraud data it shares PCA's blind spot, as the previous slide showed. Use
it to look at data, never as the measurement. How t-SNE works inside is outside
this course; the further reading of the handout points to the original paper.

Handout section 5.4.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| eight columns, 38 | `StandardScaler` → correlation 0.859 |
| the widest shadow, 39-42 | NumPy, then `PCA` → 1.859 and 0.141; shares 0.9297 and 0.0703 |
| scree, 43 | `PCA().fit`, `explained_variance_ratio_` → 93.5% in 3 components |
| fraud, 44-47 | `PCA(n_components=1)`, `inverse_transform` → errors 0.01 and 3.75 |
| 37 of 40, 48-49 | `PCA(n_components=3)` → 37 of 40; 32 to 38 over 8 batches |
| 2-D pictures, 50-51 | `PCA(n_components=2)`, `TSNE` → 1.30 against 2.20 |

::: notes
PCA and anomalies, on Aurora's account table. Run
Notebooks/03_pca_and_anomalies.ipynb. 20 minutes.

Before section 3 prints, have them guess the angle with the widest shadow from
the scatter alone. Before section 7, have them predict where the red points
will sit on the 2-D pictures, and why.

If time runs short, keep sections 6 and 7 together: the reconstruction error
catching the anomalies and the 2-D pictures missing them are one argument, and
half of it is not worth much.

If somebody finishes early: flag 1% and 5% instead of 2%, and report how many
anomalies are caught and how many genuine accounts are flagged in each case.
:::

# What to take away

- k-means always stops, not always at the best: **374.1** against **1,390.4**
- A core inside a ring defeats k-means and Ward; DBSCAN scores **0.941**
- The silhouette measures compactness: it preferred the wrong answer
- **37 of 40** disguised accounts caught by reconstruction error, invisible
  to a 2-D scatter

::: notes
The number to carry out of the room is the last one: 37 of 40 accounts caught
using nothing but how eight ordinary-looking columns relate to each other, on a
problem the natural 2-D picture gets backwards.

Recall the course's other carry-home numbers so far: 77% accuracy on coin-flip
labels from leakage; 94 of 128 imputed rows borrowing from the test set; 365
against billions for a penalised coefficient; 96.20% accuracy from a model that
finds no failure; 0.885 to 1.000 from 200 splits of the same data; 0.613 to
0.947 for a linear model against a kernel; 54% of importance on noise. Now 37 of
40.

The habit to keep: before trusting an unlabelled result, ask what the measure
in front of you is able to see. A silhouette cannot see a wrong idea of what a
group is, and a 2-D picture cannot see what the discarded dimensions held.
Handout summary.
:::

# Homework

- **Exercise 8**, discussed at the start of lesson 9, **Friday 4 December**
- `Exercises/08_unsupervised_learning.md`
- Two account books: predict, **before plotting**, where a 2-D plot will
  find the fraud and where it will not

::: notes
Set it explicitly and say when it comes back: the first ten minutes of lesson 9,
Friday 4 December. Nothing is collected; at the exam one of the ten exercises
is drawn and the student talks through their own notebook.

The exercise gives two account books from one fictional bank. On one, flagging
the edge of a 2-D plot works almost as well as reconstruction error; on the
other it catches nothing. The skill is to read how each book's fraud was built
well enough to predict which is which before plotting, so insist: write the
prediction down first.
:::

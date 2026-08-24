---
title: "Lesson 7 — Trees and Ensembles"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "6 November 2026"
---

# Agenda

- A fourth family, built on a threshold
- Decision trees, and why one alone is mediocre
- Depth as the bias-variance dial, again
- Bagging and random forests
- Feature importance, and its limits
- Gradient boosting
- Choosing among trees, forests and boosting

::: notes
Say what changes and what doesn't. Lessons 3 and 4 fit a line. Lesson 6
offered three ways of bending a boundary a line cannot draw. Today adds a
fourth primitive — a threshold, not a distance or an inner product — and
then spends most of the three hours on a different question: not "does a
tree work", but "what happens when you build more than one".

Tell them the punchline early, without the number yet: a single tree is one
of the few models in this course a person with no mathematics background can
read end to end, and it is also, on its own, a mediocre model. Ensembles of
trees are what most tabular-data competitions are still won with, and the
reason is the subject of today.

Handout section 1 sets this framing out in full.
:::

# Exercise 6 returned

- Marks were for the **defence of which family**, not for the winning score
- The recurring gap: a strong accuracy number with no check of *why* the
  assumption behind it held
- Today's models come with the same obligation

::: notes
Hand back exercise 6 quickly — a sentence on what the class did well (most
picked a defensible family) and a sentence on the common gap (several reports
quoted a cross-validated score without ever checking whether the model's
assumption, such as within-class independence, actually held on that data).

Link forward explicitly: today introduces a family with its own convenient
assumption — that the columns you feed a random forest are all worth
listening to — and by the end of the lesson you will have measured exactly
how badly that assumption can fail.
:::

# 1,200 loan applicants, two figures each

- Income in thousands of euros a year, debt-to-income ratio
- Underwriting is a **checklist**, not a formula: an income floor, a debt
  ceiling, and two "stressed" combinations in between
- **7%** of labels are deliberately flipped; **38.7%** of applicants
  defaulted, so the majority baseline is **0.613**

::: notes
Describe the physical logic before the picture. A bank's rule here is closer
to a checklist than a smooth function: almost nobody is approved below an
income floor or above a debt ceiling, and there are two further regions where
income and debt are each unremarkable on their own but risky in combination.

Ask the room what shape that logic implies before you show the scatter. The
useful answer is: not one connected region, so no single straight line can
separate the classes — a different failure from lesson 6's disc-in-a-ring,
but a failure of the same kind.

Handout section 1.1.
:::

# Four disconnected risky regions

![](loan_scatter.png)

::: notes
Let it sit for a moment before pointing anything out. Then trace the shape
out loud: a strip of low income on one edge, a strip of high debt on
another, and two disconnected rectangles in the interior where neither
reading alone is remarkable.

Ask for a straight line that separates repaid from defaulted, exactly as you
did with the pumps in lesson 6. This time there is a difference worth
naming: unlike the pumps, the class you would have to isolate is not even
one connected region, so "no line separates them" is true for a different
reason than a ring around a disc.

Handout section 1.1.
:::

# No straight line separates them

| Model | Cross-validated accuracy |
|---|---|
| Always predict the majority class | 0.613 |
| Logistic regression | **0.748 ± 0.030** |

::: notes
Unlike lesson 6's pumps, logistic regression clears the baseline by a wide
margin here — it has learned something. Three of the four risky regions are
monotonic in at least one feature, and a linear boundary handles monotonic
effects reasonably well.

What it cannot do is wall off the two interior islands without cutting into
healthy territory everywhere else. Ask the room what kind of boundary could
close that gap, and let a few guesses land before moving on — a threshold
rule, applied more than once, is exactly the answer the rest of the lesson
builds.

Handout section 1.1.
:::

# The whole algorithm, in one sentence

- Find the single (feature, threshold) split that makes the two resulting
  groups **as pure as possible**
- Apply it. Repeat on each group, independently
- Stop when a stopping rule is met
- Nothing is estimated — no coefficient, no gradient with respect to a
  parameter vector
- **Greedy**: whichever split helps most *right now*, with no lookahead

::: notes
Read the algorithm as one sentence and let it sit — it really is that short.
Contrast it explicitly with lessons 3 and 4: no cost function is minimised
over a continuous parameter, there is no gradient step. A tree is a sequence
of yes/no questions, chosen one at a time.

The word to stress is "greedy". At every step the tree takes whichever split
helps most immediately, with no mechanism for looking two levels ahead. Tell
them to hold onto that word — it is what explains, later this hour, why a
shallow tree can miss the interior islands entirely even though the
information to find them is right there in the data.

Handout section 2.1.
:::

# How pure is a group of examples?

- A leaf that is all one class needs no more questions
- A leaf split evenly between classes is the worst case — a coin flip
- **Impurity** is a number that captures this, so "as pure as possible" has
  a target to maximise against

::: notes
Before any symbol appears, ask the room what "pure" should mean for a group
of examples. Most will land close to the right answer on their own: a group
where everyone shares a label needs nothing further, and a fifty-fifty split
is as bad as it gets.

Say plainly that the standard choice for classification is called the Gini
impurity, and that it is not a probability of misclassification — it is the
probability that two examples drawn at random with replacement from the
group, and labelled according to the group's own class frequencies, would
disagree.

Handout section 2.2.
:::

# Gini impurity

$$G = 1 - \sum_{c=1}^{C} p_c^2$$

::: notes
$p_c$ is the fraction of the group belonging to class $c$, across $C$
classes. Do not derive anything here — state what it does: subtract, from
one, the chance two examples share a class if drawn independently from the
group's own frequencies. A pure group scores zero; a maximally mixed one
scores as high as the formula allows.

Give them a moment to notice the formula asks nothing about *which* class is
common, only about how mixed the group is — that symmetry matters later,
when a split is chosen purely by how much it reduces this number.

Handout section 2.2 has the full definition.
:::

# For two classes, one number

- $G = 2p(1-p)$, where $p$ is the fraction of one class
- **0** when the group is pure ($p = 0$ or $p = 1$)
- **0.5**, the maximum, at $p = 0.5$
- On the full 1,200 applicants, $p = 0.387$ gives $G = 0.474$

::: notes
Two classes is the case that matters for this dataset, and the formula
collapses to something with an obvious shape: zero at the extremes, a single
peak at fifty-fifty. Ask the room to sanity-check the peak location before
you confirm it — it is the point where a random guess is least useful.

Give them the concrete number: on the full loan dataset, 38.7% defaulted, so
p = 0.387 and G(parent) = 2 × 0.387 × 0.613 = 0.474. That is the impurity
every candidate root split is measured against.

Handout section 2.2.
:::

# The gain a split buys

- A candidate split divides a group into a left and a right child
- It is worth taking if the children are, **on average, purer** than the
  parent was
- "On average" is weighted by how many examples land in each child — a
  split that purifies nine points and dumps one troublemaker into its own
  leaf is not free

::: notes
Set up the idea in words before the formula: a split is only useful if the
two resulting groups are, taken together, purer than the one group was. The
weighting matters because a split that isolates one awkward point while
leaving the rest exactly as mixed as before should not score as well as one
that improves the picture broadly.

Ask what could go wrong if you *didn't* weight by group size — the answer is
that a tree would happily carve off tiny, perfectly pure single-point groups
at every step, which is exactly the overfitting behaviour section 3 measures
in a few minutes.

Handout section 2.2.
:::

# The reduction in impurity

$$\Delta G = G(\text{parent}) - \left(\frac{m_L}{m_P}\, G(\text{left}) + \frac{m_R}{m_P}\, G(\text{right})\right)$$

::: notes
$m_L$ and $m_R$ are the sizes of the two children, $m_P = m_L + m_R$. Read it
as: what the parent's impurity was, minus what is left after the split,
weighted by how the group was divided. The tree keeps whichever candidate
split makes $\Delta G$ largest.

No derivation needed here — the formula is exactly the weighted comparison
the previous slide described in words. Handout section 2.2 has it worked
through on the loan data's actual root split.
:::

# Scanning for the best split

- For a continuous feature, scan **every midpoint** between adjacent sorted
  values as a candidate threshold
- Do this for **every feature**; keep the best pair
- Repeat the search inside each child
- **Greedy**: $\Delta G$ is judged one split at a time, never jointly

::: notes
This is the exhaustive-search part of the algorithm, and it is worth being
concrete about the cost: every feature, every possible threshold, at every
node, all the way down. It is brute force, and on a dataset this size it is
cheap.

The last bullet is the one to slow down on. A feature that only helps in
combination with a second split further down the tree can be — and in this
dataset's two interior islands, is — invisible to a shallow search, because
$\Delta G$ is never evaluated for two splits together.

Handout section 2.2.
:::

# Worked example: the root split

- $p = 0.387$ defaulted, so the parent group's impurity is **0.474**
- Exhaustive search finds `debt_ratio <= 0.82` as the best root split
- Recognisably the **debt ceiling** the data was built with

::: notes
Walk the numbers once: parent impurity 0.474, and the winning split is the
debt ceiling, found by brute-force search with no hint about how the data
was generated. That agreement with the known generating rule is the first
piece of evidence that the algorithm is doing something sensible.

Ask the room why finding the debt ceiling by blind search, with no
knowledge of how the data was built, is a stronger check than it might
first appear — the answer is that it rules out the split being an artefact
of how the search was coded, since nothing in the algorithm knows the
generating rule.

Handout section 2.2.
:::

# Checked against scikit-learn

- The from-scratch implementation agrees with scikit-learn's
  `DecisionTreeClassifier` to the **fourth decimal place** of resulting
  accuracy, at every depth tested
- Not a coincidence: both search the same space by the same greedy rule
- From here on, the notebooks use scikit-learn's implementation — the same
  algorithm, compiled, not a different one

::: notes
This is the check notebook 1 runs live, and it matters for trust rather
than for novelty: once the from-scratch version is shown to agree with
scikit-learn exactly, every later result in the lesson — obtained with
scikit-learn — inherits that confidence.

Point out the "Try this" box in the handout: computing Gini impurity for
the income-floor subgroup alone and comparing it against the full dataset's
0.474 is a two-line exercise that makes the impurity number concrete rather
than abstract.

Handout section 2.2, and the "Try this" box that follows it.
:::

# What greedy misses

- The search never asks "which pair of splits helps most together" — only
  "which single split helps most right now"
- A feature that only pays off **in combination** with a later split can be
  invisible to a shallow tree
- Exactly what happens to this dataset's two interior "stressed" islands —
  covered next

::: notes
Close the Gini segment with the limitation, because it sets up everything
that follows. Ask the room: if the debt ceiling and the income floor are the
two easiest wins, what happens to the two interior islands, which need *two*
conditions at once to identify?

The honest answer is that a shallow tree simply cannot see them — not
because the information isn't there, but because greedy search only ever
commits to the best split available at each individual step. Depth is what
buys the tree enough steps to eventually stumble onto them, which is exactly
where the next segment goes.

Handout sections 2.2 and 3.
:::

# Depth is the bias-variance dial

- `max_depth`: how many questions a tree may ask before a leaf must stop and
  vote by majority
- **Shallow tree** — few questions, only the coarsest structure. High bias,
  low variance
- **Deep tree** — enough questions to isolate almost any subset, down to
  single points. At full depth, training accuracy is **exactly 1.000 on any
  dataset**, including one with no structure at all

::: notes
Name the dial explicitly and connect it to lesson 6: k in k-nearest
neighbours traded bias against variance directly, and depth is a tree's
version of the same knob.

The training-accuracy-1.000 claim is the one to make them justify. Ask why
an unconstrained tree gets every training point right, always, on any
data — the answer is that splitting never has to stop until every leaf
holds one example, which it then "predicts" perfectly by definition. That
number demonstrates the model class can memorise, not that anything has
been learned — the same reading lesson 6 gave k = 1's training accuracy.

Handout section 3.
:::

# Measured, at seven depths

| `max_depth` | Training accuracy | Cross-validated accuracy |
|---|---|---|
| 2 | 0.840 | 0.836 ± 0.024 |
| 4 | 0.843 | 0.827 ± 0.025 |
| 6 | 0.898 | 0.866 ± 0.016 |
| **8** | 0.951 | **0.882 ± 0.010** |
| 12 | 0.986 | 0.862 ± 0.018 |
| 16 | 0.993 | 0.852 ± 0.021 |
| unconstrained | **1.000** | 0.852 ± 0.021 |

::: notes
Read the two columns against each other, never in isolation, exactly as
lesson 5 taught. Training accuracy climbs monotonically towards 1.000.
Cross-validated accuracy rises, peaks at depth 8, and then falls back — even
though the model keeps getting strictly more flexible.

Ask which column they would have picked a tree by, if they had only seen
the training column. The far right, the unconstrained tree, looks unbeatable
on that evidence and is in fact the worst honest model on this table.

The unconstrained tree loses **3 points** of cross-validated accuracy
relative to the depth-8 peak, for a training score nobody should have
trusted in the first place. Handout section 3.
:::

# The curve bends downward

![](depth_curve.png)

::: notes
Point at the two curves in order: training climbing steadily towards the
top, cross-validated rising and then turning over. The peak is where the
tree is deep enough to isolate both stressed islands and shallow enough to
leave the flipped labels as errors rather than as leaves of their own.

Say plainly what past the peak is buying: every extra split is spent
fitting some of the 7% of labels this dataset deliberately flips, and a
flipped label has no pattern to learn. A tree that classifies it "correctly"
on training data has memorised it, not generalised.

This is the second time this course has shown a curve bend this way — lesson
3's Lasso penalty was the first. Handout section 3.
:::

# The same rule, three depths

![](tree_boundaries.png)

::: notes
Left to right: at depth 2 the tree affords only the income floor and the
debt ceiling — the two biggest, easiest-to-find regions — and neither
interior island is visible at all. At depth 8 both islands appear as clean
rectangles, close to the rule that generated the data. At full depth the
boundary has grown a fringe of tiny rectangles chasing individual flipped
labels.

Make the point explicit: the full-depth boundary is not a smoother version
of the depth-8 boundary. It is a noisier one, built out of the same
mechanism that gave the unconstrained tree its perfect training score.

Ask the room which of the three they would trust on a loan applicant they
had never seen. Handout section 3.
:::

# Reading a tree, and the knob that keeps it small

- `max_depth` grows the tree **breadth-first** — every leaf at level $d$
  splits before any leaf reaches $d+1$, useful or not
- `max_leaf_nodes` grows it **best-first** — always expanding whichever leaf
  offers the largest $\Delta G$
- With `max_leaf_nodes = 9`: **90.8%** training accuracy, every threshold
  recognisable — 28 the income floor, 0.82 the debt ceiling

::: notes
This is the readability half of the story. For a tree meant to be read
rather than merely scored, best-first growth is the better knob: a small
budget of splits goes to the ones that matter most, instead of being spent
uniformly across every branch regardless of whether it's worth it.

The nine-leaf tree is worth reading aloud as a sequence of if/else
statements — that is the entire model, and it is the thing k-nearest
neighbours and support vector machines cannot offer at all. For a decision
that has to be justified to the person it affects, that can matter as much
as accuracy.

Handout section 4.
:::

# A tree that doesn't stay still

- Two trees at `max_depth = 6`, fit to independent 65% resamples of the
  same 1,200 applicants
- The debt ceiling — the strongest split — agrees to within **0.001**
- The trees grow **31 and 33 leaves**, and **disagree on 11.7%** of
  predictions on the full dataset
- The strongest split is stable; the many weaker ones are not, and they
  decide most of a tree's shape

::: notes
This is the instability that ensembles turn into raw material. The
strongest split barely moves under resampling, because the evidence behind
it — 1,200 points supporting the debt ceiling — is overwhelming. Everything
weaker than that is supported by far fewer points, and those splits are
what actually determine a tree's shape.

Ask the room what 11.7% disagreement between two trees fit to almost the
same data implies about trusting any single tree's exact boundary. The
answer is the whole of the next segment: a single tree's instability looks
like a flaw, and it is about to become a strength.

Handout section 4.
:::

# Notebook 1, live

- Decision trees from scratch, checked against scikit-learn
- Depth swept, the two boundaries drawn, and the instability measured

::: notes
Run Notebooks/01_decision_trees_from_scratch.ipynb. Twenty minutes.

Have them watch the from-scratch implementation confirm the debt-ceiling
root split before scikit-learn's version ever runs — the point is that the
agreement is not assumed, it is checked, at every depth.

The cell worth protecting if time is short is the two-tree resampling
comparison at the end: printing the 11.7% disagreement figure themselves
lands harder than reading it off a slide.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. The next segment is where the lesson's real content starts —
one tree's instability, turned into an ensemble's strength — so they need to
come back with that framing fresh.
:::

# Bagging: averaging away the instability

- If a tree's mistakes are somewhat random — different across resamples, as
  just measured — **many trees, averaged**, should partly cancel them out
- Draw a **bootstrap sample**: $m$ rows chosen **with replacement** from
  the $m$ training rows
- Fit a tree to it. Repeat `n_estimators` times; predict by **majority
  vote**
- The tree algorithm itself is unchanged — only what surrounds it is new

::: notes
State the logic before the name. Section 4 just measured that two trees on
overlapping resamples disagree on more than a tenth of predictions. If those
disagreements are close to random, averaging many such trees should cancel
some of them out — that is the entire idea behind bagging, bootstrap
aggregating.

Be precise about "with replacement": some rows appear more than once in a
given sample, some not at all. That leftover fraction is not wasted — it
becomes free validation data, on the next slide.

Handout section 5.
:::

# How much data one bootstrap sample sees

- Each bootstrap sample is drawn from the same $m$ rows, with replacement
- About **63.2%** of the distinct rows are drawn at least once
- The remaining **36.8%** are left out entirely — called
  **out-of-bag (OOB)**
- At $m = 1{,}200$: a single simulated draw left out 36.6% of rows, close to
  the limit

::: notes
State the result, do not derive it — the limit that gives 63.2%/36.8% is a
standard exponential limit, worked out in full in the handout, and it does
not belong on a slide. What matters here is the consequence: every
bootstrap tree automatically leaves out over a third of the data it never
saw, for free.

Ask the room what that leftover third is good for. The answer is the next
slide: it is validation data for exactly the one tree that never saw it.

Handout section 5.1.
:::

# Out-of-bag: free validation

- Every out-of-bag row is validation data for the tree that missed it
- Averaging them gives the **OOB score**, at no extra split
- A 300-tree forest: OOB **0.9117**, 5-fold cross-validated
  **0.9117 ± 0.021**
- That match to four decimals is **this seed's luck**: over twelve seeds the
  gap is 0.003 — agreement to within their own noise, not to every decimal

::: notes
The agreement is the point, but be careful how you sell it. On this seed the
two numbers match to four decimal places, and that is luck: over twelve seeds
the typical gap is 0.0026 and the worst is 0.0075. What is not luck is that
they track each other well inside cross-validation's own ±0.021 spread,
because OOB score *is* cross-validation with different bookkeeping — each
tree supplying its own held-out fold, the rows it never bootstrapped, instead
of the dataset being split up front.

Worth pausing on, because it is a habit rather than a fact: **a single
coincidence is not evidence of a law.** If you catch yourself saying
"exactly", run it again with another seed. They will need that reflex in
their own exercises, and it is cheaper to learn it here.

Then make the practical consequence explicit: with a bagged ensemble you get
a trustworthy estimate of generalisation for free, without setting aside a
separate validation set at all — just not one to read to the fourth
decimal.

Handout section 5.1.
:::

# Why averaging reduces variance

- Model each tree's prediction as noisy, with variance $\sigma^2$, and let
  $\rho$ be the correlation between any two trees
- Averaging $B$ trees drives variance toward a **floor** of $\rho\sigma^2$,
  **not zero** — adding trees past that point buys almost nothing
- Averaging **cannot fix bias**: unbiased-but-noisy trees average to
  unbiased; trees sharing a systematic error keep that error exactly

::: notes
State the headline, not the algebra — the covariance expansion behind this
is in the handout, and it does not belong on a slide. The two consequences
are what matter: there is a floor variance cannot fall below, set by how
correlated the trees are with each other, and averaging is a variance tool
only, powerless against a systematic mistake shared by every tree.

That second point is worth a beat: bagging does nothing for a tree that is
too shallow to represent the rule in the first place. It cleans up noise, it
does not add capability.

Handout section 5.2.
:::

# Bagging, measured

| Model | Mean test accuracy | Standard deviation |
|---|---|---|
| Single unconstrained tree | 0.856 | 0.0185 |
| Bagging, 100 trees | **0.902** | **0.0153** |

::: notes
Across 30 independent 70/30 splits of the loan data, both numbers moved in
the predicted direction. The mean rose, because bagging also averages out
some of a deep tree's overfitting; the spread fell, which is exactly what
the previous slide's formula says averaging correlated estimators should
do.

Ask the room to name which of the two changes matters more for trusting a
single deployed model — the tighter spread, because it means the number you
report is less likely to be a lucky split.

Handout section 5.2.
:::

# Random forests: decorrelating the trees

- The variance floor is $\rho\sigma^2$ — lowering it further means making
  the trees agree **less**
- Bagged trees still see **every feature** at every split, so a strong
  feature pulls most of them towards splitting on it first
- A **random forest** restricts each split to a random subset of
  `max_features` (default $\sqrt{n}$, rounded down) — different trees see
  different features at the same point, lowering $\rho$

::: notes
Connect this directly to the previous slide's algebra: if the floor is
$\rho\sigma^2$, the only way to push it lower is to push $\rho$ down, and
bootstrap resampling alone only does that partially, because every bagged
tree still gets to see the full feature list.

The random forest's one change is a restriction, not an addition: at each
split, only a random handful of candidate features is even offered. That
forces disagreement between trees at the same point in the tree, purely by
denying them the same menu.

Handout section 6.
:::

# Random forests, measured

| Model | Mean test accuracy | Standard deviation |
|---|---|---|
| Single unconstrained tree | 0.856 | 0.0185 |
| Bagging, 100 trees | 0.902 | 0.0153 |
| **Random forest, 100 trees** | **0.908** | **0.0137** |

::: notes
Same 30 splits as the bagging comparison. The random forest beats plain
bagging on both counts — a better mean and a tighter spread — consistent
with $\rho$ having fallen further, exactly as the restriction on the
previous slide predicts.

Say plainly what has and hasn't changed: nothing about the splitting rule
is different from section 2. The only change across all three rows of this
table is what surrounds the tree — resampling, then also restricting the
feature menu.

Handout section 6.
:::

# Feature importance: what the forest claims to tell you

- A random forest reports **feature importance**: total impurity reduction
  each feature is responsible for, summed and averaged across every tree
  and every split
- Tempting to read as "the forest tells you which features matter"
- With only the two real features in this dataset, it does exactly that
- The question: what happens once most of the columns carry **nothing**?

::: notes
State the definition plainly, then set up the experiment. With just income
and debt ratio in the dataset, feature importance is a clean, correct
answer — both real features get essentially all the credit, because there
is nothing else to compete with them.

Ask the room to predict what happens as you add columns of pure noise —
`np.random.normal`, wired to nothing. Most will predict the noise columns
get close to zero importance, because averaging over hundreds of trees
should wash out anything with no signal. Do not confirm or deny yet — the
next slide is the answer.

Handout section 7.
:::

# More columns, more noise

| Noise columns added | Total columns | Cross-validated accuracy | Importance on noise |
|---|---|---|---|
| 0 | 2 | 0.912 | 0.0% |
| 5 | 7 | 0.874 | 33.9% |
| 20 | 22 | 0.837 | **54.5%** |

::: notes
Read down the last column, slowly, and let the bottom row sit before
explaining anything. With 20 pure-noise columns against 2 real features,
more than half of the forest's own importance scores land on columns that
are `np.random.normal` noise by construction.

Name the instinct this breaks, and defend it before moving on: expecting an
average over hundreds of trees to wash out noise is entirely reasonable —
the variance-reduction result from section 5.2 is real and measured, and it
does point the same way. It reduces the variance of the *prediction*. It
does not guarantee small importance for irrelevant columns, because the very
restriction that decorrelates the trees — a random, incomplete menu at each
split — is what occasionally hands a split to noise with nothing better on
offer.

The mechanism is exact and is in handout section 7. The practical
consequence to leave them with: treat a random forest's importances as
suggestive, not definitive, when real features are few against many
candidate columns.
:::

# Notebook 2, live

- Bagging and random forests, from scratch and against scikit-learn
- The out-of-bag score checked against cross-validation
- The noise-column experiment, run live

::: notes
Run Notebooks/02_bagging_and_random_forests.ipynb. Eighteen minutes.

The cell to protect is the noise-column sweep. Let them add the columns
themselves and watch the importance share climb — reading the number off a
slide does not land the same way as watching it happen.

If time allows, have them try `max_features=None` instead of the default
`"sqrt"` and predict what happens to the noise share before running it —
it should fall, at the cost of the decorrelation that made the forest work
in the first place.
:::

# Gradient boosting: correcting mistakes in sequence

- Bagging and random forests build trees **independently** and average at
  the end
- **Boosting** builds trees **in sequence**: each new tree is fit
  specifically to correct what the trees so far got wrong
- Each addition is a shallow tree — depth 2 or 3, a **weak learner**
- The learning rate $\alpha$ sets **how much** of each new tree's correction
  is actually applied

::: notes
Contrast the two strategies directly, because the difference is the whole
segment. Bagging and random forests are parallel: every tree is grown once,
independently, and combined only at prediction time. Boosting is
sequential: tree $t$ exists specifically because of what trees $1$ through
$t-1$ still got wrong.

Name $\alpha$ carefully and tie it to the course's shared notation — this is
the same symbol as lesson 3's gradient descent learning rate, doing an
analogous job: how large a step each correction takes.

Handout section 8.
:::

# Building the ensemble, one tree at a time

$$F_0(x) = \bar{y}, \qquad F_t(x) = F_{t-1}(x) + \alpha \cdot h_t(x)$$

::: notes
$F_0$ is the flat starting guess, the mean of the target. Each subsequent
$F_t$ is the previous ensemble plus a scaled correction from the newest
tree, $h_t$. Read it left to right: start flat, then keep adding small,
scaled corrections.

Do not derive anything further here — the next slide states what $h_t$ is
actually fit to. Handout section 8.
:::

# What each tree is fit to

- For squared-error regression, "what remains unexplained" has an exact
  name: the **residual**, $y - F_{t-1}(x)$
- This is a special case of a more general idea — **functional gradient
  descent**: $h_t$ approximates the negative gradient of the loss with
  respect to $F_{t-1}$
- For squared error, "fit the residual" and "fit the negative gradient"
  coincide exactly

::: notes
State the result without the calculus. For plain squared-error regression,
what a new tree corrects is literally the residual — the gap between the
truth and the ensemble's current prediction. That is intuitive on its own
and does not need a derivative to justify.

The second bullet is the generalisation that pays off in the very next
segment: the same recipe — fit a tree to the negative gradient of whatever
loss you're using — covers classification too, once the loss changes from
squared error to something else. Handout section 8 has the one line of
algebra connecting the two.
:::

# A sum of shallow trees converging on the residuals

![](boosting_residuals.png)

::: notes
Sixty depth-2 trees, learning rate 0.3, fit to a step function plus a
gentle sine wave, observed with noise. Walk through the panels in order.

Say the numbers as you point: one tree barely moves the flat starting
guess. Five trees sketch the coarse shape. Twenty trace the step function
closely — the ensemble's best point, on the next slide. By sixty, the
ensemble is tracking individual noisy observations as well as the
underlying shape.

Handout section 8.
:::

# Reading the picture

- **1 tree**: mean squared error (MSE) against the true function 0.395 —
  barely moves the flat guess
- **5 trees**: MSE 0.068 — the coarse shape appears
- **20 trees**: MSE **0.012** — the ensemble's best point
- **60 trees**: MSE rises back to 0.018 — tracking noise, not signal

::: notes
Put a number on each panel of the previous slide. Twenty trees is the sweet
spot; by sixty the error against the true, noise-free function has risen
again, even though the ensemble is still adding trees that reduce error on
the *training* residuals.

Name what this is: the same overfitting story as the unconstrained single
tree from section 3, reached by a completely different route — not one
tree grown too deep, but too many shallow trees each chasing whatever
residual noise is left.

Handout section 8.
:::

# Classification: the same idea, a different gradient

- The label $y$ is 0 or 1; the ensemble's output becomes a probability,
  $p = \sigma(F)$
- The loss is **log-loss** — lesson 4's loss for logistic regression
- The negative gradient, the **pseudo-residual** each tree fits, is $y - p$
- True label minus predicted probability: lesson 4's error term exactly

::: notes
State the result, not the chain rule behind it — the handout carries the
full derivative. What matters for this course is the connection: boosting
for classification fits a sequence of small trees to $y - p$, one
correction at a time, rather than adjusting a fixed set of linear
coefficients against the same quantity in one continuous descent.

Ask the room to name what's genuinely different between this and lesson
4's logistic regression, given they're descending against the same
quantity. The answer: lesson 4 moves a fixed number of coefficients;
boosting adds entirely new trees, so the model's structure itself grows
with training rather than staying fixed.

Handout section 9.
:::

# On the loan data

- `GradientBoostingClassifier`, defaults: **100 trees**, learning rate
  $\alpha = 0.1$
- Reaches **0.902 ± 0.020** cross-validated accuracy
- Already ahead of any single tree from earlier in the lesson

::: notes
Give the number and let it sit against everything shown so far this
lesson: 0.902, from a method run at scikit-learn's defaults, no tuning at
all. Ask the room to recall the single tuned tree's number from section 3
— 0.882 at depth 8 — and note that boosting has already passed it without
being told anything about depth.

The rest of this segment is what happens when boosting is *not* left at
sensible defaults.

Handout section 9.
:::

# Boosting never stops fitting on its own

| `n_estimators` ($\alpha = 0.3$) | Training accuracy | Cross-validated accuracy |
|---|---|---|
| 5 | 0.840 | 0.846 ± 0.025 |
| 10 | 0.888 | 0.870 ± 0.024 |
| 20 | 0.930 | 0.895 ± 0.027 |
| **30** | 0.948 | **0.898 ± 0.020** |
| 80 | 0.978 | 0.893 ± 0.012 |
| 200 | 1.000 | 0.895 ± 0.017 |
| 800 | 1.000 | 0.890 ± 0.019 |

::: notes
Unlike bagging, boosting keeps reducing training error for as long as it
runs, because every new tree is built specifically to reduce what remains —
including the 7% of labels that are noise by construction. There is nothing
in the algorithm itself that says "stop".

Point at the peak, 30 trees, and then at 800: training accuracy has been at
1.000 since well before 200, and cross-validated accuracy has drifted down
by about a point since the peak. Contrast the size of that drop with
section 3's unconstrained tree, which lost three points past its own peak —
boosting's overfitting is real but gentler.

Handout section 10.
:::

# Training climbs to 1.000 and stays; the honest curve turns over first

![](gbm_learning_curve.png)

::: notes
Two curves again, the same shape of story as the depth curve earlier this
lesson but reached by adding trees instead of depth. Training reaches 1.000
by around 200 trees and sits there. Cross-validated accuracy climbs
quickly, peaks near 30 trees, and drifts down over the following 770.

The point worth making explicit: the damage per added tree here is small
and easy to miss on a single run — which is exactly why early stopping
matters for boosting rather than being optional, even though the overfit is
gentler than a single unconstrained tree's.

Handout section 10.
:::

# Fixing 120 trees, varying only the learning rate

| Learning rate $\alpha$ | Training accuracy | Cross-validated accuracy |
|---|---|---|
| 0.02 | 0.888 | 0.872 ± 0.025 |
| 0.05 | 0.914 | 0.897 ± 0.022 |
| **0.10** | 0.939 | **0.902 ± 0.022** |
| 0.30 | 0.994 | 0.895 ± 0.020 |
| 1.00 | 1.000 | 0.878 ± 0.020 |

::: notes
Learning rate and tree count are not independent — both control how much
total correction the ensemble applies, so they trade off against each
other. At $\alpha = 0.02$, 120 trees haven't finished fitting the signal:
training accuracy is still under 0.89. At $\alpha = 1.00$, the ensemble has
memorised the training set completely and given back several points of
cross-validated accuracy for it.

The best setting sits in between — large enough to make real progress
within the tree budget, small enough that any one correction is easy to
outvote if it turns out to be wrong. Ask the room which direction they'd
guess is "more regularised" before revealing that it's the *small* $\alpha$
side, the same kind of inverted-direction trap lesson 6's $C$ was.

Handout section 10.
:::

# Notebook 3, live

- Boosting from scratch on residuals, and against scikit-learn
- The classification pseudo-residual, and the learning-rate sweep

::: notes
Run Notebooks/03_gradient_boosting.ipynb. Eighteen minutes.

The cell to protect is the residual sweep at 1, 5, 20 and 60 trees — seeing
the ensemble's fit against the true function rise and then fall again, in
real time, is the most direct experience of boosting's overfitting
available in this course.

If time is short, skip the learning-rate sweep live and set it as reading —
the table is in handout section 10.
:::

# Every method, on the same five folds

| Model | Cross-validated accuracy |
|---|---|
| Majority baseline | 0.613 |
| Logistic regression | 0.748 |
| Single tree, unconstrained | 0.852 |
| Single tree, `max_depth = 8` | 0.883 |
| Gradient boosting, 30 trees | 0.898 |
| Bagging, 100 trees | 0.904 |
| **Random forest, 100 trees** | **0.911** |
| *noise ceiling* | *≈ 0.93* |

::: notes
Everything from this lesson, cross-validated the same way. Read down the
column and let the pattern speak before naming it: every ensemble method —
bagging, the random forest, gradient boosting — lands within about a point
and a half of the others, and all three sit closer to the ceiling than
either single tree.

The ceiling itself: 1 minus the 7% of labels deliberately flipped. No
method reaches it, and none should — a model that classified the flipped
labels "correctly" would be modelling the noise, not the rule.

Handout section 11.
:::

# Three findings

- Every ensemble lands within about a point and a half of the others —
  **which strategy** you pick matters far less than **whether you ensemble
  at all**
- The single tuned tree is not far behind, at 0.883 against 0.911, and it
  is the only model on the table a person could read start to finish
- The **unconstrained tree is the worst model on the list**, despite being
  the most flexible one

::: notes
Read the three findings as a set — together they are this lesson's real
content, more than any single accuracy number.

The third finding is the one to land hardest, because it repeats a pattern
they've now seen four times with four different methods: flexibility
without a stopping rule is not an advantage. Ask the room to name the
earlier three occurrences before you do — k = 1 in lesson 6, an
unregularised fit in lesson 3, and today's own unconstrained tree from
section 3.

Handout section 11.
:::

# How to choose, in practice

| Situation | Reach for |
|---|---|
| The decision must be explained to the person it affects | A single tree, depth-tuned, or nothing on this list |
| Tabular data, accuracy matters most, no explanation required | A random forest, as a strong and low-effort default |
| Training time matters, or the signal needs sequential refinement | Gradient boosting, tuned on a validation set |
| Very high-dimensional, few informative columns | Any tree method, cautiously |
| Rows only, and OOB score suffices for validation | A random forest — no explicit train/test split needed |

::: notes
A table to photograph, exactly as lesson 6 closed with one. Work down it
and attach today's evidence to each row rather than leaving it as generic
advice.

The fourth row is worth a sentence of its own, tying back to the noise
experiment: with few real features against many candidate columns, trust
importances less as the column count grows past what the data can support.

Handout section 11.
:::

# 54% of the importance landed on noise

- 20 noise columns, 2 real features — not a small effect, and not a bug
- The same restriction that **decorrelates** the trees occasionally hands a
  split to noise instead
- Averaging reduces **variance of the prediction**; it does **not**
  guarantee small importance for irrelevant columns
- The fix: cross-check against permutation importance on held-out data

::: notes
This is the number to carry out of the room. Say it once more, slowly: with
twenty pure-noise columns against two real features, more than half of a
random forest's importance mass sat on columns wired to nothing.

Ask them to write it down — derivations fade, this kind of number tends
not to. It is the fourth lesson this course has produced one of these: coin
flip labels, borrowed test-set values, an overfit coefficient of 247,514,
and now this.

Handout section 7 has the full mechanism.
:::

# What to take away

- A tree splits greedily on Gini impurity, verified to the **fourth
  decimal place** against scikit-learn
- Depth is the bias-variance dial: accuracy peaked at **depth 8**, three
  points above the unconstrained tree
- Bootstrap resampling leaves **36.8%** of rows out-of-bag, free validation
- **54% of a random forest's importance landed on 20 noise columns** —
  today's number worth remembering

::: notes
Close on the same shape as lesson 6's summary: a handful of numbers that
carry the lesson's argument without needing the mathematics behind them
re-derived.

If they remember one habit from today, make it this: before trusting a
random forest's importance ranking, ask how many candidate columns compete
for attention at each split, because that number — not the accuracy score —
is what decides how much noise gets credit it didn't earn.

Next lesson turns from labelled data to unlabelled data — a fourth question
this course asks of a dataset, after "what's the relationship", "which
class", and "how do you know you're not fooling yourself": what structure
is there when nobody has told you the answer at all.

Handout summary and notation table.
:::

# Homework

- **Exercise 7**, due **Friday 13 November 2026, 23:59**
- `Exercises/07_trees_and_ensembles.md`

::: notes
Set it explicitly and say the deadline out loud: Friday 13 November,
23:59 — the same date lesson 8 begins.

Remind them of the standing rule from lesson 5: every reported number needs
a cross-validated spread, and any feature-importance claim needs the check
this lesson just demonstrated — a random forest's importance ranking is a
claim to be verified, not a fact to be quoted.

Next week: unsupervised learning, the first lesson this course asks of data
with no labels at all.
:::

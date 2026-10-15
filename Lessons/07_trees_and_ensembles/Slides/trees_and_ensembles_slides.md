---
title: "Lesson 7: Trees and Ensembles"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "13 November 2026"
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
fourth kind of model, built on a different basic step: comparing one feature
with a threshold, where k-nearest neighbours measured distances and the
support vector machine computed kernels. It then spends most of the three hours on a different question: not "does a
tree work", but "what happens when you build more than one".

Tell them the punchline early, without the number yet: a single tree is one
of the few models in this course a person with no mathematics background can
read end to end, and it is also, on its own, a mediocre model. Ensembles of
trees are what most tabular-data competitions are still won with, and the
reason is the subject of today.

Handout section 1 sets this framing out in full.
:::

# Exercise 6, before we start

- What counts is the **defence of which family**, not the winning score
- The gap to watch for: a strong accuracy number with no check of *why* the
  assumption behind it held
- Today's models come with the same obligation

::: notes
Nothing to collect and nothing to hand back - this is a discussion. Ask who
picked which family and why, and listen for the common gap: a cross-validated
score quoted without any check of whether the model's assumption, such as
within-class independence, actually held on that data.

Link forward explicitly: today introduces a family with its own convenient
assumption - that the columns you feed a random forest are all worth
listening to - and by the end of the lesson you will have measured exactly
how badly that assumption can fail.
:::

# 1,200 loan applicants, two figures each

- Income in thousands of euros a year, and debt-to-income ratio
- Underwriting is a **checklist**: an income floor, a debt ceiling, and two
  "stressed" combinations, each figure ordinary alone but risky together
- **7%** of labels flipped: no honest model beats about **0.93**, the
  **noise ceiling**
- **38.7%** defaulted, so the majority baseline is **0.613**

::: notes
Describe the physical logic before the picture. A bank's rule here is closer
to a checklist than a smooth function: almost nobody is approved below an
income floor or above a debt ceiling, and there are two further regions where
income and debt are each unremarkable on their own but risky in combination.

Ask the room what shape that logic implies before you show the scatter. The
useful answer is: not one connected region, so no single straight line can
separate the classes - a different failure from lesson 6's disc-in-a-ring,
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
margin here - it has learned something. Two of the four risky regions, the
low-income strip and the high-debt strip, get riskier steadily as one feature
moves in one direction, and a straight boundary handles that kind of effect
reasonably well.

What it cannot do is wall off the two interior islands without cutting into
healthy territory everywhere else. Ask the room what kind of boundary could
close that gap, and let a few guesses land before moving on - a threshold
rule, applied more than once, is exactly the answer the rest of the lesson
builds.

Handout section 1.1.
:::

# The whole algorithm, in one sentence

- Find the (feature, threshold) split that makes the two groups **as pure
  as possible**
- Apply it; repeat on each group, independently
- Stop at a depth limit, a pure group, or too few examples to split
- Nothing is estimated: no coefficient, no gradient
- **Greedy**: whichever split helps most *right now*, with no lookahead

::: notes
Read the algorithm as one sentence and let it sit - it really is that short.
Contrast it explicitly with lessons 3 and 4: no cost function is minimised
over a continuous parameter, there is no gradient step. A tree is a sequence
of yes/no questions, chosen one at a time.

The word to stress is "greedy". At every step the tree takes whichever split
helps most immediately, with no mechanism for looking two levels ahead. Tell
them to hold onto that word - it is what explains, later this hour, why a
shallow tree can miss the interior islands entirely even though the
information to find them is right there in the data.

Handout section 2.1.
:::

# The algorithm, drawn

![](tree_growth.png)

::: notes
The previous slide's sentence, run three times on the loan data. Each column
is one more round; on top the cuts in the plane, the new ones in gold, and
underneath the tree that asks the same questions. Point at the two halves of
each column in turn: every gold line on top is a gold box underneath.

Step 1: of every possible single question, debt ratio at or below 0.82 is the
one that leaves the two groups purest - 971 applicants at 26% default against
229 at 94%. Do not say why it is the best yet; the next slides give the number
that ranks it.

Step 2 is the word "recursive" made visible. The tree now asks a question
inside each group, and the two questions are different - income on the left,
debt again on the right - because each group is searched on its own, without
regard to the other. That is "independently" in the sentence.

Step 3 is where to slow down, because it shows greed. On the right, one split
sets aside a single applicant from 68 others - the cut at 0.948, a thousandth
above the 0.947 one, so in the plane the two gold lines lie on top of each
other. Someone will ask why the tree asks about debt twice. The two
thresholds are different questions: exactly one applicant has a debt ratio
between 0.947 and 0.948, that applicant repaid, and the tree spent a whole
split on separating them. Another split divides 160 applicants into groups
at 90% and 97% default - both groups are still predicted "default", so that
cut changes no decision at all. Each was the best available move at that node; the tree
has no way to ask whether it was worth making. And neither interior island has
appeared yet: three rounds were spent on the easy strips. Hold that thought -
it is the "what greedy misses" slide in ten minutes.

The printout of every node is in notebook 1, section 2. Handout section 2.1.
:::

# How pure is a group of examples?

- A leaf that is all one class needs no more questions
- A leaf split evenly between classes is the worst case: a coin flip
- **Impurity** is a number that captures this: 0 for a pure group, highest
  for an even mix. "As pure as possible" then means "with the lowest
  impurity"

::: notes
Before any symbol appears, ask the room what "pure" should mean for a group
of examples. Most will land close to the right answer on their own: a group
where everyone shares a label needs nothing further, and a fifty-fifty split
is as bad as it gets.

Say plainly that the standard choice for classification is called the Gini
impurity, and that it is not a probability of misclassification - it is the
probability that two examples drawn at random with replacement from the
group, and labelled according to the group's own class frequencies, would
disagree.

Handout section 2.2.
:::

# Gini impurity

$$G = 1 - \sum_{c=1}^{C} p_c^2$$

::: notes
Each p in the sum is the fraction of the group belonging to one class,
across C classes. Do not derive anything here - state what it does: subtract, from
one, the chance two examples share a class if drawn independently from the
group's own frequencies. A pure group scores zero; a maximally mixed one
scores as high as the formula allows.

Give them a moment to notice the formula asks nothing about *which* class is
common, only about how mixed the group is - that symmetry matters later,
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
you confirm it - it is the point where a random guess is least useful.

Give them the concrete number: on the full loan dataset, 38.7% defaulted, so
p = 0.387 and G(parent) = 2 × 0.387 × 0.613 = 0.474. That is the impurity
every candidate root split is measured against.

Handout section 2.2.
:::

# The gain a split buys

- A candidate split divides a parent group P into a left child L and a
  right child R; *m* counts the examples in each
- Worth taking if the children are, **on average, purer** than the parent
- The average is **weighted by size**: one example cut off alone would
  halve a plain average, and barely moves a weighted one

::: notes
Set up the idea in words before the formula: a split is only useful if the
two resulting groups are, taken together, purer than the one group was. The
weighting matters because a split that isolates one awkward point while
leaving the rest exactly as mixed as before should not score as well as one
that improves the picture broadly.

Work the last bullet through. Cut a single example off a mixed group of 1,200.
The one-example child is pure, impurity 0; the other child is as mixed as the
parent was. A plain average of the two is half the parent's impurity, so the
split would look like a large gain. Weighted by size, the pure child counts
for 1/1,200 of the average and the other child for the rest, so the average is
almost exactly the parent's impurity again, and the gain is close to zero.

Ask what could go wrong if you *didn't* weight by group size - the answer is
that a tree would happily carve off tiny, perfectly pure single-point groups
at every step, which is exactly the overfitting behaviour the depth table
shows in a few minutes.

Handout section 2.2.
:::

# The reduction in impurity

$$\Delta G = G_P - \left(\frac{m_L}{m_P}\, G_L + \frac{m_R}{m_P}\, G_R\right)$$

::: notes
Say what the symbols are before reading the formula: P, L and R are the
parent group and its two children, as on the previous slide. Each G is the
Gini impurity of one group, and each m is the number of examples in it; the
two children's m add up to the parent's. Read it as: the parent's impurity,
minus the children's impurities averaged with weights equal to the share of
the parent that went to each child. ΔG (delta G) is the gain. The tree keeps
whichever candidate split makes $\Delta G$ largest.

No derivation needed here - the formula is exactly the weighted comparison
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
combination with a second split further down the tree can be - and in this
dataset's two interior islands, is - invisible to a shallow search, because
$\Delta G$ is never evaluated for two splits together.

Handout section 2.2.
:::

# Worked example: the root split

- $p = 0.387$ defaulted, so the parent group's impurity is **0.474**
- Exhaustive search finds `debt_ratio <= 0.82`: the **debt ceiling** the
  data was built with
- 971 applicants left (0.256 defaulted, impurity 0.381), 229 right (0.939,
  impurity 0.115)
- Weighted by size, the children average 0.3305 against the parent's
  0.4743: gain **0.144**

::: notes
Walk the numbers once: parent impurity 0.474, and the winning split is the
debt ceiling, found by brute-force search with no hint about how the data
was generated. The children's impurities, 2p(1 - p) at their own default
rates, are 0.3814 for the left child and 0.1148 for the right. Their average weighted
by size is 971/1,200 × 0.3814 + 229/1,200 × 0.1148 = 0.3305, and the gain is
the parent's 0.4743 minus that, 0.1438, which rounds to 0.144. Use four
decimals if you work it on the board: with the impurities rounded to the
slide's 0.381 and 0.115, the weighted average comes out at 0.3302, and
someone will notice. That agreement with the known generating rule is the first
piece of evidence that the algorithm is doing something sensible.

Ask the room why finding the debt ceiling by blind search, with no
knowledge of how the data was built, is a stronger check than it might
first appear - the answer is that it rules out the split being an artefact
of how the search was coded, since nothing in the algorithm knows the
generating rule.

Handout section 2.2.
:::

# The root split, on the Gini curve

![](gini_root_split.png)

::: notes
The previous slide's arithmetic, drawn. The curve is 2p(1-p): zero for a pure
group, 0.5 for a fifty-fifty one. The black dot is all 1,200 applicants at
p = 0.387, impurity 0.474. The split sends 971 left, at p = 0.256, and 229
right, at p = 0.939 - teal and rust.

The gold square is the two children averaged by size, 0.3305, and it sits on
the dotted straight line between them. Point at the bulge: the curve lies above
every such line, so any split that separates the classes at all lowers the
weighted impurity. The gap, 0.144, is the gain, and the tree takes whichever
split makes it largest. Handout section 2.2.
:::

# Checked against scikit-learn

- The from-scratch implementation and scikit-learn's
  `DecisionTreeClassifier` make **the same prediction for all 1,200
  applicants**, at every depth tested
- Not a coincidence: both search the same space by the same greedy rule
- From here on, the notebooks use scikit-learn's implementation: the same
  algorithm, compiled, not a different one

::: notes
This is the check notebook 1 runs live, and it matters for trust rather
than for novelty: once the from-scratch version is shown to agree with
scikit-learn exactly, every later result in the lesson - obtained with
scikit-learn - inherits that confidence.

Two questions to pre-empt. Why predictions and not accuracy? Because two
different trees can get the same number of applicants wrong while getting
different applicants wrong; identical predictions, one by one, rule that out.
And why on the training data, after lesson 5? Because this checks the code,
not the model: the question is whether two programs build the same tree from
the same rows, and the rows they learned from are exactly the right ones to
ask it on. No accuracy here is a claim about new applicants.

Point out the "Try this" box in the handout: computing Gini impurity for
the income-floor subgroup alone and comparing it against the full dataset's
0.474 is a two-line exercise that makes the impurity number concrete rather
than abstract.

Handout section 2.2, and the "Try this" box that follows it.
:::

# What greedy misses

- The search never asks "which pair of splits helps most together": only
  "which single split helps most right now"
- A feature that only pays off **in combination** with a later split can be
  invisible to a shallow tree
- Exactly what happens to this dataset's two interior "stressed" islands,
  covered next

::: notes
Close the Gini segment with the limitation, because it sets up everything
that follows. Ask the room: if the debt ceiling and the income floor are the
two easiest wins, what happens to the two interior islands, which need *two*
conditions at once to identify?

The honest answer is that a shallow tree simply cannot see them - not
because the information isn't there, but because greedy search only ever
commits to the best split available at each individual step. Depth is what
buys the tree enough steps to eventually stumble onto them, which is exactly
where the next segment goes.

Handout sections 2.2 and 3.
:::

# Depth is the bias-variance dial

- `max_depth`: how many questions a tree may ask before a leaf must stop and
  vote by majority
- **Shallow tree**: few questions, only the coarsest structure. High bias,
  low variance
- **Deep tree**: enough questions to isolate almost any subset, down to
  single points. At full depth, training accuracy is **exactly 1.000**, even
  on data with no structure at all, unless two examples share every feature
  value but not the label

::: notes
Name the dial explicitly and connect it to lesson 6: k in k-nearest
neighbours traded bias against variance directly, and depth is a tree's
version of the same knob.

The training-accuracy-1.000 claim is the one to make them justify. Ask why
an unconstrained tree gets every training point right - the answer is that
splitting does not have to stop until every leaf is pure, at worst one
example per leaf, which it then "predicts" perfectly by definition. The one
exception is the condition on the slide: two applicants with identical
income and debt but different outcomes can never be split apart, so one of
them stays wrong. With two continuous features drawn at random, as here, that
does not happen. That
number demonstrates the model class can memorise, not that anything has
been learned - the same reading lesson 6 gave k = 1's training accuracy.

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
Cross-validated accuracy rises, peaks at depth 8, and then falls back - even
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

The gold dashed line is the noise ceiling from the dataset slide, 0.93: one
minus the 7% of flipped labels.

This is not the first time this course has shown a curve bend this way:
lesson 3's test error turned back up as the polynomial degree grew, and
lesson 6's k-nearest neighbours did the same as k shrank towards 1. Handout
section 3.
:::

# The same rule, three depths

![](tree_boundaries.png)

::: notes
Left to right: at depth 2 the tree affords only the income floor and the
debt ceiling - the two biggest, easiest-to-find regions - and neither
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

- `max_depth` grows the tree **breadth-first**: every leaf at level $d$
  splits before any leaf reaches $d+1$
- `max_leaf_nodes` grows it **best-first**: always the leaf with the largest
  $\Delta G$
- `max_leaf_nodes = 9`: **90.8%** training accuracy; thresholds 28, the
  income floor, and 0.82, the debt ceiling
- In code: `export_text(tree)` prints the tree as if/else rules

::: notes
This is the readability half of the story. For a tree meant to be read
rather than merely scored, best-first growth is the better knob: a small
budget of splits goes to the ones that matter most, instead of being spent
uniformly across every branch regardless of whether it's worth it.

The nine-leaf tree is worth reading aloud as a sequence of if/else
statements - that is the entire model, and it is the thing k-nearest
neighbours and support vector machines cannot offer at all. For a decision
that has to be justified to the person it affects, that can matter as much
as accuracy.

Handout section 4.
:::

# A tree that doesn't stay still

- Two trees at `max_depth = 6`, each on a random 65% of the
  applicants (no replacement)
- The debt ceiling, the strongest split, agrees to within **0.001**
- The trees grow **31 and 33 leaves**, and **disagree on 11.7%** of
  predictions on the full dataset
- The strongest split is stable; the weaker ones, which shape most
  of the tree, are not

::: notes
This is the instability that ensembles turn into raw material. The
strongest split barely moves between the two samples, 0.820 against 0.819,
because the evidence behind it is overwhelming: each tree sees 780
applicants, and on the full data the debt ceiling separates 26% default
from 94%, a contrast no sample of that size can hide. Every later split is made inside a smaller group, supported by far
fewer applicants, and those splits are what actually determine a tree's
shape.

Ask the room what 11.7% disagreement between two trees fit to overlapping
samples of the same applicants implies about trusting any single tree's
exact boundary. The
answer is the whole of the next segment: a single tree's instability looks
like a flaw, and it is about to become a strength.

Handout section 4.
:::

# Notebook 1, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 1. look at it first, 4-6 | `make_pipeline(StandardScaler(), LogisticRegression())` in `cross_val_score(..., cv=StratifiedKFold(5))` → 0.613 against 0.748 |
| 2. a tree, split by split, 7-18 | your own `gini` and `best_split`, then `DecisionTreeClassifier(max_depth=d)` → 100.0% identical predictions; gain 0.1438 |
| 3. the depth dial, 19-22 | `DecisionTreeClassifier(max_depth=d)` in `cross_val_score` → peak 0.882 at depth 8 |
| 4. what a tree costs, 23-24 | `DecisionTreeClassifier(max_leaf_nodes=9)`, `export_text` → 0.908; two trees agree on 0.883 |

::: notes
The table is the notebook's map: each section, the calls it makes, the slides
that introduced the idea, and the number to come back with. Decision trees
from scratch, checked against scikit-learn; depth swept, the boundaries
drawn, and the instability measured.

Run Notebooks/01_decision_trees_from_scratch.ipynb. Twenty minutes.

Have them watch the from-scratch implementation confirm the debt-ceiling
root split before scikit-learn's version ever runs - the point is that the
agreement is not assumed, it is checked, at every depth.

The cell worth protecting if time is short is the two-tree resampling
comparison at the end: the notebook prints an agreement of 0.883, which is
the 11.7% disagreement of the previous slide, and seeing it appear on their
own screen lands harder than reading it off a slide.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. The next segment is where the lesson's real content starts  - 
one tree's instability, turned into an ensemble's strength - so they need to
come back with that framing fresh.
:::

# Bagging: averaging away the instability

- Trees differ across samples, so **many trees, averaged** should partly
  cancel their mistakes
- **Bootstrap sample**: $m$ rows drawn **with replacement** from the $m$
  training rows
- One tree per sample, `n_estimators` times; predict by **majority vote**.
  That is **bagging**: bootstrap aggregating
- In code: `BaggingClassifier()`, whose default model is a decision tree

::: notes
State the logic before the name. Two slides before the break measured that
two trees fit to overlapping samples disagree on more than a tenth of
predictions. If those
disagreements are close to random, averaging many such trees should cancel
some of them out - that is the entire idea behind bagging, bootstrap
aggregating.

Say that the tree algorithm itself is unchanged: bagging reuses the first
hour's tree as it is, and only what surrounds it is new.

Be precise about "with replacement": some rows appear more than once in a
given sample, some not at all. That leftover fraction is not wasted - it
becomes free validation data, on the next slide.

Handout section 5.
:::

# How much data one bootstrap sample sees

- One row is missed by all $m$ draws with probability $(1 - 1/m)^m$, which
  tends to $e^{-1} \approx 0.368$ as $m$ grows
- So about **63.2%** of the distinct rows are drawn at least once
- The remaining **36.8%** are left out entirely: called
  **out-of-bag (OOB)**
- At $m = 1{,}200$: a single simulated draw left out 36.6% of rows, close to
  the limit

::: notes
Give the one line of reasoning, not the limit. Each draw misses a given row
with probability 1 - 1/m, and the m draws are independent, so the row is
missed by all of them with probability (1 - 1/m) to the power m. At m = 1,200
that is 0.3677; as m grows it tends to 1/e, 0.368, which is the standard
exponential limit worked out in handout section 5.1. What matters here is the
consequence: every bootstrap tree automatically leaves out over a third of
the rows, and never sees them.

Ask the room what that leftover third is good for. The answer is the next
slide: it is validation data for exactly the one tree that never saw it.

Handout section 5.1.
:::

# One bootstrap sample, counted

![](bootstrap_counts.png)

::: notes
How often each of the 1,200 loans turns up in one bootstrap sample, blue,
against the limit e^-1 / k!, gold. The first bar is the out-of-bag share: 0.366
here, 0.368 in the limit. The rest are rows drawn once, twice, three times.

Where the gold bars come from, if asked: the number of times one row is drawn
counts successes in 1,200 draws, each with probability 1/1,200. As the number
of draws grows with the chance kept at one per row, that count follows the
Poisson distribution with mean 1, which gives a fraction e^-1 / k! of rows
drawn exactly k times: 0.368 never, 0.368 once, 0.184 twice, 0.061 three
times.

The point to make: about one row in four appears two or more times. Every tree
sees a different mix of repeats and absences, and that is where their
disagreement - the raw material of bagging - comes from. Handout section 5.1.
:::

# Out-of-bag: free validation

- Predict each row by a vote of only the trees that never saw it: the
  accuracy of those votes is the **OOB score**, with no rows set aside
- A 300-tree random forest (bagged trees plus one rule, slide 34): OOB
  **0.9117**; 5-fold cross-validation **0.9117 ± 0.021**
- The four-decimal match is **luck**: over twelve seeds the two differ by
  0.0026 on average, well inside ±0.021

::: notes
The forest here comes four slides before its definition: a random forest is
bagging with one extra rule, which slide 34 gives, and scikit-learn computes
the OOB score for it with `oob_score=True`. For this slide, read it as "300
bagged trees".

The agreement is the point, but be careful how you sell it. On this seed the
two numbers match to four decimal places, and that is luck: over twelve seeds
the typical gap is 0.0026 and the worst is 0.0075. What is not luck is that
they track each other well inside cross-validation's own ±0.021 spread,
because OOB score *is* cross-validation with different bookkeeping - each
tree supplying its own held-out fold, the rows it never bootstrapped, instead
of the dataset being split up front.

Worth pausing on, because it is a habit rather than a fact: **a single
coincidence is not evidence of a law.** If you catch yourself saying
"exactly", run it again with another seed. They will need that reflex in
their own exercises, and it is cheaper to learn it here.

Then make the practical consequence explicit: with a bagged ensemble you get
a trustworthy estimate of generalisation for free, without setting aside a
separate validation set at all - just not one to read to the fourth
decimal.

Handout section 5.1.
:::

# Why averaging reduces variance

- Model each tree's prediction as noisy, with variance $\sigma^2$ over the
  bootstrap sample it drew, and let $\rho$ be the correlation between two trees
- The average of $B$ trees has variance $\sigma^2/B + \rho\sigma^2 (B-1)/B$:
  a **floor** of $\rho\sigma^2$, **not zero**: more trees soon buy little
- Averaging **cannot fix bias**: unbiased-but-noisy trees average to
  unbiased; trees sharing a systematic error keep that error exactly

::: notes
State the headline, not the algebra - the covariance expansion behind this
is in handout section 5.2, and it does not belong on a slide. Read the
formula on the slide in two pieces: the first term, sigma squared over B,
shrinks as trees are added; the second, (B - 1)/B times rho sigma squared,
grows towards rho sigma squared and stays there. The two consequences
are what matter: there is a floor variance cannot fall below, set by how
correlated the trees are with each other, and averaging is a variance tool
only, powerless against a systematic mistake shared by every tree.

That second point is worth a beat: bagging does nothing for a tree that is
too shallow to represent the rule in the first place. It cleans up noise, it
does not add capability.

Handout section 5.2.
:::

# Averaging stops at a floor

![](variance_floor.png)

::: notes
The previous slide's formula as curves: the variance of an average of B trees,
as a share of one tree's, for rho = 0, 0.2, 0.5 and 0.8.

Only the rho = 0 curve goes to zero. Every other one flattens onto its dotted
floor: at rho = 0.5, ten trees keep 0.550 of one tree's variance and a hundred
keep 0.505. Ask the room what the curves say about adding trees past fifty -
almost nothing - and what the only remaining lever is: lower rho. That is the
random forest, two slides on. Handout section 5.2.
:::

# Bagging, measured

| Model | Mean test accuracy, 30 random 70/30 splits | Standard deviation across the 30 |
|---|---|---|
| Single unconstrained tree | 0.856 | 0.0185 |
| Bagging, 100 trees | **0.902** | **0.0153** |

::: notes
Across 30 random 70/30 splits of the loan data, both numbers moved in
the predicted direction. The mean rose, because bagging also averages out
some of a deep tree's overfitting; the spread fell, which is exactly what
the previous slide's formula says averaging correlated estimators should
do.

Ask the room to name which of the two changes matters more for trusting a
single deployed model - the tighter spread, because it means the number you
report is less likely to be a lucky split.

Handout section 5.2.
:::

# Random forests: decorrelating the trees

- The floor is $\rho\sigma^2$: lowering it means making the trees agree
  **less**
- Bagged trees see **every feature**, so a strong one pulls them all the
  same way
- A **random forest** offers each split a random subset of `max_features`
  (default $\sqrt{n}$, rounded down), lowering $\rho$
- In code: `RandomForestClassifier()`, 100 trees by default

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

| Model | Mean test accuracy, 30 random 70/30 splits | Standard deviation across the 30 |
|---|---|---|
| Single unconstrained tree | 0.856 | 0.0185 |
| Bagging, 100 trees | 0.902 | 0.0153 |
| **Random forest, 100 trees** | **0.908** | **0.0137** |

::: notes
Same 30 splits as the bagging comparison. The random forest beats plain
bagging on both counts - a better mean and a tighter spread - consistent
with $\rho$ having fallen further, exactly as the restriction on the
previous slide predicts.

Say plainly what has and hasn't changed: nothing about the splitting rule
is different from the first hour's trees. The only change across all three rows of this
table is what surrounds the tree - resampling, then also restricting the
feature menu.

Handout section 6.
:::

# Feature importance: what the forest claims to tell you

- **Feature importance**: the impurity reduction credited to each feature
  over every tree and split, scaled to add up to 1
- With only the two real features here, it names exactly those two
- What happens once we add columns of random numbers, unrelated to default?
- In code: `forest.feature_importances_`

::: notes
State the definition plainly, then set up the experiment. With just income
and debt ratio in the dataset, feature importance is a clean, correct
answer - both real features get essentially all the credit, because there
is nothing else to compete with them.

Ask the room to predict what happens as you add columns of pure noise  - 
`np.random.normal`, wired to nothing. Most will predict the noise columns
get close to zero importance, because averaging over hundreds of trees
should wash out anything with no signal. Do not confirm or deny yet - the
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
average over hundreds of trees to wash out noise is entirely reasonable  - 
the variance-reduction result of the averaging slides is real and measured, and it
does point the same way. It reduces the variance of the *prediction*. It
does not guarantee small importance for irrelevant columns, because the very
restriction that decorrelates the trees - a random, incomplete menu at each
split - is what occasionally hands a split to noise with nothing better on
offer.

The mechanism is exact and is in handout section 7. The practical
consequence to leave them with: treat a random forest's importances as
suggestive, not definitive, when real features are few against many
candidate columns.
:::

# Notebook 2, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 1. bagging from scratch, 27 | `train_test_split(..., test_size=0.3, stratify=y)`, your own bagging against `BaggingClassifier`, both scored by `accuracy_score` → 0.872 and 0.872 |
| 2. why averaging works, 28-32 | a forest with `oob_score=True`: its `.oob_score_` against `cross_val_score` → 0.9117 and 0.9117; 0.0026 over 12 seeds |
| 3. variance, measured, 33-35 | a tree, `BaggingClassifier`, `RandomForestClassifier` on 30 splits → spread 0.0185, 0.0153, 0.0137 |
| 5. feature importance, 36-37 | `.feature_importances_` with 0, 5 and 20 noise columns → 54.5% on noise |

::: notes
The table is the notebook's map: bagging from scratch and against
scikit-learn, the out-of-bag score checked against cross-validation, the
variance measured over 30 splits, and the noise-column experiment. Section 4
of the notebook is reading only, which is why the table skips it.

Run Notebooks/02_bagging_and_random_forests.ipynb. Eighteen minutes.

The cell to protect is the noise-column sweep. Let them add the columns
themselves and watch the importance share climb - reading the number off a
slide does not land the same way as watching it happen.

If time allows, have them try `max_features=None` instead of the default
`"sqrt"` and predict what happens to the noise share before running it  - 
it should fall, at the cost of the decorrelation that made the forest work
in the first place.
:::

# Gradient boosting: correcting mistakes in sequence

- Bagging and random forests build trees **independently**, then average
- **Boosting** builds them **in sequence**: each new tree corrects what the
  trees so far got wrong
- Each is a shallow **regression tree**, a **weak learner**: leaves predict
  a mean, splits reduce squared error
- The learning rate $\alpha$ sets **how much** of each correction is applied

::: notes
Contrast the two strategies directly, because the difference is the whole
segment. Bagging and random forests are parallel: every tree is grown once,
independently, and combined only at prediction time. Boosting is
sequential: tree $t$ exists specifically because of what trees $1$ through
$t-1$ still got wrong.

Slow down on "regression tree", because it is new. "Shallow" means depth 2
or 3 here. Everything so far this
lesson was a classification tree: leaves vote, splits are scored by Gini. A
boosted tree is fit to corrections, which are numbers like +0.613 or -0.387,
not classes, so it is the regression version of the same algorithm: the same
greedy search over features and thresholds, but each leaf predicts the
average correction of the examples in it, and a split is scored by how much
it lowers the squared error around those averages. Even
`GradientBoostingClassifier` grows regression trees inside.

Name $\alpha$ carefully and tie it to the course's shared notation - this is
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

Do not derive anything further here - the next slide states what $h_t$ is
actually fit to. Handout section 8.
:::

# What each tree is fit to

- For squared error, what remains unexplained is the **residual**,
  $y - F_{t-1}(x)$
- In general, **functional gradient descent**: $h_t$ approximates the
  negative gradient of the loss with respect to $F_{t-1}$
- With $L = \tfrac{1}{2}(y - F)^2$, $-\partial L/\partial F = y - F$: the two coincide
- In code: `DecisionTreeRegressor(max_depth=2)`, fit to `y - prediction`

::: notes
The In code line is the whole from-scratch loop of notebook 3, section 1:
fit a depth-2 regression tree to the residuals, add 0.3 times its predictions
to the running prediction, repeat sixty times.

State the result without the calculus. For plain squared-error regression,
what a new tree corrects is literally the residual - the gap between the
truth and the ensemble's current prediction. That is intuitive on its own
and does not need a derivative to justify.

The second bullet is the generalisation that pays off in the very next
segment: the same recipe - fit a tree to the negative gradient of whatever
loss you're using - covers classification too, once the loss changes from
squared error to something else. The one line of algebra connecting the
two is on the slide; handout section 8 sets it out.
:::

# A step plus a sine wave, fit by 60 shallow trees

![](boosting_residuals.png)

::: notes
A new toy dataset, so say what it is before anything else: 200 points on one
input x, whose true value is a three-level step function plus a gentle sine
wave (the black line), observed with noise (the grey dots). Sixty depth-2
trees, learning rate 0.3, are fit to it. Walk through the coloured curves in
order.

Say the numbers as you point: one tree barely moves the flat starting
guess. Five trees sketch the coarse shape. Twenty trace the step function
closely - the ensemble's best point, on the next slide. By sixty, the
ensemble is tracking individual noisy observations as well as the
underlying shape.

Handout section 8.
:::

# Reading the picture

- **1 tree**: mean squared error (MSE) against the true function 0.395,
  barely moves the flat guess
- **5 trees**: MSE 0.068, the coarse shape appears
- **20 trees**: MSE **0.012**, the ensemble's best point
- **60 trees**: MSE rises back to 0.018, tracking noise, not signal

::: notes
Put a number on each curve of the previous slide. Twenty trees is the sweet
spot; by sixty the error against the true, noise-free function has risen
again, even though the ensemble is still adding trees that reduce error on
the *training* residuals.

Name what this is: the same overfitting story as the unconstrained single
tree of the depth table, reached by a completely different route - not one
tree grown too deep, but too many shallow trees each chasing whatever
residual noise is left.

Handout section 8.
:::

# Classification: the same idea, a different gradient

- $y$ is 0 or 1; the score $F$ becomes a probability through lesson 4's
  logistic function, $p = \sigma(F)$
- The loss is lesson 4's **log-loss**
- Each tree fits the **pseudo-residual** $y - p$, the negative gradient:
  lesson 4's error term
- Start: $p = 0.387$ for all, so the first tree fits +0.613 to defaulters
  and −0.387 to the rest

::: notes
State the result, not the chain rule behind it - the handout carries the
full derivative. What matters for this course is the connection: boosting
for classification fits a sequence of small trees to $y - p$, one
correction at a time, rather than adjusting a fixed set of linear
coefficients against the same quantity in one continuous descent.

Ask the room to name what's genuinely different between this and lesson
4's logistic regression, given they're descending against the same
quantity. The answer: lesson 4 moves a fixed number of coefficients;
boosting adds entirely new trees, so the model's structure itself grows
with training rather than staying fixed.

The sigma here is lesson 4's logistic function, 1 / (1 + e^-F), not the
sigma squared of the variance slides: say so once, since the same letter has
just meant something else.

Give them the first step in numbers, because it makes y - p concrete. The
ensemble starts at the base rate's log-odds: 464 of the 1,200 defaulted, so
F0 = log(464 / 736) = -0.461, which the logistic function turns back into
0.387 for everyone. The first tree is fit to y - p: +0.613 for every applicant
who defaulted, -0.387 for every one who repaid.

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
all. Ask the room to recall the single tuned tree's number from the depth
table, 0.882 at depth 8, and note that boosting has already passed it without
being told anything about depth.

The rest of this segment is what happens when boosting is *not* left at
sensible defaults.

Handout section 9.
:::

# Boosting never stops fitting on its own

| `n_estimators` ($\alpha = 0.3$, depth 3) | Training accuracy | Cross-validated accuracy |
|---|---|---|
| 5 | 0.840 | 0.846 ± 0.025 |
| 10 | 0.888 | 0.870 ± 0.024 |
| 20 | 0.930 | 0.895 ± 0.027 |
| **30** | 0.948 | **0.897 ± 0.020** |
| 80 | 0.978 | 0.893 ± 0.012 |
| 200 | 1.000 | 0.895 ± 0.017 |
| 800 | 1.000 | 0.890 ± 0.019 |

::: notes
Unlike bagging, boosting keeps reducing training error for as long as it
runs, because every new tree is built specifically to reduce what remains  - 
including the 7% of labels that are noise by construction. There is nothing
in the algorithm itself that says "stop".

Point at the peak, 30 trees, and then at 800: training accuracy reached
1.000 by 200 trees and stayed there, and cross-validated accuracy has
drifted down from 0.897 to 0.890, under a point, since the peak. Contrast
the size of that drop with the unconstrained single tree of the depth table,
which lost three points past its own peak - boosting's overfitting is real
but gentler.

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
and easy to miss on a single run - which is exactly why early stopping
matters for boosting rather than being optional, even though the overfit is
gentler than a single unconstrained tree's. Say what early stopping is, since
the deck has not: set aside part of the training data, watch the score on it
as trees are added, and stop once it has not improved for a number of rounds.
In scikit-learn that is the `n_iter_no_change` argument of
`GradientBoostingClassifier`.

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
Learning rate and tree count are not independent - both control how much
total correction the ensemble applies, so they trade off against each
other. At $\alpha = 0.02$, 120 trees haven't finished fitting the signal:
training accuracy is still under 0.89. At $\alpha = 1.00$, the ensemble has
memorised the training set completely and given back more than two points
of cross-validated accuracy for it, 0.902 down to 0.878.

The best setting sits in between - large enough to make real progress
within the tree budget, small enough that any one correction is easy to
outvote if it turns out to be wrong. Ask the room which direction they'd
guess is "more regularised" before revealing that it's the *small* $\alpha$
side, the same kind of inverted-direction trap lesson 6's $C$ was.

Handout section 10.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 1. boosting from scratch, 39-43 | `DecisionTreeRegressor(max_depth=2)` on the residuals, 60 times → MSE 0.0119 at 20 trees |
| 2. classification, 44-45 | `GradientBoostingClassifier()` in `cross_val_score` → F0 = −0.461; 0.902 |
| 3. more trees, 46-47 | the same, `learning_rate=0.3`, `n_estimators` from 5 to 800 → peak 0.897 at 30 |
| 4. learning rate, 48 | the same with `n_estimators=120` and `learning_rate` from 0.02 to 1.0 → best 0.902 at 0.1 |
| 5. leaderboard, 50 | every model in one `cross_val_score` loop → random forest 0.911 |

::: notes
The table is the notebook's map: boosting from scratch on the residuals,
then scikit-learn's classifier, the tree-count and learning-rate sweeps, and
the leaderboard of the next slide.

Run Notebooks/03_gradient_boosting.ipynb. Eighteen minutes.

The cell to protect is the residual sweep at 1, 5, 20 and 60 trees - seeing
the ensemble's fit against the true function rise and then fall again, in
real time, is the most direct experience of boosting's overfitting
available in this course.

If time is short, skip the learning-rate sweep live and set it as reading  - 
the table is in handout section 10.
:::

# Every method, on the same five folds

| Model | Cross-validated accuracy |
|---|---|
| Majority baseline | 0.613 |
| Logistic regression | 0.748 |
| Single tree, unconstrained | 0.852 |
| Single tree, `max_depth = 8` | 0.882 |
| Gradient boosting, 30 trees | 0.897 |
| Bagging, 100 trees | 0.904 |
| **Random forest, 100 trees** | **0.911** |
| *noise ceiling* | *≈ 0.93* |

::: notes
Everything from this lesson, cross-validated the same way. Read down the
column and let the pattern speak before naming it: every ensemble method  - 
bagging, the random forest, gradient boosting - lands within about a point
and a half of the others, and all three sit closer to the ceiling than
either single tree.

The ceiling itself: 1 minus the 7% of labels deliberately flipped. No
method reaches it, and none should - a model that classified the flipped
labels "correctly" would be modelling the noise, not the rule.

One honest caveat, in lesson 5's terms. The depth of 8 and the 30 boosted
trees were picked because they scored best on these same five folds, so
those two rows are slightly optimistic; nested cross-validation would remove
that. Bagging and the random forest ran at their defaults, chose nothing,
and are not affected - and they are still on top.

Handout section 11.
:::

# Three findings

- Every ensemble lands within a point and a half of the others: **which
  strategy** matters far less than **whether you ensemble at all**
- The tuned single tree is close behind, 0.882 against 0.911, and the only
  model here a person can read
- The **unconstrained tree is the worst model on the list**, despite being
  the most flexible one

::: notes
Read the three findings as a set - together they are this lesson's real
content, more than any single accuracy number.

The third finding is the one to land hardest, because it repeats a pattern
they have now seen with several different methods: flexibility without a
stopping rule is not an advantage. Ask the room to name the earlier
occurrences before you do - the degree-12 polynomial in lesson 3, k = 1 in
lesson 6, and, earlier today, the boosted ensemble that kept adding trees
past its peak.

Handout section 11.
:::

# How to choose, in practice

| Situation | Reach for |
|---|---|
| The decision must be explained to the person it affects | A single tree, small enough to read; if even that is too opaque, none of these models |
| Tabular data, accuracy matters most, no explanation required | A random forest: the best score here, at its default settings |
| Accuracy matters, and there is time to tune the number of trees and the learning rate together | Gradient boosting, tuned on validation data |
| Many columns, few of them informative | Any tree method, but check its importances by permutation |
| Too few rows to set a validation set aside | A random forest, judged by its out-of-bag score |

::: notes
A table to photograph, exactly as lesson 6 closed with one. Work down it
and attach today's evidence to each row rather than leaving it as generic
advice: the nine-leaf tree for the first row, the leaderboard for the
second, the tree-count and learning-rate tables for the third, the noise
experiment for the fourth, the out-of-bag slide for the fifth.

The fourth row is worth a sentence of its own, tying back to the noise
experiment: with two real features among 22 columns and 1,200 rows, more
than half the importance already landed on noise. Many rows do not protect
against it; what matters is how many useless columns compete at each split.
Permutation importance is on the next slide.

Handout section 11.
:::

# 54% of the importance landed on noise

- 20 noise columns, 2 real features: not a small effect, not a bug
- Each split is offered 4 of the 22 columns: in two splits of three
  (0.662), neither real feature
- Averaging shrinks the **prediction's variance**, not the credit given
  to noise
- Check: **permutation importance**, the accuracy lost when a held-out
  column is shuffled

::: notes
This is the number to carry out of the room. Say it once more, slowly: with
twenty pure-noise columns against two real features, more than half of a
random forest's importance mass sat on columns wired to nothing.

Ask them to write it down - derivations fade, this kind of number tends
not to. Every lesson so far has left one of these - lesson 1's 77% accuracy
on coin-flip labels, lesson 3's coefficient of 365 where an unpenalised fit
wanted billions - and this is lesson 7's.

The second bullet is the mechanism in one number. The square root of 22,
rounded down, is 4, so each split chooses among 4 random columns. The chance
that all 4 come from the 20 noise columns is 4,845 / 7,315 = 0.662. At those
splits the tree still has to pick something, and among 1,200 noisy rows some
noise column always reduces the impurity a little, by chance.

The last bullet, permutation importance, is the remedy the handout points
to: a column the model really uses costs accuracy when its values are
shuffled; a noise column costs nothing. It is measured on data the forest
did not train on, so a column cannot earn credit by having been memorised.

Handout section 7 has the full mechanism.
:::

# What to take away

- A tree splits greedily on Gini impurity, matching scikit-learn
- Depth is the bias-variance dial: accuracy peaked at **depth 8**
- Bootstrap resampling leaves **36.8%** of rows out-of-bag: free validation
- **54% of a random forest's importance landed on 20 noise columns**,
  today's number

::: notes
Close on the same shape as lesson 6's summary: a handful of numbers that
carry the lesson's argument without needing the mathematics behind them
re-derived.

If they remember one habit from today, make it this: before trusting a
random forest's importance ranking, ask how many candidate columns compete
for attention at each split, because that number - not the accuracy score  - 
is what decides how much noise gets credit it didn't earn.

Next lesson turns from labelled data to unlabelled data. Every model so far
learned from examples whose answer was known: a price, a class, a default.
Lesson 8 asks what structure there is in data when nobody has given the
answer at all.

Handout summary and notation table.
:::

# Homework

- **Exercise 7**, discussed at the start of **Friday 20 November**
- `Exercises/07_trees_and_ensembles.md`

::: notes
Set it explicitly and say out loud when it comes back: the first ten
minutes of lesson 8, Friday 20 November.

Remind them of the standing rule from lesson 5: every reported number needs
a cross-validated spread, and any feature-importance claim needs the check
this lesson just demonstrated - a random forest's importance ranking is a
claim to be verified, not a fact to be quoted.

Next week: unsupervised learning, the first lesson in which the data comes
with no labels at all.
:::

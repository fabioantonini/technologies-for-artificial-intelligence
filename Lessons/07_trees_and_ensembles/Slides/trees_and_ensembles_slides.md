---
title: "Lesson 7: Trees and Ensembles"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "20 November 2026"
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

Why "about" 0.93: each label is flipped with probability 7%, so 0.93 is what the
true rule scores on average. In this particular sample 85 of the 1,200 labels were
flipped, so the rule itself scores 1,115 / 1,200 = 0.929, and a cross-validated
score can land a little either side of the ceiling by chance.

Handout section 1.1.
:::

# Four risky rules, three risky regions

![](loan_scatter.png)

::: notes
Let it sit for a moment before pointing anything out. Then trace the shape
out loud. Four rules make an applicant risky: income below a floor, or debt
above a ceiling, or income and debt inside one of two "stressed" rectangles,
where neither reading alone is remarkable. The low-income strip runs up the
left edge and the high-debt strip along the top, and they meet in the top-left
corner, so together they are one L-shaped region. The two rectangles stand
apart, in the interior. Four rules, then, but three separate regions.

A predictable slip: reading the floor and the ceiling as "low income AND high
debt", one corner of the plane, instead of "low income OR high debt", the
whole L. It is a reasonable reading, because the two stressed rectangles really
are an AND of two ranges; the floor and the ceiling are not, each applies on
its own.

Ask for a straight line that separates repaid from defaulted, exactly as you
did with the pumps in lesson 6. This time there is a difference worth
naming: unlike the pumps, the class you would have to isolate is not even
one connected region, so "no line separates them" is true for a different
reason than a ring around a disc.

Handout section 1.1.
:::

# No straight line separates them

![](loan_logistic.png)

::: notes
The figure is the line logistic regression actually draws, fitted on all
1,200 applicants: above it the model predicts default, below it repaid. It
runs from a debt ratio of 0.40 at an income of 20 thousand euros to 0.88 at
100 thousand (notebook 1, section 1). The dashed outlines are the true risky
regions, the L and the two islands.

The title carries the two numbers to compare. Always predicting the majority
class, "repaid", scores 0.613, because 38.7% defaulted and 1 - 0.387 = 0.613.
Logistic regression scores 0.748 ± 0.030 in five-fold cross-validation (the
± is the standard deviation across the five folds). Unlike lesson 6's pumps,
it clears the baseline by a wide margin: it has learned something. Most of the
high-debt strip lies above the line, and the risk there rises steadily as one
feature moves in one direction, which a straight boundary handles well.

Then point at where the line goes wrong. Bottom left: income below the floor
with low debt is risky, but it lies below the line; of the 87 applicants there,
the model predicts 50 as repaid. Far right: at high income the line climbs
above the ceiling, so debt between 0.82 and 0.88 is predicted repaid. And the
line cuts straight through both islands, calling part of each healthy and
the healthy territory beside them risky. Ask the room what kind of boundary could
close those gaps, and let a few guesses land before moving on - a threshold
rule, applied more than once, is exactly the answer the rest of the lesson
builds.

Handout section 1.1.
:::

# The whole algorithm, in one sentence

- Try **every feature and every threshold**; keep the split leaving the
  **purest** two groups
- Apply it; repeat on each group, independently
- Stop at a depth limit, a pure group, or too few examples to split
- No coefficients, no gradient: thresholds are found by **searching**
- **Greedy**: whichever split helps most *right now*, with no lookahead

::: notes
Read the algorithm as one sentence and let it sit - it really is that short.
The first bullet is the one students compress: nobody chooses the feature
first. At every node the tree tries every feature and every threshold and
keeps the best pair; how "best" is scored is the next part of the lesson.
Contrast it explicitly with lessons 3 and 4: no cost function is minimised
over a continuous parameter, there is no gradient step. The tree still learns
from the data - which feature to ask about and where to put each threshold -
but it learns them by trying candidates and keeping the best, not by
descending a slope. A tree is a sequence of yes/no questions, chosen one at a
time.

The word to stress is "greedy". At every step the tree takes whichever split
helps most immediately, with no mechanism for looking two levels ahead. Tell
them to hold onto that word - it is what explains, later this hour, why a
shallow tree can miss the interior islands entirely even though the
information to find them is right there in the data.

A predictable slip: "one feature per level, applied to all the applicants, then
the next feature". It is a natural reading, because a drawn tree looks like a
fixed sequence of questions and the first question really is asked of
everyone. It is wrong in two ways. After the first split each group is
searched on its own, so the two groups can get different questions; and
nothing forbids a feature from coming back - the next slide asks about debt
three times on one path.

The third bullet's stopping rules act branch by branch, not on the whole tree. A
pure group stops its own branch while the others carry on, and the algorithm is
finished when every branch has stopped. Without a depth limit, every branch runs
until it is pure, which is why an unconstrained tree reaches a training accuracy of 1.000.

Handout section 2.1.
:::

# The algorithm, drawn

![](tree_growth.png)

::: notes
The previous slide's sentence, run three times on the loan data. Each column
is one more round; on top the cuts in the plane, the new ones in gold, and
underneath the tree that asks the same questions. Point at the two halves of
each column in turn: every gold line on top is a gold box underneath. The
legend at the bottom gives the colours: gold, a question added in this step;
white, a question from an earlier step; teal or rust, a group not split yet.
Every box, question or group, shows its applicants and the share who
defaulted, and the gain its split buys, ΔG - computed exactly a little later - how much
purer the two groups below are than the group above.

Step 1: every possible single question was tried - every threshold on income
and every threshold on debt, 2,234 candidates - and debt ratio at or below
0.82 is the one that leaves the two groups purest: 971 applicants at 26%
default against 229 at 94%. Do not say why it is the best yet; the next slides build the number that ranks it.

Step 2 is the word "recursive" made visible, and the place where a student
asks where the two groups of step 1 went. They are still there: each group's
box has become the question asked of it. The 971 now reads "income ≤ 28?" with
971 · 26% under it, and splits into 87 + 884 = 971; the 229 reads
"debt ≤ 0.947?" and splits into 160 + 69 = 229. A leaf that is split becomes a
question; its applicants do not go anywhere. The two questions are different -
income on the left, debt again on the right - because each group is searched
on its own, without regard to the other. That is "independently" in the
sentence. Compare the two gains: the income question on the left buys 0.0836,
the debt question on the right 0.0026, about thirty times less. The right one
changes no prediction at all: its two groups, 160 at 96% and 69 at 88%, are both
still mostly defaulted, as the 229 were. The tree makes the split anyway,
because it judges impurity, not decisions, and any gain above zero is taken.

Step 3 is where to slow down, because it shows greed. On the right, one split
sets aside a single applicant from 68 others - the cut at 0.948, a thousandth
above the 0.947 one, so in the plane the two gold lines lie on top of each
other. Someone will ask why the tree asks about debt twice. The two
thresholds are different questions: exactly one applicant has a debt ratio
between 0.947 and 0.948, that applicant repaid, and the tree spent a whole
split on separating them. Another split divides 160 applicants into groups
at 90% and 97% default - both groups are still predicted "default", so that
cut changes no decision at all. Each was the best available move at that node; the tree
has no way to ask whether it was worth making. Add it up: the four splits of
the third level change exactly one prediction in 1,200 - training accuracy goes
from 0.8400 at depth 2 to 0.8408 at depth 3 - and that one applicant is a
flipped label. Income 23.22 is below the floor and debt 0.9475 above the
ceiling, so both rules say default, yet the label says repaid. The tree spent a
split memorising noise. And neither interior island has
appeared yet: three rounds were spent on the easy strips. Hold that thought -
it is the "what greedy misses" slide in ten minutes.

The printout of every node is in notebook 1, section 2. Handout section 2.1.
:::

# How pure is a group of examples?

- A leaf that is all one class needs no more questions
- A group half one class, half the other, is the worst case: a coin flip
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

A predictable slip: "this split is impure". Impurity belongs to a group, not to
a split. A split is then judged by the impurity of the two groups it makes,
weighted by their sizes.

Handout section 2.2.
:::

# Gini impurity

$$G = 1 - \sum_{c=1}^{C} p_c^2 \\ p_c\text{: share of class } c\text{ (defaulted, repaid)} \\ \text{slide 8's 229: } 1 - 0.939^2 - 0.061^2 = 0.115$$

::: notes
Each p in the sum is the fraction of the group belonging to one class,
across C classes. The classes are the labels a group can hold: here two,
defaulted and repaid, so C = 2, and the two shares add up to 1. The third line
works it on a group the room has just seen: slide 8's right-hand group of step
1, before it is split again - 215 of the 229 defaulted, so p = 215/229 = 0.939
for defaulted and 14/229 = 0.061 for repaid; 0.939² + 0.061² = 0.885, and
1 - 0.885 = 0.115. Nearly pure. The single repaid applicant of step 3 is a pure
group: 1 - (0² + 1²) = 0. Do not derive anything here - state what it does: subtract, from
one, the chance two examples share a class if drawn independently from the
group's own frequencies. A pure group scores zero; a maximally mixed one
scores as high as the formula allows.

Give them a moment to notice the formula asks nothing about *which* class is
common, only about how mixed the group is - that symmetry matters later,
when a split is chosen purely by how much it reduces this number.

Handout section 2.2 has the full definition.

A line of colour, if there is a minute: the name is Italian. Corrado Gini
(1884-1965) was a statistician, the first president of the Istituto Centrale di
Statistica in 1926, the body that became ISTAT. 1 - Σ p_c² is his index of
heterogeneity, "mutabilità", from 1912; Leo Breiman and his co-authors made it
the splitting criterion of their CART trees in 1984, and scikit-learn kept it.
It is not the Gini coefficient of income inequality, his better-known index:
same man, a different quantity, and a student who knows the coefficient may
think they are one thing.
:::

# For two classes, one number

- $G = 2p(1-p)$, where $p$ is the share defaulted and $1-p$ the share repaid
- **0** when the group is pure ($p = 0$ or $p = 1$)
- **0.5**, the maximum, at $p = 0.5$
- On the full 1,200 applicants, $p = 0.387$ gives $G = 0.474$

::: notes
Two classes is the case that matters for this dataset, and the formula
collapses: with shares p and 1 - p,
1 - (p² + (1 - p)²) = 1 - p² - (1 - 2p + p²) = 2p - 2p² = 2p(1 - p). Write that
line on the board; it is the only algebra in this segment. The result has an
obvious shape: zero at the extremes, a single
peak at fifty-fifty. Ask the room to sanity-check the peak location before
you confirm it - it is the point where a random guess is least useful.

Give them the concrete number: on the full loan dataset, 38.7% defaulted, so
p = 0.387 and G(parent) = 2 × 0.387 × 0.613 = 0.474. That is the impurity
every candidate root split is measured against.

Handout section 2.2.
:::

# The gain a split buys

- A split divides a parent P into a left child L and a right child R; *m*
  counts each one's examples: at the root, 1,200 in P = 971 in L + 229 in R
- Worth taking if the children's impurities, **averaged**, are lower than
  the parent's: at the root 0.474 against 0.330
- The average is **weighted by size**: one example cut off alone would
  halve a plain average, and barely moves a weighted one

::: notes
Set up the idea in words before the formula: a split is only useful if the
two resulting groups are, taken together, purer than the one group was. At the
root the parent's impurity is 0.474; the left child's is
2 × 0.256 × 0.744 = 0.381 (249 of 971 defaulted), the right child's
2 × 0.939 × 0.061 = 0.115 (215 of 229); averaged with weights 971/1,200 = 0.809
and 229/1,200 = 0.191, 0.308 + 0.022 = 0.330, lower than 0.474. The
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

A predictable slip: subtracting the weighted average from 1 instead of from
the parent's impurity - 1 - 0.330 = 0.670 instead of 0.474 - 0.330 = 0.144 at
the root. It is natural, because the Gini formula three slides back starts with
"1 -". The check that catches it: a split cannot remove more impurity than the
parent had, so a gain is never larger than G of the parent, and with two classes
never above 0.5.
:::

# Scanning for the best split

- Sort a feature's values: every **midpoint** between neighbours is a
  candidate threshold
- Infinitely many thresholds, but **finitely many splits**: any threshold
  between the same two neighbours makes the same two groups
- Score every candidate on **every feature**; keep the best pair, then
  repeat inside each child
- **Greedy**: $\Delta G$ is judged one split at a time, never jointly

::: notes
This is the exhaustive-search part of the algorithm, and it is worth being
concrete about the cost: every feature, every possible threshold, at every
node, all the way down. It is brute force, and on a dataset this size it is
cheap.

The second bullet answers the question someone will ask: a threshold is a real
number, so are there not infinitely many to try? There are, but they make only
finitely many different splits. At the root, two neighbouring debt ratios in
the data are 0.8198 and 0.8205, with no applicant between them; a threshold of
0.8199, 0.8200 or 0.8204 sends the same 971 applicants left and the same 229
right. So one threshold per gap is enough, and the convention is the midpoint,
(0.8198 + 0.8205) / 2 = 0.82015, printed 0.820 - halfway, so the threshold sits
as far as possible from both values seen in training. Debt ratio has 1,139
distinct values in the data, so 1,138 gaps; income has 1,097, so 1,096. The
search is exact, not an approximation. A predictable slip is to think it must
be approximate because thresholds are continuous: the data are finite, so the
splits are too.

The last bullet is the one to slow down on. A feature that only helps in
combination with a second split further down the tree can be - and in this
dataset's two interior islands, is - invisible to a shallow search, because
$\Delta G$ is never evaluated for two splits together.

Handout section 2.2.
:::

# Every threshold, scored

![](threshold_scan.png)

::: notes
The previous slide's search, drawn at the root. Each point on a curve is one
candidate threshold: on the left the 1,138 on debt ratio, on the right the
1,096 on income, 2,234 in all. Its height is the weighted Gini impurity of the
two groups that threshold makes - each group's 2p(1 - p), weighted by its share
of the 1,200 applicants, so lower is purer. The dotted line at 0.474 is no
split at all. The search keeps the lowest point on either curve: debt at 0.82,
0.330. The gain of slide 13 is the drop from the dotted line to that point,
0.474 - 0.330 = 0.144, worked out on the next slide.

Point at the alternatives. Debt at 0.5 splits 627 applicants at 18% default
from 573 at 61%: both groups stay mixed, 0.382. On income, the best threshold
is close to the true floor of 28, but it scores only 0.418, because the
low-income strip holds 111 applicants and leaves the other 1,089 mixed.

A predictable slip: "the best split puts the most defaults on one side". It
is a natural reading of "purest", but the score counts proportions, weighted
by size, not numbers of defaults. Debt at 0.9 makes a right-hand group almost
as pure as 0.82 does, 129 applicants at 92% default against 229 at 94%, yet
scores 0.405 against 0.330, because it leaves the large left group more mixed:
1,071 at 32% against 971 at 26%. And a group pure in "repaid" counts exactly
as much as one pure in "default": the formula does not ask which class is
common.

Notebook 1, section 2, "Every threshold, scored". Handout section 2.2.
:::

# Worked example: the root split

- Parent: 464 of 1,200 defaulted, G = **0.474**; the search finds
  `debt_ratio <= 0.82`, the **debt ceiling**
- Left: 249 of 971 defaulted, p = 0.256, G = 2 × 0.256 × 0.744 = **0.381**
- Right: 215 of 229, p = 0.939, G = 2 × 0.939 × 0.061 = **0.115**
- Weighted: 0.809 × 0.381 + 0.191 × 0.115 = 0.330; gain 0.474 - 0.330 =
  **0.144**

::: notes
Walk the numbers once: parent impurity 0.474, and the winning split is the
debt ceiling, found by brute-force search with no hint about how the data
was generated. Each step starts from a count: 249 of the 971 on the left
defaulted and 215 of the 229 on the right, and 249 + 215 = 464, all the
defaults. The weights are each child's share of the parent, 971/1,200 = 0.809
and 229/1,200 = 0.191. The children's impurities, 2p(1 - p) at their own default
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

Two predictable slips on this slide. The first is rebuilding a count from a
rounded percentage: slide 8 prints the left group as 26% default, and
0.26 × 971 = 252, not 249 - the true share is 249/971 = 25.6%. Counts come
first; a share rounded to two digits loses up to half a percent of the group.
The second is slide 13's: the gain is the parent's impurity minus the weighted
average, 0.474 - 0.330, not 1 - 0.330.

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
the dotted straight line between them. Someone will ask why it sits exactly
under the parent, at p = 0.387. Because the children's p, averaged with the
same weights, gives the parent's p back, and with counts it is exact:
971/1,200 × 249/971 + 229/1,200 × 215/229 = 249/1,200 + 215/1,200 =
464/1,200 = 0.387. (With the rounded 0.809 × 0.256 + 0.191 × 0.939 it comes to
0.386: rounding again.) A split moves defaulters from one group to another; it
does not create or remove any. So parent and square share one horizontal
position, and the vertical gap between them is the gain. Point at the bulge: the curve lies above
every such line, so any split that separates the classes at all lowers the
weighted impurity. The gap, 0.144, is the gain, and the tree takes whichever
split makes it largest. Handout section 2.2.
:::

# The loop, in seven lines

```python
def grow(g):
  if stop(g): return leaf(g)
  gain = {c: ΔG(g, c) for c in splits(g)}
  best = max(gain, key=gain.get)
  if gain[best] <= 0: return leaf(g)
  L, R = split(g, best)
  return node(best, grow(L), grow(R))
```

::: notes
Slides 7 to 17 in one place, as the code of notebook 1 runs it (`build_tree`
and `best_split`, section 2). The names, line by line: g is the group being grown; stop(g) is true when the
group sits at the depth limit, is pure, or has fewer than 2 examples (slide 7);
leaf(g) is a leaf that predicts the group's majority class; splits(g) lists
every candidate, a feature with one of its midpoint thresholds (slide 14);
ΔG(g, c) is slide 13's formula on the two groups candidate c would make; max
with key=gain.get picks the candidate with the largest gain; split makes the
two groups, L and R as on slide 12; node keeps the question and its two
subtrees. Read it top to bottom with the room and point at where
ΔG does its two jobs, both at one node at a time. It ranks that node's
candidates - the largest wins. And it decides whether to split at all - if
nothing gains, the node is a leaf.

Then the recursion, which is the line students skip: each child calls grow on
its own examples, with its own G as the parent's and its own size as m. ΔG is
never added up across nodes or compared between them in this loop - `max_leaf_nodes`, met later in the lesson, is the exception, and compares them on purpose. Walk slide 8's tree
through it: the root, 2,234 candidates, the best gains 0.1438; the left group of
971 runs the loop again and its best gains 0.0836; the right group of 229 runs
it and its best gains 0.0026, small but above 0, so the split is made; at depth
3 every call stops on the first line.

Handout section 2.1.

What the two return, because the code hides it. leaf(g) is a new object that
keeps one thing, the majority class of g, as its prediction; it does not keep
the examples. node(best, grow(L), grow(R)) is a question node holding three
things: the question, and the two subtrees. Python evaluates the arguments
first, so grow(L) and grow(R) run all the way down before the node that holds
them is built: the tree is assembled from the leaves upward, depth first. At the
root that is node((debt, 0.82), grow(the 971), grow(the 229)). And max with
key=gain.get returns the candidate whose gain is largest, not the gain;
gain[best] is the gain. Two predictable slips follow from reading too fast:
"the function returns g" (it returns a leaf built from g), and "max gives the
best gain" (it gives the best candidate).

Training builds the tree; inference walks it. To predict for a new applicant,
start at the root: at a question node go left if the applicant's value is at or
below the threshold, right otherwise, until a leaf, and return its class -
notebook 1's predict_one. The costs are lopsided: training scores every
candidate at every node; a prediction is at most max_depth comparisons, 8 for
the best tree here, however large the training set. Lesson 6's k-nearest
neighbours is the opposite: nothing to train, and every prediction compares the
new point with all the training points.
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
Identical predictions do not by themselves prove the two trees are identical
node by node - two different trees can agree on every row they are shown - but
the notebook also prints the same root split, debt ratio at 0.82, from both.
And why on the training data, after lesson 5? Because this checks the code,
not the model: the question is whether two programs make the same decisions
from the same rows, and the rows they learned from are exactly the right ones
to ask it on. No accuracy here is a claim about new applicants.

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

Greedy is short-sighted, not blind. Each edge of an island pays a little on
its own, so with enough depth the tree finds the islands one edge at a time:
slide 8's third step already cuts the 884 at debt 0.404, next to island A's
lower edge at 0.40, and by depth 8 both islands are drawn. Greedy
fails outright only where a single cut pays nothing at all - a chessboard, or
the exclusive-or (XOR) of two features, where every first cut leaves both
sides exactly as mixed as before.
:::

# Depth is the bias-variance dial

- `max_depth`: the most questions **one applicant** answers
- **Shallow tree**: few questions, only the coarsest structure. High bias,
  low variance
- **Deep tree**: enough questions to isolate single points. Training
  accuracy **exactly 1.000**, even on data with no structure, unless two
  examples share every feature value but not the label

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

Two intuitions that land, both from notebook 1's depth table. Who votes in a
leaf: a leaf predicts the majority of its applicants. At depth 2 there are 4
leaves with a median of 123.5 applicants each, a vote of a hundred; the
unconstrained tree has 156 leaves with a median of 2, and 72 of them hold a
single applicant - a prediction made from one person's label, lesson 6's k = 1.
And the line at 0.929: the true rule itself scores 1,115/1,200 = 0.929 on these
labels, because 85 were flipped. A training accuracy above that line means the
tree agrees with the labels better than the truth does, so it must be
reproducing flipped labels: depth 8 is just past it at 0.951, the unconstrained
tree has learned all 85. If an
image helps: a checklist of three questions written from last year's customers
works on this year's; fifty questions, each written about one past customer,
do not.
:::

# Measured, at seven depths

| `max_depth` | Training accuracy | Cross-validated, 5 folds (mean ± standard deviation) |
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

What "cross-validated" and "±" mean here, since the table leans on lesson 5.
Five-fold stratified cross-validation: the 1,200 are cut into five folds of 240,
each with the same default share; five times, the tree trains on 960 and is
scored on the 240 it did not see. At depth 8 the five scores are 210, 211, 215,
214 and 209 correct out of 240: 0.875, 0.879, 0.896, 0.892, 0.871. The table
shows their mean, (0.875 + 0.879 + 0.896 + 0.892 + 0.871) / 5 = 0.882, and
after ± their standard deviation, 0.010 - how much the score moves from one
fold to another. It is not a confidence interval.
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

Why the blue curve may cross the gold line and the rust one may not. The
ceiling is the best a model can expect on new applicants, whose flipped labels
it cannot know. On its own training data it can do better, by memorising them:
the crossing, just before depth 8, is where memorising starts, and at 1.000 all
85 flipped labels are learned. A predictable slip: "a score above the noise
ceiling means a bug". True for new data, false for training data. Two reading
details: the shaded band is ± one standard deviation across the five folds, and
the plot has eleven depths where the table had seven; "20" on the axis stands
for the unconstrained tree, whose real depth is 24.
:::

# The same rule, three depths

![](tree_boundaries.png)

::: notes
Left to right: at depth 2 every applicant answers at most two questions. The
tree asks three in all - it is exactly the tree of step 2 on slide 8 - but the
third, on the right, splits 229 high-debt applicants into two groups that are
both predicted "default", so it changes no prediction and draws no line. What
the boundary shows is the income floor and the debt ceiling - the two biggest,
easiest-to-find regions - and neither interior island is visible at all. At depth 8 both islands appear, close to
the rectangles of the rule that generated the data though not exact, and a few
thin slivers have already started - its training accuracy, 0.951, is past the
0.929 line. At full depth the boundary has grown a fringe of tiny rectangles
chasing individual flipped labels, in both colours: rust ones in healthy
territory and teal ones inside the risky strips, where a flipped "repaid" label
sits - slide 8's applicant at income 23.22 and debt 0.9475 is one, in the
top-left corner.

Make the point explicit: the full-depth boundary is not a smoother version
of the depth-8 boundary. It is a noisier one, built out of the same
mechanism that gave the unconstrained tree its perfect training score.

Ask the room which of the three they would trust on a loan applicant they
had never seen. Handout section 3.
:::

# Reading a tree, and the knob that keeps it small

- `max_depth` **caps every path**: each node above the cap is split if it
  can be, worth it or not
- `max_leaf_nodes` is a **budget of leaves**, spent on the largest
  $\Delta G$ first
- `max_leaf_nodes = 9`: **0.908** training, **0.901** cross-validated, above
  depth 8's 0.882; thresholds 28 and 0.82
- In code: `export_text(tree)` prints the tree as if/else rules

::: notes
This is the readability half of the story. The two knobs limit different
things. A depth cap limits how long any one path can be, and inside that cap
every node that can be split is split - slide 24's depth-2 tree spent one of
its three questions on a split that changed no prediction. A leaf budget
limits how many leaves the whole tree may have, and spends them one at a time
on whichever leaf gains most - compared across all the current leaves, each
leaf's gain weighted by its share of all the applicants. It is the one place
where gains at different nodes are compared, which slide 18's loop never does. For a tree meant to be read rather than merely
scored, the budget is the better knob: a small number of splits goes to the
ones that matter most.

The nine-leaf tree is worth reading aloud as a sequence of if/else
statements - that is the entire model, and it is the thing k-nearest
neighbours and support vector machines cannot offer at all. For a decision
that has to be justified to the person it affects, that can matter as much
as accuracy.

Handout section 4.

The cross-validated 0.901 is the surprise to dwell on: nine leaves spent where
they gain most beat 71 spent evenly by the depth-8 tree (0.882), and come
within a point of bagging, as the leaderboard at the end of the lesson shows. It is this dataset: its rule is made of
axis-aligned rectangles, exactly the shape a few well-placed thresholds draw.
The 9 was chosen for readability, not by trying budgets on the folds, so it is
not tuned to them.
:::

# A tree that doesn't stay still

- Two trees at `max_depth = 6`, each on 780 of the 1,200 applicants, drawn
  **without replacement** (no one twice)
- The root, decided on all 780, agrees to within **0.001**: the debt ceiling
- The 29 or so questions below it, each on fewer applicants, do not: **31
  and 33 leaves**, **opposite verdicts for 140** of the 1,200 (11.7%)

::: notes
This is the instability that ensembles turn into raw material. The
strongest split barely moves between the two samples, 0.820 against 0.819,
because the evidence behind it is overwhelming: inside each sample of
780, the debt ceiling separates 636 applicants at about 25% default from 144
at 94%, and a contrast that size does not change from one sample to the
next. The two thresholds differ only because each is the midpoint between
the two neighbouring debt values its sample holds (slide 14): sample A has
applicants at 0.8198 and 0.8205, midpoint 0.82015; sample B lacks the one at
0.8198, so its neighbours are 0.8182 and 0.8205, midpoint 0.81935. Both send
every applicant of their sample the same way. Every later split is made
inside a smaller group, supported by far fewer applicants, and those splits
are what actually determine a tree's shape: 31 leaves mean 30 questions, one
at the root and 29 below it.

"Without replacement" means each of the 780 is a different applicant, like
cards dealt without putting them back; the two samples share 499. With
replacement, a card goes back and can come out twice. Ask the room what 140 opposite verdicts between
two trees fit to overlapping samples of the same applicants implies about trusting any single tree's
exact boundary. The
answer: no single tree's exact boundary deserves much trust, and that is
the problem the rest of the lesson takes on.

Handout section 4.
:::

# Two trees, 140 different verdicts

![](two_trees_disagree.png)

::: notes
The previous slide's two trees, drawn. Left and middle: their boundaries, from
two random samples of 780 drawn from the same 1,200 applicants. The strips agree:
both find the income floor and the debt ceiling. The interior does not: tree A
draws one long band across both islands, tree B finds island A and misses
island B. Right: the 140 applicants they give opposite verdicts to, in gold.

Let the room notice where the gold points sit before saying it: three in four
of them in or right next to the two islands (105 of 140; 75 inside them),
where each split was decided by a few dozen applicants; the rest lie along the
edges of the two strips (11 below the income floor, 11 above the debt
ceiling). That is a tree's variance, made visible.

Notebook 1, section 4. Handout section 4.
:::

# Notebook 1, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 1. look at it first, 4-6 | `make_pipeline(StandardScaler(), LogisticRegression())` in `cross_val_score(..., cv=StratifiedKFold(5))` → 0.613 against 0.748 |
| 2. a tree, split by split, 7-20 | your own `gini` and `best_split`, then `DecisionTreeClassifier(max_depth=d)` → 100.0% identical predictions; gain 0.1438 |
| 3. the depth dial, 21-24 | `DecisionTreeClassifier(max_depth=d)` in `cross_val_score` → peak 0.882 at depth 8 |
| 4. what a tree costs, 25-27 | `DecisionTreeClassifier(max_leaf_nodes=9)`, `export_text` → 0.908 training, 0.901 cross-validated; two trees, 140 opposite verdicts |

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
comparison at the end: the notebook prints an agreement of 0.883 and 140
opposite verdicts, the numbers of slides 26 and 27, and draws slide 27's figure;
seeing it appear on their own screen lands harder than reading it off a slide.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. The next segment is where the lesson's real content starts  - 
one tree's instability, turned into an ensemble's strength - so they need to
come back with that framing fresh.
:::

# Bagging: many trees, one vote

- Slide 27: two trees on samples of the same applicants disagreed on 140
  of 1,200
- So train **many trees**, each on its own sample, and **vote**: one
  sample's bad luck is outvoted
- Each sample: $m$ rows drawn **with replacement** from the $m$ training
  rows, some twice, some never: a **bootstrap sample**
- Fully grown trees and a majority vote: **bagging**

::: notes
State the logic before the name. Slides 26 and 27, before the break, measured
that two trees fit to overlapping samples give opposite verdicts on 140 of the
1,200 applicants. If those disagreements come from the luck of each sample,
many trees combined by a vote should cancel them.

Why a vote cancels them, one applicant at a time. A tree's verdict on an
applicant depends on two things: what the data really say in that part of the
plane, which is the same for every tree, and the luck of its own bootstrap
sample, which is different for every tree. A mistake caused by luck falls on
different applicants for different trees, so on any one applicant the unlucky
trees are a minority, and the vote outvotes them. Notebook 2, section 1, on its
360 test applicants: the applicant at income 39.22 and debt ratio 0.9721, who
defaulted, gets "default" from 69 of the 100 trees and "repaid" from 31, tree 1
among them; the vote is right. Over all 360, one tree alone is wrong on 58.7 on
average; the vote gets 18.9 of those right; the vote itself is wrong on 46.

A predictable mistake: "the trees cancel each other's errors". Reasonable,
because the picture of slide 27 is all disagreement. But the vote cancels only
the mistakes that differ from tree to tree. A flipped label is not one: on a new
applicant whose label was flipped, the trees have learned the rule that
generated the data (slide 5) from its neighbours, and on average 95 trees in 100
vote what the rule says. Of the 29 flipped labels among the 360 test
applicants, the vote gets 1 right.

The one-line framing to leave them with: bagging averages what an unstable
tree would have answered on many plausible datasets. The bootstrap imitates
collecting the loans again, without new data. It does not create the
disagreements; a single tree already has them, unseen. So bagging helps only
unstable models: on the same split a depth-2 tree scores 0.806, and a bagging of
100 depth-2 trees also 0.806 - those trees barely differ, and there is nothing
to average; the unconstrained tree goes from 0.825 to 0.872.

A comparison to put to the room: is bagging to a deep tree what cross-validation
is to logistic regression? No. Cross-validation measures how much a result
depends on the sample; bagging reduces that dependence. The closer analogy is
Ridge on lesson 3's degree-12 polynomial: a flexible, unstable model and a
remedy for its variance - Ridge by restraining the coefficients, bagging by
averaging many models. Neither touches the bias.

The m rows are the training rows - in notebook 2, the 840 of a 70/30 split.
Drawing m of them with replacement gives every tree as many rows as a single
tree would have, while every sample is different. The trees are independent of
one another, so they can be grown in parallel (n_jobs). scikit-learn's default
is n_estimators=10; notebook 2 uses 100. scikit-learn averages the trees'
predicted probabilities rather than counting votes; with fully grown trees every
leaf is pure, so the two are the same.

Handout section 5.

History, in one line: bagging is Leo Breiman's, published in 1996 under the
title "Bagging predictors" - the same Breiman who co-wrote the CART trees of
1984 and, five years later, the random forest.
:::

# Bagging, drawn

![](bagging_drawn.png)

::: notes
The previous slide as one picture, from notebook 2, section 1: the 840
training loans of its first split. How it is made, in four steps. A bagging of
100 trees is trained on the 840. Top row: for each of the first three trees,
the loans its bootstrap sample drew, as dots in the income-debt plane coloured
by outcome; a dot is bigger the more times that loan was drawn, and loans never
drawn are simply absent. Every sample is 840 draws; the number of different
loans varies by luck around 840 x 0.632 = 531 - 535, 518 and 542 here. Bottom
row: each tree is asked about a dense grid of imaginary applicants covering the
whole plane, and every point is coloured by its verdict: that is the tree's
decision map, not its training loans. The last panel is the same map for the
vote of all 100.

Let the room compare the three trees first: three samples of the same 840
loans, three different shapes - slide 27 again, now a hundred times over. Then
the vote: the two strips and the two islands survive, and most single-tree
slivers do not, because a sliver one tree draws, most of the others do not.

Handout section 5.
:::

# How much data one bootstrap sample sees

- A draw picks a given row with probability $1/m$, so it misses it with
  $1 - 1/m$; all $m$ independent draws miss it with $(1 - 1/m)^m$, about
  $e^{-1} = 0.368$
- So each sample holds about **63.2%** of the different rows; the other
  **36.8%** are that tree's **out-of-bag (OOB)** rows, which it never sees
- Still $m$ draws: at $m = 1{,}200$, on average
  $1{,}200 \times (1 - 0.3677) = 759$ different rows, and $441$ repeated draws

::: notes
Give the one line of reasoning, not the limit. At m = 1,200 the probability
of being missed is 0.3677; as m grows it tends to 1/e, 0.368, the standard
exponential limit worked out in handout section 5.1.

A predictable slip: "so each tree is trained on 63.2% of the data, 759 rows".
It is reasonable, because 63.2% is the number on the slide. But the sample
still has 1,200 draws: about 759 different rows, and the other 441 draws pick a
row already drawn. The tree sees 1,200 rows, some of them twice or three times.

A second one: "the out-of-bag rows are left out of the training set". The word
"out" makes it sound like a test set. But out-of-bag is decided tree by tree:
the row is missing from this tree's sample, and the other trees, about 63% of
them, do see it.

One identity worth saying: the 1,200 draws are the different rows plus the
repeated draws, and the 1,200 rows are the different rows plus the rows never
drawn; so the repeated draws are exactly as many as the out-of-bag rows. Every
repeat takes the place of a row left out. The rows drawn twice or more are
fewer, about 1,200 x 0.264 = 317, because some take three or four draws. In
notebook 2 the bagging is trained on 840 rows, so 840 x 0.632 = 531 different
loans per sample, as on the previous slide.

Handout section 5.1.
:::

# One bootstrap sample, counted against the theory

![](bootstrap_counts.png)

::: notes
Each pair of bars is one value of k, the number of times a loan is drawn into
the sample. A bar's height is the share of the 1,200 loans drawn exactly k
times: blue in one real sample, gold the theory for large m. The k = 0 bars are
the loans never drawn, the out-of-bag ones: 0.366 in this sample, 0.368 in
theory. A bar times 1,200 is a count: 0.168 at k = 2 is about 202 loans drawn
exactly twice.

Why "once" is as common as "never", if asked. Think of 1,200 balls thrown at
random into 1,200 boxes: on average one ball per box. Never: all 1,200 draws
miss the loan, (1 - 1/1,200) to the 1,200th = 0.3677. Once: one of the 1,200
draws picks it - 1,200 choices, each with chance 1/1,200, so 1 in all - and the
other 1,199 miss it: (1 - 1/1,200) to the 1,199th = 0.3680. The two differ by a
single factor 0.9992. With a mean of one draw per loan, "once" and "never" are
equally likely; with 2,400 draws, "once" would be twice as likely. The general
formula, e^-1 / k!, is in handout section 5.1; the slide needs only these two.

The point to make: about one loan in four appears two or more times. Every
tree sees a different mix of repeats and absences, and that is where their
disagreement comes from.
:::

# Out-of-bag: each loan judged by the trees that never saw it

![](oob_grid.png)

::: notes
Ten bagged trees, trained on all 1,200 loans, and the first eight loans. Read
one row before explaining the rule. Grey cells: the loan is in that tree's
sample - "seen x2" means drawn twice - so that tree does not vote on it: it is
a student who has already read the answer. Gold cells: the tree never saw the
loan, and its vote is written in, D for defaulted, R for repaid. The out-of-bag
verdict is the majority of the gold votes, compared with the true label.

Loan 0: only tree 9 never saw it; it votes D, the loan defaulted, right. Loan 3:
trees 3, 5, 6, 7 and 10 never saw it and all vote R, but it defaulted - it sits
inside the second stressed island, which those trees missed. A tie, if it
happens, goes to "repaid": scikit-learn takes the class with the highest
average, and an exact tie goes to the first class.

Do this for all 1,200 loans and count how often the verdict is right: that share
is the out-of-bag accuracy. Every loan is judged only by trees for which it is
unseen data, so it works like a validation set - with nothing set aside and
nothing trained again. With 10 trees about 0.368 x 10 = 3.7 vote per loan.

Handout section 5.1.
:::

# Out-of-bag against cross-validation

- How accurate is a bagging of 300 trees on loans **it never saw**? All
  1,200 loans train it
- **Out-of-bag**: each loan judged by the trees that never saw it (about
  $0.368 \times 300 = 110$): **1,090** right, accuracy **0.9083**
- **5-fold cross-validation** (lesson 5, next slide): **1,083** right,
  accuracy **0.9025** ± 0.021
- They agree within 0.006; out-of-bag needs **no loans set aside, no extra
  training**

::: notes
Both numbers are accuracies: the share of loans whose default-or-repaid verdict
is right. Out-of-bag, each loan is judged by the trees whose sample missed it -
110.5 of the 300 on average, from 81 to 133, a different set for each loan - and
the trained bagging already contains everything needed: scikit-learn reports it
as oob_score_. Cross-validation trains five more baggings, one per round. That
is why the comparison matters: cross-validation is lesson 5's trusted check,
and out-of-bag nearly matches it for free.

Not to be read to the fourth decimal: over twelve seeds the gap between the two
averages 0.0040 (handout section 5.1).
:::

# Cross-validation, step by step: five groups of 240

| Round: the group left out | Right of its 240 | Accuracy = right ÷ 240 | Deviation = accuracy − 0.9025 | Deviation² |
|---|---|---|---|---|
| 1 | 212 | 0.8833 | −0.0192 | 0.000367 |
| 2 | 215 | 0.8958 | −0.0067 | 0.000044 |
| 3 | 223 | 0.9292 | +0.0267 | 0.000711 |
| 4 | 222 | 0.9250 | +0.0225 | 0.000506 |
| 5 | 211 | 0.8792 | −0.0233 | 0.000544 |
| **Mean** = sum ÷ 5 | 1,083 | **0.9025** | | sum 0.002174 |
| **Standard deviation** = √(0.002174 ÷ 5) | | **0.021** | | |

::: notes
The 1,200 loans are split into 5 groups of 240. In each round a bagging of 300
trees is trained on the other four groups, 960 loans, and its verdicts are
counted on the group left out. Every loan is judged once, by a bagging that
never saw it.

The mean of the five accuracies, 0.9025, equals 1,083 / 1,200, because the
groups are the same size. The plus-or-minus is the standard deviation of the
five accuracies: each one's deviation from the mean, squared, summed to 0.002174,
divided by 5, square root, 0.0208. One detail a student redoing it by hand will
meet: numpy's np.std, which scikit-learn's examples use, divides by 5; some
statistics texts divide by 4, which gives 0.023.

Handout section 5.1.
:::

# Why voting helps, and its floor

- A tree's verdict depends partly on the **luck of its sample**, different
  for each tree: a vote cancels it
- All trees grow from the **same 1,200 loans**: what those loans say, every
  tree learns, and that **does not cancel**
- Ten experts reading one newspaper: quirks cancel, its errors stay
- So voting stops at a **floor**; lower needs trees **less alike**

::: notes
This is the intuition; the formula is in handout section 5.2 for those who
want it, and it says these words in symbols. Two kinds of luck go into each
tree. Which bootstrap sample it drew is private to that tree, so many trees
cancel it. Which 1,200 loans the lender happened to collect is shared by every
tree: had the lender collected another 1,200, all the trees would have moved
together. That shared part is the floor. The next slide measures it: with
another 1,200 loans, one tree changes its verdict on 14.5% of the applicants, a
vote of 99 trees on 3.9%.

The same reason explains what bagging cannot fix: a mistake every tree makes -
the flipped labels of slide 30, or a tree too shallow to represent the rule -
is shared, so it stays.
:::

# Another 1,200 loans: how many verdicts change?

![](verdict_change.png)

::: notes
Measured, not computed. The generator can draw fresh datasets, as if the lender
had collected another 1,200 loans: notebook 2 draws 40 of them, trains a bagging
of 100 trees on each, and asks every one the same 300 imaginary applicants
spread over the plane. For an ensemble of the first 1, 3, 5, up to 99 trees, it
counts on how many applicants the verdict differs between two datasets, averaged
over all pairs of datasets.

Read the curve left to right. One tree: the verdict changes on 14.5% of the
applicants, about one in seven. Nine trees: 6.2%. Ninety-nine trees: 3.9%, and
from about 50 trees on the curve is flat. That flat part is the floor of the
previous slide: what every tree shares because every tree grew from the same
1,200 loans. Ask the room what adding trees past fifty buys - almost nothing -
and what the only remaining lever is: trees that are less alike.

Handout section 5.2, where the same floor is also derived as a formula.
:::

# Bagging against one tree, on 30 splits

![](bagging_vs_tree.png)

::: notes
Read the procedure strip first, because the result means nothing without it:
the 1,200 loans are split at random into 840 for training and 360 for test;
one unconstrained tree and a bagging of 100 trees are both trained on the same
840 and scored on the same 360; and this is repeated with 30 different random
splits. Each dot is one split's test accuracy; the red line is the mean of the
30; a grey line joins the two models on the same split.

The message is the mean: 0.856 against 0.902, so on 360 test applicants
bagging gets (0.902 - 0.856) x 360 = 16.6, about 17, more right. And split by
split: every grey line rises - bagging is ahead on all 30 splits, by 3 to 33
applicants. Why: the single tree's slivers and memorised flipped labels (slide
24) are outvoted. The dots are also a little closer together, standard
deviation 0.0153 against 0.0185; much of that spread is the luck of which 360
applicants land in the test set, which no model removes.

Handout section 5.2.
:::

# Random forests: trees less alike

- Bagging: every split may use **every column**, so all 100 trees start on
  `debt_ratio`
- **Random forest**: bagging, but each split may use only `max_features`
  random columns, **redrawn at every split**
- Default $\sqrt{n}$ rounded down, $n$ = number of **columns**; here $n = 2$:
  **one** per split
- A tree is **worse** alone (0.828 vs 0.837), the forest **better**

::: notes
The intuition first: a committee of experts who all look at the strongest clue
first reach the same verdicts and make the same mistakes, so averaging them is
like asking one. Force each, at every decision, to choose among a few clues
drawn at random, and their reasoning diverges: each is a little worse alone,
the committee is better. A crowd is wise only when its members err
independently.

A predictable slip: "n is the number of classes". Reasonable, because here
both are 2. But n counts the columns: with the 22 columns of the noise
experiment coming up, each split gets the square root of 22 rounded down, 4
columns, while the classes are still 2. With one column offered, the split
still picks the best threshold on that column; and because the draw is redone
at every split, a tree denied debt_ratio at the root can use it further down.

Notebook 2, section 4, counts the roots on the first of the 30 splits: all 100
bagged trees open on debt_ratio; the forest's open 51 on debt_ratio and 49 on
income_k. The trade, because someone will ask: each forest tree alone is a
little worse, 0.828 against 0.837 mean test accuracy, since some splits are
denied the better column. A predictable mistake is to conclude the forest must
be worse. The reasoning is right about each tree and wrong about their vote:
what a vote needs is trees that err differently, not trees that are each the
best possible.

In code, RandomForestClassifier() grows 100 trees by default. Handout section 6.

History: Leo Breiman's paper "Random Forests" appeared in 2001 (it is in the
handout's further reading). Breiman, a Berkeley statistician who had spent years
as an industrial consultant, also wrote that year's "Statistical Modeling: The
Two Cultures", a defence of judging models by how well they predict.
:::

# Bagged trees against forest trees

![](forest_vs_bagging_trees.png)

::: notes
Top row bagging, bottom row the random forest, both trained on the first of
the 30 splits (notebook 2, section 4). The three bagged trees are its first
three; all open on debt and draw the same two strips. The three forest trees
are chosen to show both cases: the first tree that opens on debt and the first
two that open on income. They fill the plane differently: tree 2, offered only
debt at many of its splits, is nearly all horizontal bands.

On the right, where each ensemble's 100 trees split their vote - darker where
the split is closer to half and half. Then the count, the same one slide 27
made for two trees: two bagged trees give opposite verdicts on 38 of the 360
test applicants, on average over all pairs; two forest trees on 50. Less alike,
so their mistakes fall in more different places, and the vote has more of them
to cancel.

Handout section 6.
:::

# Random forests, measured

![](ensemble_variance.png)

::: notes
Left: the same 30 random splits as slide 39, now with the random forest added
(notebook 2, section 3). Each dot is one split's test accuracy; the box holds
the middle half of the 30, and the red line is the mean. The means are 0.856,
0.902 and 0.908, the standard deviations 0.0185, 0.0153 and 0.0137. The step
from one tree to bagging is large, about 17 more of 360 right; the step from
bagging to the forest is small, about 2 more. The vote does most of the work;
making the trees less alike adds a little more.

Right: the measurement of slide 38, with the forest added. With another 1,200
loans, the vote of 99 bagged trees changes its verdict on 3.9% of the
applicants, the forest's on 2.6%: the forest's floor is lower. At the left end
of the curves one forest tree alone changes more than one bagged tree - each
forest tree is less steady, the forest steadier.

Say plainly what has and hasn't changed: nothing about the splitting rule is
different from the first hour's trees. The only change across the three boxes
is what surrounds the tree - resampling, then also restricting the columns each
split may use.

Handout section 6.
:::

# What feature importance measures

- **Feature importance**: how much each column helped the splits. Each
  split's gain ΔG, times its share of applicants, goes to its column; the
  sums are scaled to add up to 1
- Here two real columns share it: **debt 0.61, income 0.39**
- **Predict**: add columns of random numbers, unrelated to default. How
  much credit should they get?

::: notes
Work the definition through once, on the depth-2 tree of slide 8, so that the
word "credit" means something. Each split's credit is its gain times its weight,
the weight being the applicants it splits divided by 1,200. Root, debt ratio at
0.82, 1,200 applicants: gain 0.1438, weight 1, credit 0.1438. Left, income at
28, 971 applicants: gain 0.0836, weight 0.8092, credit 0.0676. Right, debt ratio
at 0.947, 229 applicants: gain 0.0026, weight 0.1908, credit 0.0005. The handout
sets this out as a table.

Debt gets 0.1438 + 0.0005 = 0.1443, income 0.0676, total 0.2119; divided by the
total, debt 0.681 and income 0.319 - exactly what tree.feature_importances_
returns. A forest does this in every tree, averages, and scales again: a
300-tree forest on the 1,200 loans gives debt 0.608 and income 0.392.

Then set up the experiment. Ask the room to predict what happens as you add
columns of pure noise - np.random.normal, wired to nothing. Most will predict the
noise columns get close to zero importance, because averaging over hundreds of
trees should wash out anything with no signal. Do not confirm or deny yet - the
next slide is the answer.

Handout section 7.
:::

# More columns, more noise

![](noise_importance.png)

::: notes
Say what is plotted before pointing at it (notebook 2, section 5): a
300-tree random forest on the loan data with 0, 5 and 20 columns of random
numbers added. Red, left axis: cross-validated accuracy. Blue, right axis:
the share of the forest's importance that lands on the noise columns.

Follow the blue line, slowly, and let the last point sit before explaining
anything: 0% with no noise, 33.9% with 5 columns, 54.5% with 20. With 20
pure-noise columns against 2 real features, more than half of the forest's
own importance scores land on columns that are `np.random.normal` noise by
construction. The red line falls at the same time, from 0.912 to 0.837, but
gently: nothing in the accuracy warns that the importances have gone wrong.

Name the instinct this breaks, and defend it before moving on: expecting an
average over hundreds of trees to wash out noise is entirely reasonable  - 
the variance-reduction result of the averaging slides is real and measured, and it
does point the same way. It reduces the variance of the *prediction*. It
does not guarantee small importance for irrelevant columns. Two things hand
credit to noise, and the lesson comes back to take them apart: the random, incomplete menu
at each split, and the deep splits on small groups, where some noise column
always lowers the impurity a little by chance.

The mechanism is in handout section 7. The practical
consequence to leave them with: treat a random forest's importances as
suggestive, not definitive, when real features are few against many
candidate columns - and check them, which is the next slide.
:::

# Permutation importance

![](permutation_importance.png)

::: notes
Name the method and say its recipe before pointing at the figure.
**Permutation importance**: take a fitted model and data it was not trained
on; shuffle the values of one column, so that column no longer says anything
about the applicant in its row; score the model again. The accuracy lost is
that column's importance. Repeat for every column. If the model never really
used a column, shuffling it costs nothing.

The figure says it on top: the 20-noise-column forest, trained on 70% of the
applicants and checked on the other 30% (notebook 2, section 6). This forest
puts 54.8% of its own importance on noise; the 54.5% of the previous slide is
the same kind of forest trained on all 1,200 - two fits, two close numbers. Left, the importance
the forest reports for itself: more than half of it on the twenty noise
columns. Right, the same 22 columns by permutation: debt ratio costs 0.18 of
accuracy when shuffled, income 0.09, and every noise column costs almost
nothing - between 0.003 lost and 0.002 gained.

Two things to stress. Almost nothing, not exactly nothing: shuffling is
random and the test set is finite, so a useless column moves the score by
a few thousandths either way. And held-out data is what makes the check
honest: on the training rows a deep forest has memorised its noise columns,
and shuffling them would cost something there.

The check costs one function call. The lesson to leave: a ranking the model
reports about itself is a claim; shuffling on held-out data is the test of it.
Handout section 7.1.
:::

# Notebook 2, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 1. bagging from scratch, 30-31 | `train_test_split`, your bagging against `BaggingClassifier`, `accuracy_score` → 0.872 both; 1 of 29 flipped right |
| 2. why averaging works, 32-38 | `oob_score=True`: `.oob_score_` against `cross_val_score` → 0.9083, 0.9025 |
| 3. variance, measured, 39, 42 | tree, bagging, `RandomForestClassifier`, 30 splits → 0.856, 0.902, 0.908 |
| 4. what a forest adds, 40-41 | `.estimators_`, `tree_.feature[0]` → roots 100, 51 and 49 |
| 5. feature importance, 43-44 | `.feature_importances_`, 0 to 20 noise columns → 54.5% on noise |
| 6. checking the importances, 45 | `permutation_importance`, test split → real 0.09 and 0.18, noise ≈ 0 |

::: notes
The table is the notebook's map: bagging from scratch and against
scikit-learn, the out-of-bag score checked against cross-validation, the
variance measured over 30 splits, the noise-column experiment, and the
permutation check of the previous slide.

Run Notebooks/02_bagging_and_random_forests.ipynb. Thirteen minutes.

The cell to protect is the noise-column sweep. Let them add the columns
themselves and watch the importance share climb - reading the number off a
slide does not land the same way as watching it happen.

If time allows, section 6 also tries `max_features=None` instead of the
default `"sqrt"`: every split sees all 22 columns. Ask them to predict the
noise share before running it. Most will say it falls to about zero, since
the real features are now always on offer. It falls only to 28%, and the
accuracy rises from 0.837 to 0.897. Handout section 7 explains both.
:::

# Boosting: one tree after another

- Bagging and random forests grow trees **independently**, then let them
  vote
- **Gradient boosting** grows trees **in sequence**: each is trained to
  correct what the trees before it still get wrong
- Each tree is shallow and outputs a **number**, a correction, not a class:
  a **regression tree**
- The learning rate $\alpha$ decides **how much** of each correction is added

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

A predictable slip, worth naming now: "the final task is classification, so
each boosted tree votes default or repaid, like a tree in a forest". It is
reasonable - every ensemble so far was made of trees that vote. But a boosted
tree never outputs a class. It outputs a correction that is added to a
running score, and only the final score is turned into a class.

Name $\alpha$ carefully and tie it to the course's shared notation - this is
the same symbol as lesson 3's gradient descent learning rate, doing an
analogous job: how large a step each correction takes.

Handout section 8.

History: the first boosting algorithm to work in practice was AdaBoost, by Yoav
Freund and Robert Schapire (1995-1997), which reweighted the examples the trees
so far got wrong; it won them the Gödel Prize in 2003. Jerome Friedman's 2001
paper (in the further reading) recast the idea as gradient descent on a loss,
which is the version this lesson teaches and scikit-learn implements.
:::

# Building the ensemble, one tree at a time, for squared error

$$F_0(x) = \bar{y}\text{: start from the mean} \\ F_t(x) = F_{t-1}(x) + \alpha\, h_t(x)\text{: add a scaled tree}$$

::: notes
$F_0$ is the flat starting guess: the best single constant for the loss
being used. For squared error, as on this slide and the next three, that is
the mean of the target; for classification it will be the log-odds of the
base rate. Each subsequent
$F_t$ is the previous ensemble plus a scaled correction from the newest
tree, $h_t$. Read it left to right: start flat, then keep adding small,
scaled corrections.

Do not derive anything further here - the next slide states what $h_t$ is
actually fit to. Handout section 8.
:::

# What each tree is fit to

- For squared error, the next tree $h_t$ is trained on the **residual**
  $y - F_{t-1}(x)$: what the trees so far have not explained
- In general, $h_t$ is trained on the **negative gradient** of the loss
- With $L = \tfrac{1}{2}(y - F)^2$, $-\partial L/\partial F = y - F$: the
  same thing as the residual
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

- Now $y$ is 0 or 1. The running score $F$ becomes a probability through
  lesson 4's logistic function, $p = \sigma(F)$
- The loss is lesson 4's **log-loss**; its negative gradient is $y - p$, the
  **pseudo-residual**, lesson 4's error term: each tree is trained on it
- Start: $p = 464/1{,}200 = 0.387$ for all (the share who defaulted), so the
  first tree fits +0.613 to defaulters and −0.387 to the rest

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

The sigma here is lesson 4's logistic function, 1 / (1 + e^-F). If anyone has
read handout section 5.2, say that its sigma squared, the variance of one
tree's verdict, is a different quantity that happens to share the letter.

Give them the first step in numbers, because it makes y - p concrete. The
ensemble starts at the base rate's log-odds: 464 of the 1,200 defaulted, so
F0 = log(464 / 736) = -0.461, which the logistic function turns back into
0.387 for everyone. The first tree is fit to y - p: +0.613 for every applicant
who defaulted, -0.387 for every one who repaid.

A predictable slip: "so the update is p plus alpha times (y - p)". It is
reasonable, because y - p is a difference of probabilities. But the trees
correct the score F, not p: F moves by alpha times the tree's output, and
only then does the logistic function turn the new F into a new p. That is
what keeps p between 0 and 1 however large the corrections add up.

One detail, if someone has read scikit-learn's source. The tree's
splits are chosen on y - p exactly as described. The number written in each
leaf is then recomputed: the sum of y - p over the leaf divided by the sum
of p(1 - p), one Newton step on the log-loss rather than the plain mean. The
splits come from the residuals; the size of each step takes the curvature of
the loss into account. Handout section 9 has it in one sentence; nothing in
the lesson depends on it.

Handout section 9.
:::

# On the loan data

- `GradientBoostingClassifier`, defaults: **100 trees** of depth 3,
  learning rate $\alpha = 0.1$
- Reaches **0.902 ± 0.020** cross-validated accuracy, on the same five folds
- Ahead of the depth-8 tree (0.882), level with the nine-leaf tree (0.901)

::: notes
Give the number and let it sit against everything shown so far this
lesson: 0.902, from a method run at scikit-learn's defaults, no tuning at
all. Ask the room to recall the single trees' numbers: 0.882 at depth 8, which
boosting has passed without being told anything about depth, and 0.901 for
the nine-leaf tree of slide 25, which it only matches - on this rectangle
data a well-budgeted tree is hard to beat on the score alone.

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
Unlike bagging, boosting keeps reducing its training loss for as long as it
runs, because every new tree is built specifically to reduce what remains  - 
including the 7% of labels that are noise by construction. There is nothing
in the algorithm itself that says "stop". Be exact about which quantity: the
log-loss on the training data never goes up, while training accuracy only
follows it on the whole - fit to all 1,200 applicants with 800 trees, it
dips at a handful of individual steps, too small to show in this table
(handout section 10).

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
and easy to miss on a single run - which is why the number of trees has to
be chosen on held-out data, never left to run, even though the overfit is
gentler than a single unconstrained tree's. There are two ways to choose it.
This lesson cross-validated a list of tree counts and kept the best, 30.
The other is early stopping, which the deck has not shown, so say what it
is: set aside part of the training data, watch the score on it as trees are
added, and stop once it has not improved for a number of rounds. In
scikit-learn that is the `n_iter_no_change` argument of
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

A predictable slip: "only alpha times the number of trees matters, so half
the learning rate with twice the trees gives the same model". It is
reasonable - the notes above just said both knobs set the total correction.
But they are not interchangeable: each tree is fit to what the ensemble
before it still gets wrong, so many small steps follow a different path from
a few large ones. A large step overshoots on the noisy labels and the next
trees spend themselves undoing it; a small step lets later trees outvote an
early mistake. The trade-off is real, the equivalence is not.

Handout section 10.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 1. boosting from scratch, 47-51 | `DecisionTreeRegressor(max_depth=2)` on the residuals, 60 times → MSE 0.0119 at 20 trees |
| 2. classification, 52-53 | `GradientBoostingClassifier()` in `cross_val_score` → F0 = −0.461; 0.902 |
| 3. more trees, 54-55 | the same, `learning_rate=0.3`, `n_estimators` from 5 to 800 → peak 0.897 at 30 |
| 4. learning rate, 56 | the same with `n_estimators=120` and `learning_rate` from 0.02 to 1.0 → best 0.902 at 0.1 |
| 5. leaderboard, 58 | every model in one `cross_val_score` loop → random forest 0.911 |

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

| Model | Cross-validated accuracy | ± over the 5 folds |
|---|---|---|
| Majority baseline | 0.613 | |
| Logistic regression | 0.748 | 0.030 |
| Single tree, unconstrained | 0.852 | 0.021 |
| Single tree, `max_depth = 8` | 0.882 | 0.010 |
| Gradient boosting, 30 trees | 0.897 | 0.020 |
| Single tree, `max_leaf_nodes = 9` | 0.901 | 0.020 |
| Bagging, 100 trees | 0.904 | 0.020 |
| **Random forest, 100 trees** | **0.911** | 0.018 |
| *noise ceiling* | *≈ 0.93* | |

::: notes
Everything from this lesson, cross-validated the same way: 5 folds of the
1,200 loans, the same folds for every model. The third column is the standard
deviation of the five fold scores, the spread lesson 5 asks to see beside
every mean: the top four rows lie within one spread of each other. Read down
the column and let the pattern speak before naming it: the ensembles - bagging,
the random forest, gradient boosting - land within about a point and a half of
each other, the random forest on top. And one single tree sits among them: the
nine-leaf tree of slide 25, at 0.901, above gradient boosting and a point
behind bagging. On this dataset that is no accident - the rule is made of
axis-aligned rectangles, the shape a few well-placed thresholds draw.

The ceiling itself: 1 minus the 7% of labels deliberately flipped. No
method reaches it, and none should - a model that classified the flipped
labels "correctly" would be modelling the noise, not the rule.

One honest caveat, in lesson 5's terms. The depth of 8 and the 30 boosted
trees were picked because they scored best on these same five folds, so
those two rows are slightly optimistic; nested cross-validation would remove
that. Bagging and the random forest ran at their defaults, chose nothing,
and are not affected - and they are still on top. The nine leaves were chosen
for readability, not on these folds.

Handout section 11.
:::

# Three findings

- **On this dataset** the three ensembles land within 1.5 points of each
  other, and a **nine-leaf tree** among them (0.901): one a person can read
- What the ensembles buy here is **steadiness**: over 30 splits, the nine-leaf
  tree's spread is 0.020, the forest's 0.0137
- The **unconstrained tree is the worst**, though the most flexible

::: notes
Read the three findings as a set - together they are this lesson's real
content, more than any single accuracy number.

The second finding is the honest surprise. The ensembles were not built to
score higher than a well-budgeted tree on data made of rectangles - here they
barely do. What they buy is steadiness: on the same 30 random splits as slide
39, the nine-leaf tree averages 0.897 with a standard deviation of 0.020, the
forest 0.908 with 0.0137 (notebook 2, section 3). Slide 27 showed where a
single tree's verdicts change.

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
Permutation importance, the check, was slide 45; the next slide comes back to it.

Handout section 11.
:::

# 54% of the importance on noise

- 20 random columns, 2 real ones: over half the importance lands on the
  random ones
- **The menu**: each split may use 4 of the 22 columns; 4,845 of the 7,315
  possible sets of 4 hold no real column: two splits in three (0.662)
- **Small groups**: with all 22 on offer, noise still takes **28%**: on a
  few applicants deep in a tree, a random column fits a little by chance
- The check: **permutation importance**

::: notes
This is the number to carry out of the room. Say it once more, slowly: with
twenty pure-noise columns against two real features, more than half of a
random forest's importance mass sat on columns wired to nothing.

Ask them to write it down - derivations fade, this kind of number tends
not to. Every lesson so far has left one of these - lesson 1's 77% accuracy
on coin-flip labels, lesson 3's coefficient of 365 where an unpenalised fit
wanted billions - and this is lesson 7's.

The second and third bullets are the mechanism, and there are two causes,
not one. Take them in order.

The menu. The square root of 22, rounded down, is 4, so each split chooses
among 4 random columns. The chance that all 4 come from the 20 noise columns
is 4,845 / 7,315 = 0.662: two splits in three offer no real feature. At those
splits the tree still has to pick something, and it picks the noise column
that happens to separate the classes best.

Small groups. The menu is not the whole story, and notebook 2 shows it: with
`max_features=None`, every split offered all 22 columns, the noise share
falls from 54.5% to 28%, not to zero. Deep in a tree a node holds a handful of
applicants, and among twenty columns of random numbers one will almost always
separate a handful of applicants a little better than income or debt can,
purely by chance. Every such split credits noise.

Most will predict the noise share falls to zero with the full menu, and the
reasoning is sound as far as it goes - a real feature is now always there to
be chosen near the root. It is the bottom of the trees that it misses. The
same run raised the accuracy from 0.837 to 0.897: with 20 useless columns out
of 22, the random menu costs more accuracy than decorrelating the trees buys
back. The default is a good default, not a law.

The last bullet sends them back to the check of slide 45: shuffling a
column on held-out data costs the two real features 0.09 and 0.18 of
accuracy, and each noise column almost nothing.

Handout section 7 has the full mechanism.
:::

# What to take away

- A tree splits greedily on Gini impurity; depth is the bias-variance dial
- Bootstrap resampling leaves **36.8%** of rows out-of-bag: free validation
- A vote cancels only what **differs** between trees; boosting must be
  **stopped** on held-out data
- **54% of a random forest's importance landed on 20 noise columns**,
  today's number

::: notes
Close on the same shape as lesson 6's summary: a handful of numbers that
carry the lesson's argument without needing the mathematics behind them
re-derived.

If they remember one habit from today, make it this: before trusting a
random forest's importance ranking, ask how many of the columns could
plausibly matter at all - many useless columns are what hand credit to
noise, and the accuracy score gives no warning - and then check the ranking
by shuffling on held-out data, as slide 45 did.

Next lesson turns from labelled data to unlabelled data. Every model so far
learned from examples whose answer was known: a price, a class, a default.
Lesson 8 asks what structure there is in data when nobody has given the
answer at all.

Handout summary and notation table.
:::

# Homework

- **Exercise 7**, discussed at the start of **Friday 27 November**
- `Exercises/07_trees_and_ensembles.md`

::: notes
Set it explicitly and say out loud when it comes back: the first ten
minutes of lesson 8, Friday 27 November.

Remind them of the standing rule from lesson 5: every reported number needs
a cross-validated spread, and any feature-importance claim needs the check
this lesson just demonstrated - a random forest's importance ranking is a
claim to be verified, not a fact to be quoted.

Next week: unsupervised learning, the first lesson in which the data comes
with no labels at all.
:::

---
title: "Lesson 2: Data Exploration and Preparation"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "2 October 2026"
---

# Before we start

Exercise 1 was set last week. We discuss it now.

- The wine-quality workflow, end to end
- What counts is **methodology**, not accuracy
- A show of hands on how you framed it

::: notes
Open with the exercise, briefly - two or three minutes, not a review session.
Nothing is collected, so this is the whole of the feedback: ask for a show of
hands, who treated quality as binary and who kept it as an integer regression
problem. Both are defensible if justified; that was the whole point of Task 1.

Say once, here and only here, why it is worth having done: at the exam one of
the ten exercises is drawn and they talk through their own notebook for it.
Then move to today's material.
:::

# What today builds on

- Lesson 1's rule: **nothing is learned before the split**
- Lesson 1's tool: `Pipeline`, so the rule is structural, not just discipline
- Today: apply both to data that is actually messy

::: notes
Recap in one sentence each, then land the framing for the whole lesson:
Pipeline stops being "the scaling trick" and becomes the central tool of the
lesson, because everything from here on - imputing, encoding, engineering -
is another thing that learns from data and must sit inside it.

Handout Section 1 makes this explicit.
:::

# Today

- Exploratory analysis
- Missing values
- Outliers
- Scaling and encoding
- Feature engineering and pipelines
- Leakage, in a form nothing warns you about

::: notes
Agenda slide. Flag the last item now as the one to remember in five years -
everything before it is the machinery that makes the leakage discussion
precise rather than hand-wavy.

Timing: roughly 35 minutes to the first notebook, then a break, then scaling
and encoding, then the leakage section closes the lesson before homework.
:::

# One dataset, all lesson long

2000 synthetic telecom customers. Will they **churn**?

- Numeric: tenure, monthly charges, age, support calls
- Categorical: contract type, region, zip code (493 levels)
- Missing values, outliers, and one column built to carry **no signal**

::: notes
Explain why synthetic: every real messy teaching dataset sits behind a
network call or a licence, and this lesson needs exact control over the
missingness mechanism and the leakage story to make both demonstrable rather
than merely plausible.

Say explicitly: because we generated it, we know the ground truth. zip_code
does not affect churn at all, by construction. Keep that fact in mind - it is
what makes notebook 3 a clean experiment rather than an anecdote.

Framing, as Lesson 1 asked: predicting churn to decide who a retention team
calls. A missed churner costs a lost customer; an unnecessary call costs a
few minutes. Keep the asymmetry in mind for Lesson 4.
:::

# Look before you touch anything

`.info()`, `.describe()`, class balance, first: always.

- 2000 rows, 8 columns, mixed types
- Churn rate: **19.4%** → baseline accuracy 80.6%
- `tenure_months` max: 999. Not a typo in the slide.

::: notes
Read the three numbers off the slide and stop. The plot on the next slide is
what makes the 999 mean something, and it needs the whole slide to be legible
from the back of the room.

Ask, before turning over: what will a histogram of tenure look like, if one
customer in the file has 999 months?
:::

# Two of these four have no shape left to read

![](numeric_distributions.png)

::: notes
Do not describe shapes the room cannot see. On the first two panels there is no
shape to read: tenure and monthly_charges are each a single spike at the left,
with the axis running out to 1000 and 3500 and nothing drawn out there.

That absence IS the slide. Ask them what stretched the axis, and let someone
say it: a few values so extreme that every other customer has been squashed
into one bar. This is the figure that makes the next forty minutes necessary.

Then point at age and num_support_calls, which are on their own honest ranges
and do show a distribution - age skewed with a spike at the youngest bound,
support calls a Poisson-looking count. That contrast is the argument for
cleaning before looking any further, and it is why the same plot reappears
after Section 4.
:::

# Where the gaps are

![](missingness_pattern.png)

::: notes
Two columns have gaps: age at 8.0% and num_support_calls at 4.8%. The count is
the easy half of the question; the next three slides are the hard half - why
are they missing?

Make the point that this figure took one line to produce and is the first thing
to run on any new dataset, before any modelling thought at all. A column that is
90% empty is a column you drop; a column that is 5% empty is a decision.

Warn them about what the picture cannot show: whether the gaps are related to
each other, or to the target. That is what the next slides are for, and it
cannot be read off a bar chart.
:::

# Three reasons a value is missing

![](missingness_mechanisms.png)

::: notes
Handout Section 3.1. Define each precisely, because the distinction is not
academic - it determines which fix is valid.

MCAR, missing completely at random: probability of missing depends on
nothing. A blank field for no
systematic reason.

MAR, missing at random: depends on something you DO observe. Our num_support_calls is MAR
against tenure - long-standing customers' early call history predates the
customer relationship management (CRM) system. Conditional on tenure, the gap carries no further information.

MNAR, missing not at random: depends on the value itself, even unobserved. A customer who leaves
because of a shockingly high bill and never finishes a survey. No column in
the table can detect this - only knowing how the data was collected can.
Say plainly: this is the dangerous one, and it does not appear in today's
dataset, which is itself worth noting - most real ones have some.
:::

# age: MCAR

Some customers just left it blank. Nothing systematic.

- Safe to impute with a fixed statistic
- The bias question is not "is this fair": it is "what does filling it cost"

::: notes
Bridge to the derivation. Ask: if the missingness is completely random, is mean
imputation "free"? Let them guess before the next slide - most will say yes,
and the reasoning is sound as far as it goes: the mean is unbiased, so filling
with the mean introduces no bias in that column.

The answer is no, and the reason is that a column is never used alone. Handout
Section 3.2 derives it: filling p of the entries with a constant shrinks the
variance to (1-p) times the original, and attenuates every correlation the
column has with anything else by the square root of (1-p).

Have the numbers ready. For age at p = 0.08 the factor is 0.96 - small enough
to ignore. At p = 0.4 it is 0.77, which erases nearly a quarter of a genuine
relationship. Tell them the practical rule that follows: mean imputation is
fine when the gaps are few, and quietly destructive when they are many.
:::

# What mean imputation actually costs

Every correlation involving the imputed column shrinks, even under MCAR.

![](correlation_attenuation.png)

::: notes
Read the curve rather than the formula. Filling gaps with a constant costs correlation, and the cost grows with the fraction filled.

Point at the two markers. For age at 8% the factor is 0.96 - negligible, and this is why nobody notices. At 40% it is 0.77, which erases nearly a quarter of a real relationship.

The practical rule to state: mean imputation is fine when gaps are few, and quietly destructive when they are many. Nothing warns you at which point it crossed over.

The algebra is Handout Section 3.2: the correlation is multiplied by the square root of one minus the missing fraction. Write it on the board if the room wants it - the figure is what they will remember.
:::

# num_support_calls: MAR

Missing more often for long-tenure customers, not by chance.

- Treating it as MCAR reweights the sample the wrong way
- A **missingness indicator** column preserves the signal even after filling

::: notes
The fix that MCAR imputation does not need: add a binary column recording
whether the value was missing. It lets a model use the fact of missingness -
which, under MAR, is genuinely informative about tenure - even after the gap
itself is filled with a number. Handout Section 3.3.
:::

# What to do about a gap

- **Drop rows**: only if few, and only if MCAR
- **Drop the column**: if it is missing too often to help
- **Impute**: mean/median, most-frequent, `KNNImputer`
- **Add a missingness indicator**: keeps MAR signal alive

::: notes
Four options, and the choice is a question about WHY the data is missing rather than about how much is missing.

The one worth dwelling on is the fourth: adding a binary indicator keeps the MAR signal itself available to the model - the fact that long-tenure customers are more often missing a call count may be more informative than the count would have been.

Then the line at the bottom: every option except dropping the column learns a statistic, so all of them belong inside the training fold. That is the thread the whole lesson pulls on.
:::

# Choosing among the four

![](missing_data_decision.png)

::: notes
A decision path rather than a rule: how much is missing, does the
missingness look related to anything, and what does the column cost if it
goes.

Point out that three of the four branches end somewhere defensible, and
that the indicator branch is the one people forget. Adding a column that
records that a value was missing costs nothing and keeps the signal alive
when the fact of the gap is itself informative - which for anything
recorded by a human it usually is.

The branch worth warning about is dropping rows. It is the easiest to do
and the only one that can quietly change what the sample represents.
:::

# An outlier is far from the rest - not necessarily wrong

- `tenure_months` runs **-3 to 999**; ordinary customers sit at 0 to 72
- Three reasons: a **recording error**, a **rare but genuine** value, a
  **different population**
- The arithmetic cannot tell which: 3,344.7 may be a typo or a corporate account
- Two rules follow - standard deviations, then quartiles - and both only **flag**

::: notes
Pay off the 999 from slide 6: they have already seen it, and now it gets a name.

Put the three reasons to the room before reading them out - ask what could make
a customer's monthly charge 3,344.7 when everyone else is under 128. Someone
will say typo; ask whether they are sure, and let the silence do the work. That
is the point of the third bullet.

The last bullet is the promise the next six slides keep: rule 1 in standard
deviations, rule 2 in quartiles, then the slide where they disagree and the
slide where the robust one is the one that got it wrong. Nothing here deletes a
row - what to do once something is flagged is Section 4.2, and it has no
automatic answer.

Handout Section 4 opening.
:::

# Rule 1: distance in standard deviations, flag beyond k = 3

$$z_i = \frac{x_i - \bar{x}}{s}$$

::: notes
The z-score rule. Standardise the column, then flag anything more than k
standard deviations from the mean - we use k = 3 throughout.

Say what it assumes, because everything in the next three slides follows from
it: measuring in standard deviations only means "surprising" if the column is
a bell curve. It also computes its own ruler from the data being measured,
which is the crack the section prises open.
:::

# Rule 2: distance measured in quartiles

$$\left[\,Q_1 - 1.5\,\mathrm{IQR},\ \ Q_3 + 1.5\,\mathrm{IQR}\,\right]$$

::: notes
Tukey's fence, built on the interquartile range (IQR) - the spread between the
first and third quartiles, so the middle half of the data. Flag anything more
than 1.5 interquartile ranges past the quartiles.

It assumes no shape at all, and its ruler barely moves when a few points are
extreme. That sounds strictly better than rule 1. Two slides from now it will
be the rule making the mistakes.

Now give the probabilistic reading of each, because it is what makes them more
than recipes. Under normality |z| > 3 flags 0.27% of a column, both tails
together. And the 1.5 is not arbitrary: for a normal column the IQR is 1.349
sigma, so the fence lands at 2.698 sigma - just inside 3. Handout Section 4.1
derives it.

Then put the two side by side, counting both tails each time, because this is
where people quietly mislead themselves: 0.27% against 0.70%. The fences are
10% apart in position and a factor of 2.6 apart in what they flag.

Then make that concrete, because a percentage does not land: take 2,000 values
off a perfect bell curve, nothing wrong with any of them. The z-score rule still
nominates about 5 and Tukey about 14 - false alarms both, the tail being flagged
for being the tail. Say what that means for the twenty real billing errors in
this dataset: neither rule can tell the two kinds apart, which is why a flag is
a candidate and never a verdict.

The moral to state out loud: the two rules were built to sit in the same
neighbourhood, not to agree, and even on perfect data Tukey's is the readier
of the two to call something an outlier. The next slide draws it.
:::

# Same neighbourhood, not the same answer

![](outlier_fences.png)

::: notes
Handout Section 4.1 derives the 1.5 constant from the standard normal quantile
function.

Work the two panels in order, because the slide is an argument and not an
illustration. Left: the two fences are 0.30 sigma apart. Ask the room whether
that looks like a difference worth caring about - it does not. Right: the same
two rules flag 0.27% and 0.70% of a normal column, a factor of 2.6.

The reason is the steepness of the tail. Move a fence in by a third of a
standard deviation and you nearly triple the area beyond it. So resist saying
"the two rules agree here": they sit in the same neighbourhood, and out in the
tail a neighbourhood is a factor of 2.6.

Say this is a design choice with an assumption baked in, and the next slide is
what happens when the assumption fails.
:::

# The robust rule is the one that got it wrong

![](outlier_scatter.png)

::: notes
Put the counts up before the picture is read: 20 genuine billing errors in
monthly_charges. The z-score rule flags 20 - all of them, nothing else. Tukey
flags 32 - the same 20, plus 12 ordinary customers.

Ask the room to predict which rule was damaged by the contamination, then give
them the number: the billing errors inflate s from 17.2 to 171.3, dragging the
z-score fence from 115 out to 593. The rule was genuinely crippled. It got
every error anyway, because the smallest of them is 780 - still past the
ruined fence.

Meanwhile the twelve Tukey extras are real people on real tariffs: eight
paying 15 to 17, four paying 112 to 128, outside fences of 17.3 and 110.

The line to land, and it is the whole point of the slide: a detector's output
does not report the detector's health. The z-score rule was right by luck.
Push those errors down towards 200 and its fence sails straight past them,
exactly as the theory says. So the lesson is not "prefer IQR" - it is
"recompute s without the candidates and see what moved". Handout Section 4.2.

Point at the left panel's bottom band: those scattered red dots down in the
ordinary customers are the twelve.

tenure_months, right panel: only 4 flagged, all at 999, and here the two rules
agree exactly. Ask the room: where did the negative values go? Nobody will
know yet - next slide.
:::

# What neither rule can tell you

`tenure_months` of **-3** is impossible. Neither rule flags it.

- -3 is not *extreme* relative to a column spanning 0–72
- It is not *unusual*. It is *invalid*: a different question entirely

::: notes
This is the slide worth remembering over the two formulas. Statistical rules
find points that are unusual relative to the rest of a column; they know
nothing about domain validity. A negative tenure sits comfortably inside the
IQR fence because the column's own spread is wide enough to hide it.

The fix is a domain rule: 0 <= tenure_months <= some sane maximum. Say
plainly: a thorough pass runs both kinds of check, because they catch
different mistakes, and neither substitutes for the other. Handout Section
4.2.
:::

# Once something is flagged

Detection finds candidates. What to do next is a **domain** decision.

- **Cap** it at the fence: 3,344.7 becomes 110.0 (*Winsorising*)
- **Remove** the row
- **Correct** it, if the true value is recoverable
- **Leave** it: unusual is not the same as wrong

::: notes
A long-tenure, high-paying customer is not an error; a tenure of 999 months is.
Say plainly that a statistical rule cannot make this call - it only nominates
candidates, and the decision is a domain decision.

Walk the four options and note that each is a different claim about the world.
Capping - Winsorising, after Charles Winsor - replaces the value with the fence
itself: the billing error of 3,344.7 becomes 110.0, the row keeps every other
column it has, and the sample keeps its size. It says "the value is real but the
scale is misleading". Removing says
"this row is not from the population I am modelling". Correcting says "I know
what it should have been". Leaving it says "the model should handle this".

The one that gets chosen by default, without anyone deciding, is removal - and
it is the one that silently reweights the sample. Handout Section 4.2 closes on
exactly this. If a rule flags 3% of your rows and you drop them all, you have
changed the population your model is trained on.
:::

# Notebook 1, live

Exploration, missingness mechanisms, both outlier rules: from scratch.

::: notes
Work through notebook 1 now, 20 minutes. Let them drive; walk the room rather
than presenting.

The domain-rule cell is worth pausing on collectively: a two-line check for
negative tenure catches something no statistical rule would flag, because a
negative tenure is not extreme, it is impossible. That distinction - impossible
versus unusual - is the lesson of the whole section, and it lands better from
their own screen than from the projector.

Watch for the group that finishes early. Point them at the "try this" in the
notebook: change the missing fraction and watch the correlation attenuate as
the derivation predicts.
:::

# Break

::: notes
15 minutes. Back for scaling, encoding and the pipeline.

Use the break to check who is stuck in the notebook - the second half assumes
everyone has a working ColumnTransformer, and it is much cheaper to fix now
than during the leakage section.
:::

# Why scale?

- `tenure_months`: 0–72. `monthly_charges`: 15–128
- **Training** = choosing the model's numbers to make Lesson 1's average loss
  small
- **Gradient descent** does it by repeated small steps downhill on that loss
- One step size, along every axis, every iteration. Can it suit both columns?

::: notes
Reconnect to Lesson 1 before anything else, because the vocabulary is already
theirs: the empirical risk was the average loss over the rows we hold, and
choosing the f that makes it small was the whole definition of learning. What
was left open there is HOW you make it small. This is the how.

Say what a model's "numbers" are, once: one coefficient per feature, plus the
intercept. Training is a search over those numbers, and every candidate set of
them has a loss.

Then say what a single step is, because the slide leans on it: gradient descent
starts from some numbers, works out for each one whether nudging it up or down
lowers the loss, and moves them all at once - each by the same step size, times
how steeply the loss falls in that direction. One step size for the whole model,
not one per feature. That is the sentence the next slide draws.

Ask, before turning over: if two features live on very different scales, can a
single step size be right for both directions at once? Let the room reason
about it first.

Be honest about this dataset, because a sharp student will check: once the
outliers of Section 4 are removed, these two columns have comparable spreads -
the ratio of their standard deviations is about 0.8. They are not a dramatic
case, which is exactly why notebook 2 builds a toy pair with a 110:1 variance
ratio to make the effect unambiguous. The point is not that churn data is
pathological; it is that you cannot count on it not being.

The punchline, which handout Section 5.2 states: the safe step size is set by
the feature with the largest spread, so progress along the smallest-spread
direction is throttled by their ratio. Scaling is not cosmetic - it changes how
long training takes, and sometimes whether it converges at all. The derivation
is deliberately NOT here: it belongs with gradient descent itself, Lesson 3.
:::

# A ravine, not a bowl

![](condition_number_geometry.png)

::: notes
Do this one entirely on the picture; there is no algebra on this slide by choice.

Read the axes first, and do not skip this - a contour plot of a loss surface is
new to most of them and it is not a plot of the data. Neither axis is a column.
The horizontal axis is the coefficient the model gives feature 1, the vertical
axis is the coefficient it gives feature 2, so every point in the square is one
candidate model. The grey rings join the models that fit equally badly, the way
contour lines on a map join points of equal altitude. The blue star is the
model with the lowest loss - what training is looking for. The rust dots are
successive steps, and the line joining them is the search.

Left: the two features have different spreads, so the loss surface is a narrow
ravine - steep across, almost flat along. One step size has to serve both. Long
enough to advance along the flat direction and it overshoots the steep one and
bounces from wall to wall; short enough to be safe on the steep one and it
crawls along the flat one. Count the dots out loud: twenty-six steps and it is
still not at the star.

Right: after scaling, the ravine is a bowl and every step points at the minimum.

Give them the handle without the derivation: how stretched the ravine is - the
ratio of the largest feature variance to the smallest - is called the condition
number, and the number of steps you need grows with it.

If someone asks WHY the surface has that shape, that is the right question and
the honest answer is "Lesson 3, where we derive gradient descent" - its Section
4.4 puts the number on it: standardising the housing design matrix takes the
condition number from 285 to 3.4.
:::

# A 100:1 variance ratio, and what it costs

![](gd_convergence_scaled_vs_unscaled.png)

::: notes
Left panel: each feature set at its OWN safe rate - standardised still gets
there faster, because it tolerates a larger rate. Right panel is the sharper
point: give the RAW features the rate the standardised ones handle comfortably
and it does not converge slowly, it never converges at all.

Be precise about the failure if anyone asks, because "diverges" is the word
people reach for and it is not what this is. The loss does not climb away to
infinity - it swings between 0.69 and 8.29 and stays there, bouncing off one
wall of the ravine into the other. Run it for 2000 steps and it is still in the
same band. Worse than slow, not better: a slow run finishes.

This is a controlled toy (notebook 2), built to a 100:1 variance ratio and
landing at 110 once drawn, so the effect is unambiguous. On the real churn columns, once cleaned,
the two spreads are within about 20% of each other - so this toy is not
exaggerating a big effect in the data, it is manufacturing a clear one to show
a mechanism.
:::

# StandardScaler vs MinMaxScaler

Two standard choices, both fitted on training data only.

![](scaling_comparison.png)

::: notes
MinMaxScaler is the more outlier-sensitive of the two: one extreme billing
error stretches the whole 0-1 range, compressing every ordinary customer into
a sliver of it. StandardScaler is pulled too, through the mean and standard
deviation, but far less dramatically.

Both must be fitted on training data only - say this every time a fitted
transform appears, because it is the thread the whole lesson hangs from.
:::

# Every encoding makes a claim about the categories

A model does arithmetic, and `"month-to-month"` offers none.

| `contract_type` | rows | churn rate |
|---|---|---|
| `month-to-month` | 1,092 | 0.266 |
| `one-year` | 488 | 0.137 |
| `two-year` | 420 | 0.071 |

::: notes
Open by saying what the problem is, because nothing so far has needed it: a
model does arithmetic, and a column holding the words "month-to-month" offers
nothing to do arithmetic with. It has to become numbers, and there is more than
one way to do that.

Give them the word: the distinct values a categorical column can take are its
levels. contract_type has three, with 2,000 rows spread across them.

Then the sentence the whole section hangs on, and it is worth saying slowly:
every way of turning a category into numbers makes a claim about the
categories. The claim is the part to get right - the code is three lines either
way.

Point at the churn column before moving on. It does double duty: it is what
target encoding will use as the number, and it is the evidence that will convict
ordinal encoding two slides from now.

Handout Section 6.
:::

# One customer, three encodings

| Encoding | A one-year customer becomes | What that asserts |
|---|---|---|
| **One-hot** | `[0, 1, 0]` | simply different: no order, no distances |
| **Ordinal** | `1` | ordered, **and** equally spaced |
| **Target** | `0.137` | the level's own average outcome stands in for it |

::: notes
One customer, one contract, three numbers. Read the table a row at a time and
let the right-hand column do the work - it is what each encoding is asserting
about the world, not about the data type.

The middle row is where damage usually happens, and this column shows why the
mistake is reasonable: the order is genuine. A one-year contract really does sit
between the other two, so ordinal looks safe here in a way it would not for
region or zip code.

Ask the room which of the three claims is true for THIS column before turning
over. It is a domain question, not a technical one, and no cross-validation
score will answer it.

Handout Section 6.
:::

# Equal steps, unequal meaning

![](encoding_comparison.png)

::: notes
Left panel against right panel - that is the whole slide, and it is the answer
to the question left hanging.

Ordinal spaces the levels 0, 1, 2: equal steps by construction. The churn rates
those levels stand in for fall 0.266, 0.137, 0.071, so the first step is 0.129
and the second is 0.066 - the first is nearly twice the second. The encoding
asserts a regularity the column does not have, and a linear model has exactly
one coefficient with which to believe it.

Then the middle panel, which claims nothing at all and pays for that neutrality
in columns - the caption says k = 493 for zip_code, and that is the thread into
the next three slides.

The right panel is computed from the labels. Say that out loud and leave it
there: it is the first appearance of the leak the last third of the lesson is
about.

Handout Section 6.2.
:::

# One-hot claims nothing, and pays in columns

- A column with $k$ levels becomes $k$ binary columns, one 1 per row
- It asserts no order and no distance - which is why it is the default
- The price is width: `zip_code` would add **493** columns to 2,000 rows
- Target encoding compresses any cardinality into one column - computed from
  the labels, which is where leakage gets in

::: notes
This is the encoding to spend time on, because it is what scikit-learn's
OneHotEncoder does and what the pipeline later in the notebook uses.

The mechanics first: k levels, k binary columns, exactly one 1 in every row. No
category is closer to any other, which is precisely the neutrality that makes it
the safe default.

Then the cost, which is the honest trade: k columns. Three is nothing;
493 on 2,000 rows is the curse of dimensionality, two slides from here.

The last bullet is the hook, not a conclusion. Target encoding is the obvious
escape from the width problem and it is the more dangerous of the two
encodings in this entire lesson. Do not resolve it - the leakage section does.

One more thing this slide sets up: k columns plus an intercept is one column too
many, which the next slide proves.

Handout Section 6.1 and 6.2.
:::

# All k dummies plus an intercept cannot both be free

Most models carry a column of ones - the **intercept** - so they can fit a
baseline level.

$$\sum_{j=1}^{k} \text{dummy}_j = 1 = \text{intercept}$$

::: notes
Define the word before using it, because this is the first time in the course
they meet it: a model that fits an intercept is carrying one extra column made
entirely of ones, so that it has a baseline to work from. That is all they need
today - Lesson 3 is where it becomes a coefficient in an equation.

Now the redundancy, which needs no model at all. Ask them the survey question
from the handout: you record whether someone travels by car, by bus or by
bicycle, three yes/no columns, and everyone answered exactly once. How many of
those columns do you actually have to read? Two. The third is whatever is left,
which means the three columns carry two columns' worth of information.

The slide is that sentence in symbols, plus one step: the k dummies sum to 1 in
every row, and a column of ones is exactly what the intercept is. So k dummies
and an intercept are k+1 columns carrying k columns' worth of information, and
one of them is spare.

Stop there. What exactly breaks - the design matrix losing rank, the normal
equation no longer having a unique solution, infinitely many coefficient
vectors fitting equally well - is Lesson 3's machinery, and it is the right
place for it. Today's consequence is practical: pass drop="first".

Handout Section 6.1 carries the proof for whoever wants it now. The next slide
shows the redundancy as a number.
:::

# Rank deficient by exactly one

![](dummy_variable_trap.png)

::: notes
This is the redundancy counted by numpy rather than argued, which is the
version they will believe: matrix_rank on the k+1 columns comes back one short.

Point at the number: short by exactly one, not by an arbitrary amount. That is
the signature of a single dependence - the dummies summing to the intercept -
and not of general collinearity.

The fix is OneHotEncoder(drop="first"). The dropped category becomes the
reference level, absorbed into the intercept: no information is lost, because
that category is still recoverable as "all the other dummies are 0".

What a rank-deficient matrix does to a fit - infinitely many coefficient
vectors that predict identically, and a solver returning whichever one the
arithmetic happens to land on - is Lesson 3, once the normal equation exists to
say it with. Flag it as coming and move on.

Two caveats worth saying now. Tree-based models (Lesson 7) do not need the drop
at all. And regularised models tolerate the rank deficiency, because the
penalty picks one solution out of the many - which is a good early sketch of
what regularisation does, and Lesson 3 returns to it.
:::

# The curse of dimensionality, briefly

One-hot `zip_code`: **493** new columns, mostly zero.

- Distance-based methods (Lesson 6): everything looks equidistant
- More parameters to estimate, same sample size → more variance

::: notes
Handout Section 6.3. Deliberately quick - the point is to motivate why target
encoding looks attractive for high-cardinality columns, before the next section
shows why it needs care.

The number to dwell on: one-hot encoding zip_code produces 493 columns, almost
all zero, on 2000 rows. More columns than a quarter of the sample size, for one
field.

Two consequences to name. Distance-based methods (Lesson 6) degrade, because in
high dimensions everything becomes roughly equidistant from everything else.
And every column is a parameter to estimate from the same fixed amount of data,
which is the bias-variance trade-off from Lesson 1 arriving from a new
direction.
:::

# zip_code: 493 codes, 2,000 customers

![](onehot_width.png)

::: notes
Read the right panel out loud before making any argument with it, because it is
a histogram of the 493 codes and nobody reads that off a projector unaided: the
horizontal axis is how many customers share one zip code, the vertical axis is
how many zip codes that happens to. The tallest bar stands at 3 and 4 - about
a hundred codes each - and the whole thing stops at 12.

Then point at the zip_code row/column - or its absence, since it is not even
numeric yet - and say: notebook 1's correlation check already suggests this
column carries nothing. One-hot encoding it would add 493 mostly-empty
columns. The natural alternative - replace each code with the average churn
rate of its customers - is exactly the technique the leakage section is
about. Do not resolve the tension yet; let it hang until after the break-free
run into notebook 2's pipeline section.
:::

# Restating Lesson 1's argument

$f$ includes **every** learned parameter, not just "the model part".

$$\mathbb{E}_{T \sim \mathcal{D}^m}\left[\hat{R}_T(f)\right] = R(f)$$

::: notes
Say the condition out loud, because it is no longer written on the slide and it
is the whole content: this equality holds ONLY IF f is independent of T. It was
on the slide and made the formula too small to read from the back; it belongs in
your mouth, not in the image.

Handout Section 8.1. Lesson 1's unbiasedness argument never mentioned "the
model" specifically - it is about f, whatever function maps raw inputs to
predictions. In deployment, a fitted scaler or imputer travels WITH the
model, so f includes their learned parameters too.

The moment any of those parameters - a mean, a set of neighbours, a
per-category average - is estimated using rows that later appear in T, f is
no longer independent of T, and the equality breaks. This is the one-sentence
version of everything Section 9 is about to demonstrate concretely.
:::

# `ColumnTransformer` + `Pipeline`

![](pipeline_architecture.png)

::: notes
The structural version of the argument just stated. Different columns need
different treatment; ColumnTransformer routes each group through its own
sub-pipeline and concatenates the result; the outer Pipeline adds the
classifier.

One fit() call, on X_train, fits every imputer, the scaler, the encoder AND
the classifier - in that order, on training data only, by construction. Say
explicitly: this is not tidiness. It is what makes "nothing learned before
the split" true even when nobody is watching, which today's leakage section
is entirely about.
:::

# The code

```python
prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])
model = Pipeline([
    ("prep", prep),
    ("clf", clf),
])
model.fit(X_train, y_train)
```

::: notes
Say out loud what the five short names stand for, or someone spends the next
minute working it out instead of listening: num_pipe and cat_pipe are the
impute-then-transform pairs from the diagram two slides back, num_cols and
cat_cols are plain lists of column names, and clf is the classifier. They are
abbreviated here only to fit the slide - handout Section 8.2 prints the same
block with full names, every import, and both column lists, and it runs as
printed.

Point out there is exactly ONE call to fit, at the very bottom, on X_train
alone - everything above it declares structure, not computation.

This is the code notebook 2 runs, nearly verbatim. Say that reading it should
now feel unremarkable - that is the goal, structure that makes the right
thing the easy thing.
:::

# The model, on churn

- Baseline: **0.806**
- Model (numeric + the low-cardinality categoricals, no zip): **0.820** accuracy, **0.751** AUC (area under the receiver operating characteristic curve)

Modest accuracy gain. Why?

::: notes
Let the room answer why the accuracy gain is modest. The dataset is 80/20 imbalanced, so accuracy is dominated by the majority class - exactly as Lesson 1 warned, and the first time they meet it on data they prepared themselves.

The matrix shows where the errors sit: read the bottom-left cell, the churners the model missed. The area under the receiver operating characteristic curve - AUC, which Lesson 4 builds properly - at 0.751 is unaffected by the imbalance and says there is real signal that accuracy is hiding.

Had we reported only accuracy, this model would look barely better than the baseline and someone would reasonably conclude the features were useless.
:::

# Where the errors fall

![](churn_confusion_matrix.png)

::: notes
Say what the picture is before reading anything off it: a confusion matrix is a
table counting predictions against truth, one cell per combination - predicted
stay and did stay, predicted stay and churned, and so on. Lesson 4 gives it its
proper treatment; today it is a way of seeing where an accuracy number comes
from.

The accuracy gain was fourteen thousandths. This is where those errors
actually sit, and it explains why the gain is small.

Read the bottom row: almost every customer who churned was predicted to
stay. The model has learned to predict the majority class slightly more
cleverly than the baseline does, and on the class the business would pay
to identify it is close to useless.

Then connect it back to lesson 1 and its imbalance slide. Same shape,
different dataset, and it took a confusion matrix to see it in both
cases. Accuracy on its own would have reported this as progress.
:::

# Feature engineering

Ratios, interactions, binning: a linear model cannot invent these itself.

`total_paid` = charge × months stayed. AUC: 0.7514 → 0.7548, under four
thousandths. Clean `tenure_months` first and the same feature scores
**0.7439** - the gain was living in the 999s.

::: notes
The formula was on this slide as a rendered equation and came out at 23pt, the
smallest in the course: the column names are long enough that no amount of
trimming saves it. It is one sentence to say - total_paid is the monthly charge
times the months stayed, with tenure clipped at zero first - and the three AUCs
are what does the work in the room.

Do not let anyone in the room call three thousandths a win, and do not call it
one yourself. Quote the four decimals: at three, 0.751 against 0.755 sounds
like a difference of 0.004, and the difference is 0.0034. It is a legitimate
result, not a failed experiment - the hypothesis that what a customer has been
worth so far predicts whether they leave was reasonable, it was tested, and the
data declined it. Feature engineering is a hypothesis about the domain, not a
guarantee. Handout Section 7.

Then the second number, which is the one to spend time on. Rebuild the same
column after Section 4's domain rule - tenure clipped to 0-120, not just at
zero - and +0.0034 becomes -0.0075. The gain was never about what the customer
paid: 999 months times a billing error of 3,344.7 is a total_paid no real
customer could have, and those rows line up with the target well enough to move
the score. Multiplying two contaminated columns concentrates the contamination.
Say the moral out loud: the order of the steps is part of the method.

Then plant the flag for Lesson 5: a difference this small is inside the range
a different train/test split would produce on its own, so a single split
cannot settle it either way. Reporting it as an improvement is the first step
onto exactly the path Lesson 5 exists to close off.

Explain the max rather than skating over it, because someone will ask.
tenure_months still holds the -3 of Section 4 at this point, and multiplying by
it unclipped hands the model a customer who has paid minus two hundred euros.
That is Section 4 turning up in Section 7, and it is what feature engineering on
real data actually looks like - one line of algebra, three lines of defending it
against the column you have.

If someone objects to the obvious alternative, monthly_charges divided by
tenure_months, they are right to: monthly_charges is already a rate, so
dividing it by months again gives euros per month per month. Check the units
before trusting a construction.

The point to land: any engineered feature that involves a statistic LEARNED
from data - bin edges from quantiles, a scaled interaction - is subject to
the same rule as scaling. A pure arithmetic combination (this product) needs no
fold-awareness; a learned one does.
:::

# Notebook 2, live

Four numbers to come back with:

- Variance ratio **110:1** - learning rate **2.0** scaled, **0.1** raw
- That rate on raw features: loss swings **0.69 to 8.29**, for ever
- k dummies + intercept: **4 columns, rank 3**
- `Pipeline`, then `total_paid`: **0.7514 → 0.7548 → 0.7439**

::: notes
20 minutes. Let them work; circulate.

Say what the eight sections are for before they start, because the notebook is
long and the middle of it looks like plumbing: sections 3 and 4 are why we
scale, 5 and 6 are why we encode and what zip_code costs, 7 is the pipeline
that makes the rule true by construction, and 8 is the feature that does not
work.

The gradient descent toy in section 3 is worth running interactively - change
the variance ratio live and watch the iteration count move, per the handout's
"try this". It makes the condition number argument concrete in a way the
derivation alone does not. The number to make them say out loud is the second
one on the slide: the raw features do not diverge to infinity, they oscillate
between 0.69 and 8.29 and never settle. That is a worse failure than slowness,
because nothing errors.

Section 5 is the one that lands without discussion: matrix_rank returns 3 for
four columns, and short by exactly one is the signature of a single dependence.
They believe numpy where they might not believe the algebra.

If the room is moving fast, section 7 is where to slow down: ColumnTransformer
is the piece they will reuse in every exercise and in the project, and getting
the column selectors right is fiddlier than it looks.

And leave time for section 8, which is the one they will get wrong on their own
data: the engineered feature gains 0.0034, then loses 0.0075 once tenure is
cleaned first. Ask them why before telling them.
:::

# Same rule, broken three ways

![](invisible_leaks.png)

::: notes
Set up the closing section. Lesson 1's leak - selecting features on the full
dataset - is visible once you know to look for it: an obviously extra step.
Today's two are not. Imputing a missing value and encoding a category both
read as ordinary data cleaning, not as "training a model." They ARE training
a model, in the sense Section 8.1 makes precise: any fitted parameter is part
of f, and f must be independent of the test set.
:::

# Leak 1: impute before splitting

`KNNImputer` fills a gap using **other rows' values**.

Fit it on train + test, and some of those rows are in the test set.

::: notes
Walk the mechanism before the number. KNNImputer fills a gap using other rows'
values: it finds the nearest neighbours in feature space and averages them.
"Nearest in the whole dataset" includes test rows, if it was fitted before the
split existed.

So a TRAINING row's missing age can be filled with a value read off a TEST
row - the test set writing into the training data, one gap at a time. Note
the imputer never touches a label; it copies feature values. That is enough:
those values are what the model then fits on. The imputer did nothing wrong;
it was simply shown data it should not have seen.

Notebook 3 measures the gap. Connect it back to Lesson 1: this is the same
independence argument, applied to a step nobody thinks of as learning. The
imputer learns; therefore it belongs inside the training fold.
:::

# The smoking gun

**94 of 128** training rows with missing age had a **test-set row** among
the five donors used to fill it.

Nothing raised an error.

::: notes
94 of 128. Not an edge case - nearly three quarters of the imputed training rows borrowed a value from at least one row the model would later be scored on.

Stress that nothing in the code looks wrong. KNNImputer did exactly what it says: found the nearest neighbours in the data it was given. The error was in what it was given, and it happened one line earlier.

This is the number to put on the board if only one number from today survives.
:::

# Leakage, counted

![](smoking_gun.png)

::: notes
Ninety-four of a hundred and twenty-eight, drawn so the proportion lands.

Say what was actually done: the imputer was fitted before the split, so
when it filled a missing age it was allowed to look at test rows to decide
what to fill it with. Nearly three quarters of the affected training rows had a
test row among the five donors it consulted.

The sentence to leave hanging is the one from the slide before: nothing
raised an error. The notebook ran, the score improved, and the improvement
was the test set leaking into the training data one imputed value at a
time. This is the argument for the pipeline, made in numbers.
:::

# Leak 2: encode before splitting

Replace `zip_code` with the **average churn rate of its own customers**,
computed before anybody has split anything.

::: notes
zip_code has no real relationship with churn - the data was generated that way
deliberately, so any apparent signal is an artefact we can measure exactly.

Target encoding replaces each zip code with the average churn rate of its own
customers. Computed before the split, that average includes the test rows'
labels. The feature now contains the answer, in diluted form.

This is the most dangerous leak of the three, because the resulting column
looks entirely reasonable in a dataframe: a number between 0 and 1, sensibly
distributed, with no trace of where it came from. Nothing about it says "this
was computed from the labels".
:::

# A column with no real signal, encoded three ways

![](target_encoding_leak.png)

::: notes
Baseline without zip: AUC 0.751. Honest encoding (sklearn's TargetEncoder,
cross-fitted inside the pipeline): 0.751 - indistinguishable, correctly,
since zip carries nothing. Leaky encoding, computed before the split: 0.891.

Let that number sit. It looks like a genuinely better model. It is not one -
the improvement is manufactured entirely by the mechanism on the next slide.
This is Lesson 1's 77%-on-coin-flip-labels story again, produced by two
unremarkable lines of preprocessing instead of an obviously wrong one.
:::

# Your own disagreement, divided by your group size

$$\bar y_c - \bar y_c^{(-i)} = \frac{y_i - \bar y_c^{(-i)}}{n_c}$$

::: notes
Read the equation out as a sentence: the "leave-in" encoding differs from the
honest leave-one-out encoding by the row's own disagreement with its group,
divided by how many people are in it.

Handout Section 9.2 has the algebra. The intuition: for a zip code with a
single customer, the target encoding IS that customer's label, straight
through. With two customers it is the average of two labels. Only as the group
grows does the encoding become a genuine statistic rather than a copy of the
answer.

So the leak is worst exactly where the data is thinnest - which is also where
high-cardinality columns spend most of their mass. Most zip codes here have a
handful of customers.

Give them the diagnostic to remember: if a feature's usefulness comes mostly
from its rarest levels, be suspicious. That is the shape of a leak, not of a
signal.
:::

# The leak shrinks as 1/n_c

![](leak_shrinks_with_group_size.png)

::: notes
Notebook 3 shows the n_c=1 case literally: zip code Z472, one customer,
churn label 0, encoded value 0.000. Not correlated with the label. IS the
label, relabelled as an input.

Connect to Section 6.3: high cardinality means small groups means a bigger
leak by exactly this formula - the curse of dimensionality and the leakage
risk are the same underlying fact.
:::

# The fix

`sklearn.preprocessing.TargetEncoder`: cross-fitted inside the pipeline.

Each row's encoded value comes from **other** folds, never its own label.

::: notes
Smoothing pulls each group's mean towards the global mean, weighted by group
size, so a one-customer zip code barely moves from the overall rate. It
reduces the leak but does not remove it.

Say clearly what actually removes it: fitting the encoder inside the training
fold, exactly as with every other learned step. Smoothing is a refinement on
top of that, not a substitute for it.

This is the third time today the same rule has appeared - imputation, scaling,
encoding - and that repetition is the point of the lesson. Lesson 1 stated the
rule for one case; today it turns out to be the same rule every time.
:::

# The rule, restated

Everything that **learns from data** belongs inside the fold it is fitted on.

A mean. A median. A set of neighbours. A per-category average. A set of bin
edges.

::: notes
One test, not a growing list of special cases: does fitting this step compute
anything from the rows it is given? If yes, it goes inside the pipeline.
Handout Section 9.3.

Say explicitly: Lesson 5 returns to leakage a third time, for hyperparameter
tuning, where the same test applies once more.
:::

# Notebook 3, live

Trace the imputation leak. Watch the encoding leak manufacture 0.89 from noise.

::: notes
22 minutes. This notebook is the payoff of the lesson: the two leaks of
today, each measured against an honest pipeline on the same data. The third
way of breaking the rule, on the earlier diagram, was Lesson 1's.

The numbers matter less than the pattern - both leaks inflate the score,
neither raises an error, and both are things a competent person does by
accident. Ask them to predict the direction and rough size of each gap before
running the cell.

If time is short, the target encoding case is the one to keep: it is the
subtlest and the one most likely to appear in their own project.
:::

# What we did today

- Explored before transforming: types, gaps, shape, correlations
- Diagnosed *why* values are missing, not just *how many*
- Two outlier rules, and what neither can tell you
- Scaling, encoding, engineering: all inside `Pipeline`
- Two leaks that look nothing like leaking

::: notes
Draw the thread together explicitly. Lesson 1 said nothing is learned before
the split, using feature selection as the example. Today every preprocessing
step turned out to be a learning step: an imputer learns a mean or a set of
neighbours, a scaler learns a mean and a standard deviation, an encoder learns
a mapping from categories to numbers.

So the rule did not get more complicated - it got more general. And the
pipeline is not tidiness, it is the mechanism that enforces the rule
structurally, which is why we now build one for everything.

Preview Lesson 3 in one sentence: with the data prepared honestly, we can
finally fit something and look at what the fitting actually does.
:::

# Homework: we discuss it on Friday 9 October

`Exercises/02_data_exploration_and_preparation.md`

Build the full preparation pipeline yourself, and **find a leak on purpose**
before fixing it.

::: notes
Set it explicitly. The exercise reuses churn_data.py with a different seed,
so results will not match today's numbers exactly - that is intentional,
it stops copy-pasted answers from working.

Remind them: as in Exercise 1, there are no marks for accuracy. Marks are for
methodological correctness, and specifically here for correctly identifying
which preprocessing steps needed to be inside the pipeline and demonstrating,
with a number, what leaving one out would have cost.

Also point them at the quiz - the reasoning-tagged questions are the closest
thing to the exam they will see before the sample papers.
:::

# Before next week

- Work the three notebooks in order
- Read the handout: the derivations are examinable
- Take the quiz
- **Do the exercise**

Next: regression, the first model we derive completely.

::: notes
Set the exercise explicitly and say when it is discussed - the first ten minutes
of Lesson 3, Friday 9 October.

Point out that it uses the same churn dataset, so the exploration they did
today carries over, and that the marks are again on methodology: a pipeline
that is correct but modest beats a better score obtained by preparing the full
dataset before splitting.

Remind them the Lesson 1 exercise is due today if anyone has not handed it in,
and that the project topic must be confirmed by Lesson 4 - which is two weeks
away, so now is the time to be reading dataset descriptions.
:::

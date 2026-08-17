---
title: "Lesson 2 — Data: Exploration and Preparation"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "2 October 2026"
---

# Before we start

Exercise 1 was due at the start of today's lesson.

- The wine-quality workflow, end to end
- Marked on **methodology**, not on accuracy
- We will look at a few submissions' framing choices in a moment

::: notes
Open with the exercise, briefly - two or three minutes, not a review session.
Ask for a show of hands: who treated quality as binary, who kept it as an
integer regression problem? Both are defensible if justified; that was the
whole point of Task 1.

Do not dwell. The purpose of raising it is continuity, not grading feedback -
say that detailed feedback comes separately, and move to today's material.
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

# One dataset, all lesson

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

`.info()`, `.describe()`, class balance, first — always.

- 2000 rows, 8 columns, mixed types
- Churn rate: **19.4%** → baseline accuracy 80.6%
- `tenure_months` max: 999. Not a typo in the slide.

::: notes
Get a laugh at "999" and use it: nobody wrote a bug, somebody's system
recorded an impossible value and nothing caught it before it reached this
table. That is the entire subject of Section 4, arriving early.

State the baseline the way Lesson 1 insisted: 80.6% is what "always predict
no churn" scores. Every later number is measured against that, not against
zero. Handout Section 2.
:::

# Where the gaps are

![](missingness_pattern.png)

::: notes
Two columns have gaps: age (8.0%) and num_support_calls (4.8%). The count is
the easy half of the question; the next three slides are the hard half -
why are they missing?
:::

# Three reasons a value is missing

![](missingness_mechanisms.png)

::: notes
Handout Section 3.1. Define each precisely, because the distinction is not
academic - it determines which fix is valid.

MCAR: probability of missing depends on nothing. A blank field for no
systematic reason.

MAR: depends on something you DO observe. Our num_support_calls is MAR
against tenure - long-standing customers' early call history predates the
CRM system. Conditional on tenure, the gap carries no further information.

MNAR: depends on the value itself, even unobserved. A customer who leaves
because of a shockingly high bill and never finishes a survey. No column in
the table can detect this - only knowing how the data was collected can.
Say plainly: this is the dangerous one, and it does not appear in today's
dataset, which is itself worth noting - most real ones have some.
:::

# age: MCAR

Some customers just left it blank. Nothing systematic.

- Safe to impute with a fixed statistic
- The bias question is not "is this fair" — it is "what does filling it cost"

::: notes
Bridge to the derivation. Ask: if the missingness is completely random, is
mean imputation "free"? Let them guess before the next slide - most will say
yes.
:::

# What mean imputation actually costs

Every correlation involving the imputed column shrinks — even under MCAR.

$$\rho(X', Y) = \sqrt{1-p}\ \ \rho(X, Y)$$

::: notes
Handout Section 3.2 derives this in full: replacing the missing p-fraction of
X with its own mean shrinks Var(X) by (1-p) and Cov(X,Y) by (1-p) too, and the
asymmetry between how variance and covariance enter the correlation ratio is
where the square root comes from.

Give the number: p=0.08 for our age column, attenuation factor √0.92 ≈ 0.96 -
small. At p=0.4, it is √0.6 ≈ 0.77 - a genuine relationship substantially
erased by a constant. The mechanism is identical either way; only the size
changes.

This is not an argument against imputation - dropping rows or columns has its
own costs, covered in Section 3.3. It is an argument for knowing what you
paid.
:::

# num_support_calls: MAR

Missing more often for long-tenure customers — not by chance.

- Treating it as MCAR reweights the sample the wrong way
- A **missingness indicator** column preserves the signal even after filling

::: notes
The fix that MCAR imputation does not need: add a binary column recording
whether the value was missing. It lets a model use the fact of missingness -
which, under MAR, is genuinely informative about tenure - even after the gap
itself is filled with a number. Handout Section 3.3.
:::

# What to do about a gap

- **Drop rows** — only if few, and only if MCAR
- **Drop the column** — if it is missing too often to help
- **Impute** — mean/median, most-frequent, `KNNImputer`
- **Add a missingness indicator** — keeps MAR signal alive

::: notes
Handout Section 3.3. Four options, and the choice depends on the mechanism
just covered - dropping rows under MAR or MNAR reweights the sample in a
direction you may not intend, since the missing rows are not a random
subsample.

Whichever is chosen, flag the constraint that closes the loop back to
Pipeline: the statistic used - a mean, a set of neighbours - is learned from
data and must be learned from the training fold only. Section 9 is what
happens when that is forgotten.
:::

# Outliers, two rules

Two standard rules for flagging an unusual value:

$$z_i = \frac{x_i - \bar x}{s} \qquad\qquad [\,Q_1 - 1.5\,\mathrm{IQR},\ \ Q_3 + 1.5\,\mathrm{IQR}\,]$$

::: notes
Handout Section 4.1. z-score standardises and thresholds; Tukey's IQR rule
fences off anything more than 1.5 interquartile ranges past the quartiles.
The 1.5 is not arbitrary - next slide derives it.
:::

# Calibrated to agree — on a normal column

![](outlier_fences.png)

::: notes
Handout Section 4.1 derives the 1.5 constant from the standard normal
quantile function: it is chosen so the IQR fence and a 3-sigma z-score fence
flag comparable tails on a genuinely normal column. Say this is a design
choice with an assumption baked in, and the next slide is what happens when
the assumption fails.
:::

# The two rules disagree on real data

![](outlier_scatter.png)

::: notes
monthly_charges: z-score flags 20, IQR flags 32. The z-score rule's own mean
and standard deviation are dragged around by the outliers it is trying to
catch - a handful of billing errors inflate s directly, making the rule LESS
sensitive exactly when it should be more so. IQR barely moves under the same
contamination. That is why IQR is the safer default for skewed, error-prone
columns.

tenure_months: only 4 flagged on the right panel, all at 999. Ask the room:
where did the negative values go? Nobody will know yet - next slide.
:::

# What neither rule can tell you

`tenure_months` of **-3** is impossible. Neither rule flags it.

- -3 is not *extreme* relative to a column spanning 0–72
- It is not *unusual*. It is *invalid* — a different question entirely

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

- **Cap** it at the fence (Winsorise)
- **Remove** the row
- **Correct** it, if the true value is recoverable
- **Leave** it — unusual is not the same as wrong

::: notes
A long-tenure, high-paying customer is not an error; a tenure of 999 is. Say
plainly that a statistical rule cannot make this call - it only nominates
candidates. Handout Section 4.2 closes on exactly this distinction.
:::

# Notebook 1, live

Exploration, missingness mechanisms, both outlier rules — from scratch.

::: notes
Work through notebook 1 now. Let them drive; walk the room. The domain-rule
cell (negative tenure) is worth pausing on collectively - it is a two-line
check with an outsized lesson.

25 minutes, then break.
:::

# Break

::: notes
15 minutes. Back for scaling, encoding and the pipeline.
:::

# Why scale?

`tenure_months`: 0–70. `monthly_charges`: 15–150.

Gradient descent takes **one step size, along every axis, every iteration.**

::: notes
Set up the problem before the maths. Ask: if two features live on very
different scales, can a single learning rate be right for both directions at
once? Let the room reason about it before the derivation.
:::

# The Hessian is the feature covariance

$$H = \nabla^2 J(w) = \frac{1}{m}X^\top X$$

For uncorrelated, centred features, $H$ is diagonal:

$$H = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_n^2)$$

::: notes
Handout Section 5.2, done exactly for linear regression's quadratic cost and
carried over qualitatively to logistic regression, whose Hessian has the same
X-transpose-X structure weighted by p(1-p).

The safe step size along axis j is roughly 2/σⱼ². One global learning rate is
bounded by the LARGEST variance feature, which makes progress along the
smallest-variance direction painfully slow. That ratio is the condition
number κ = σ²max / σ²min, and standardising drives it towards 1.
:::

# A 100:1 variance ratio, and what it costs

![](gd_convergence_scaled_vs_unscaled.png)

::: notes
Left panel: each feature set at its OWN safe rate - standardised still gets
there faster, because it tolerates a larger rate. Right panel is the sharper
point: give the RAW features the rate the standardised ones handle
comfortably, and it does not converge slowly - it diverges. Same data, same
starting point, one rate stable and one exploding.

This is a controlled toy (notebook 2), built with exactly a 100:1 variance
ratio so the effect is unambiguous - real churn features show the same
direction of effect more mildly, since monthly_charges has roughly 3-4 times
the spread of tenure_months, not 100 times.
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

# The dummy variable trap

One-hot encode all $k$ categories AND fit an intercept:

$$\sum_{j=1}^{k} \text{dummy}_j = 1 = \text{intercept}$$

::: notes
The k dummy columns always sum to the intercept column, so they are not
independent of it. Handout Section 6.1 has the full proof; notebook 2
confirms it directly with matrix_rank. Next slide shows it as arithmetic.
:::

# Rank deficient by exactly one

![](dummy_variable_trap.png)

::: notes
Fix: OneHotEncoder(drop="first"). The dropped category becomes the reference,
absorbed into the intercept - no information lost, full rank restored.
Tree-based models (Lesson 7) do not need this, since they split on one dummy
at a time and do not care about collinearity.
:::

# Ordinal vs target encoding

::: columns
:::: column
**Ordinal**

- Categories → integers
- Only valid if a real **order** exists
- `low` < `medium` < `high`
::::
:::: column
**Target**

- Category → a statistic of $y$
- Compresses any cardinality to one column
- The dangerous one — next
::::
:::

::: notes
Ordinal encoding on an unordered category (North=0, South=1) tells a model
North and South are numerically closer than North and West - meaningless, and
actively misleading for anything that uses the number arithmetically.

Target encoding is why zip_code is tempting: 493 levels compressed to one
predictive-looking column. Say directly: the temptation is exactly what makes
it worth a whole section. Handout Section 6.2.
:::

# The curse of dimensionality, briefly

One-hot `zip_code`: **493** new columns, mostly zero.

- Distance-based methods (Lesson 6): everything looks equidistant
- More parameters to estimate, same sample size → more variance

::: notes
Handout Section 6.3. Quick, not a full treatment - the point is to motivate
why target encoding is attractive for high-cardinality columns before the
next section shows why it needs care.
:::

# zip_code: 493 levels

![](correlation_heatmap.png)

::: notes
Point at the zip_code row/column - or its absence, since it is not even
numeric yet - and say: notebook 1's correlation check already suggests this
column carries nothing. One-hot encoding it would add 493 mostly-empty
columns. The natural alternative - replace each code with the average churn
rate of its customers - is exactly the technique the leakage section is
about. Do not resolve the tension yet; let it hang until after the break-free
run into notebook 2's pipeline section.
:::

# Restating Lesson 1's argument

$f$ includes **every** learned parameter — not just "the model part".

$$\mathbb{E}_{T \sim \mathcal{D}^m}\left[\hat{R}_T(f)\right] = R(f) \quad \text{only if } f \text{ is independent of } T$$

::: notes
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
model = Pipeline([("prep", prep), ("clf", clf)])
model.fit(X_train, y_train)
```

::: notes
numeric_pipe and categorical_pipe are the impute-then-transform pairs from
the diagram; show the full definitions on screen in notebook 2 rather than
here. Point out there is exactly ONE call to fit, at the very bottom, on
X_train alone - everything above it declares structure, not computation.

This is the code notebook 2 runs, nearly verbatim. Say that reading it should
now feel unremarkable - that is the goal, structure that makes the right
thing the easy thing.
:::

# The model, on churn

- Baseline: **0.806**
- Model (numeric + low-card categorical, no zip): **0.820** accuracy, **0.752** AUC

Modest accuracy gain. Why?

::: notes
Let the room answer: 80/20 imbalance, so accuracy is dominated by the
majority class exactly as Lesson 1 warned. AUC, unaffected by the imbalance,
shows the model is finding real signal that accuracy alone would hide.
Lesson 4 makes both tools precise.
:::

# Feature engineering

Ratios, interactions, binning — a linear model cannot invent these itself.
AUC: 0.752 → 0.754, a modest but real gain.

$$\text{charge\_per\_tenure} = \frac{\text{monthly\_charges}}{\text{tenure\_months} + 1}$$

::: notes
A modest gain, and say plainly that this is itself a legitimate result, not a
failed experiment - feature engineering is a hypothesis about the domain, not
a guarantee. Handout Section 7.

The point to land: any engineered feature that involves a statistic LEARNED
from data - bin edges from quantiles, a scaled interaction - is subject to
the same rule as scaling. A pure arithmetic combination (this ratio) needs no
fold-awareness; a learned one does.
:::

# Notebook 2, live

Scaling from scratch, the dummy trap, `ColumnTransformer`, `Pipeline`.

::: notes
25 minutes. The gradient descent toy is worth running interactively - change
the variance ratio live if time allows, per the handout's "try this".
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

# Leak 1 — impute before splitting

`KNNImputer` fills a gap using **other rows' values**.

Fit it on train + test, and some of those rows are in the test set.

::: notes
Walk the mechanism before the number. KNNImputer finds nearest neighbours in
feature space; "nearest in the whole dataset" includes test rows if it was
fitted before the split existed. Notebook 3 does not just claim this - it
traces it, row by row.
:::

# The smoking gun

**98 of 128** training rows with missing age had a **test-set row** among
their five nearest neighbours.

Nothing raised an error.

::: notes
This is traceable specifically because we control the ground truth and can
look up which rows are in which split. In practice you rarely get this
diagnostic; the number here is instead a proof of mechanism, done once so you
believe it happens.

Downstream effect on this run: AUC 0.7548 (leaky) vs 0.7530 (honest) - small.
Land the point deliberately: a leak does not have to be dramatic to be wrong.
It is still an optimistic number reported to whoever reads it next.
:::

# Leak 2 — encode before splitting

Replace `zip_code` with the **average churn rate of its own customers**,
computed before anybody has split anything.

::: notes
zip_code has no real relationship with churn, by construction - the
correlation heatmap already told us. Watch what a leak manufactures out of
nothing.
:::

# A column with no real signal, encoded three ways

![](target_encoding_leak.png)

::: notes
Baseline without zip: AUC 0.751. Honest encoding (sklearn's TargetEncoder,
cross-fitted inside the pipeline): 0.752 - indistinguishable, correctly,
since zip carries nothing. Leaky encoding, computed before the split: 0.891.

Let that number sit. It looks like a genuinely better model. It is not one -
the improvement is manufactured entirely by the mechanism on the next slide.
This is Lesson 1's 77%-on-coin-flip-labels story again, produced by two
unremarkable lines of preprocessing instead of an obviously wrong one.
:::

# Why small groups make it worse

The "leave-in" encoding differs from the honest leave-one-out encoding by:

$$\bar y_c - \bar y_c^{(-i)} = \frac{y_i - \bar y_c^{(-i)}}{n_c}$$

::: notes
Handout Section 9.2, derived in full: the gap shrinks as 1/n_c. For n_c=1 the
formula degenerates completely - the encoded value literally equals the
label. Next slide shows the consequence directly.
:::

# The leak shrinks as 1/n_c

![](leak_shrinks_with_group_size.png)

::: notes
Notebook 3 shows the n_c=1 case literally: zip code Z378, one customer,
churned=1, encoded value 1.000. Not correlated with the label. IS the label,
relabelled as an input.

Connect to Section 6.3: high cardinality means small groups means a bigger
leak by exactly this formula - the curse of dimensionality and the leakage
risk are the same underlying fact.
:::

# The fix

`sklearn.preprocessing.TargetEncoder` — cross-fitted inside the pipeline.

Each row's encoded value comes from **other** folds, never its own label.

::: notes
The library implementation generalises leave-one-out to leave-one-fold-out
for efficiency, same idea. Used inside a Pipeline fitted on X_train alone, it
recovers the honest 0.752 - because that is the correct answer for a column
with nothing to contribute.
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
30 minutes - this is the centrepiece of the lesson, give it the time. Let
students find the singleton zip code themselves rather than just reading the
number.
:::

# What we did today

- Explored before transforming — types, gaps, shape, correlations
- Diagnosed *why* values are missing, not just *how many*
- Two outlier rules, and what neither can tell you
- Scaling, encoding, engineering — all inside `Pipeline`
- Two leaks that look nothing like leaking

::: notes
Recap slide. Keep it brisk - the room has done three notebooks and needs the
summary, not a re-lecture.
:::

# Homework — due Friday 9 October

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
- Read the handout — the derivations are examinable
- Take the quiz
- **Do the exercise**

Next: regression — the first model we derive completely.

::: notes
Close on time. Lesson 3 is where the course starts building models rather
than preparing to build them - say that the pipeline habit from today carries
forward unchanged, it is just the "model" box in the diagram that gets
interesting now.
:::

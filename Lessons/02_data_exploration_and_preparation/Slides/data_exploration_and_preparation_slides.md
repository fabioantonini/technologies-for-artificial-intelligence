---
title: "Lesson 2: Data Exploration and Preparation"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
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

![](numeric_distributions.png)

::: notes
This figure already existed in the notebook and is the reason the next forty minutes happen: monthly_charges is bimodal, tenure is nearly uniform, age is skewed and has a gap. Ask what each shape implies before moving on - a bimodal column usually means two populations mixed together.
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
CRM system. Conditional on tenure, the gap carries no further information.

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

# Outliers, two rules

Two standard rules for flagging an unusual value: the z-score, and Tukey's
fence built on the interquartile range (IQR):

$$z_i = \frac{x_i - \bar x}{s} \qquad\qquad [\,Q_1 - 1.5\,\mathrm{IQR},\ \ Q_3 + 1.5\,\mathrm{IQR}\,]$$

::: notes
Handout Section 4.1. The z-score standardises and thresholds; Tukey's rule
fences off anything more than 1.5 interquartile ranges past the quartiles.

Give the probabilistic reading of each, because it is what makes them more than
recipes. Under normality, |z| > 3 flags about 0.27% of a column. And the 1.5 is
not arbitrary: for a normal column the IQR is 1.349 sigma, so the fence lands at
2.698 sigma and flags about 0.35% per tail. Tukey chose the constant precisely
so the two rules agree on well-behaved data.

That agreement is the setup for the next section. Both rules are calibrated on
normality; the interesting question is what they do when the data is not
normal, and that is where they part company.
:::

# Calibrated to agree: on a normal column

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

- **Cap** it at the fence (Winsorise)
- **Remove** the row
- **Correct** it, if the true value is recoverable
- **Leave** it: unusual is not the same as wrong

::: notes
A long-tenure, high-paying customer is not an error; a tenure of 999 months is.
Say plainly that a statistical rule cannot make this call - it only nominates
candidates, and the decision is a domain decision.

Walk the four options and note that each is a different claim about the world.
Capping says "the value is real but the scale is misleading". Removing says
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
Work through notebook 1 now, 25 minutes. Let them drive; walk the room rather
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

`tenure_months`: 0–72. `monthly_charges`: 15–128.

Gradient descent takes **one step size, along every axis, every iteration.**

::: notes
Set up the problem before any mathematics. Ask: if two features live on very
different scales, can a single step size be right for both directions at once?
Let the room reason about it first.

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

Left: the two features have different spreads, so the cost surface is a narrow
ravine - steep across, almost flat along. One step size has to serve both. Long
enough to advance along the flat direction and it overshoots the steep one and
bounces from wall to wall; short enough to be safe on the steep one and it
crawls along the flat one.

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
point: give the RAW features the rate the standardised ones handle
comfortably, and it does not converge slowly - it diverges. Same data, same
starting point, one rate stable and one exploding.

This is a controlled toy (notebook 2), built with exactly a 100:1 variance
ratio so the effect is unambiguous. On the real churn columns, once cleaned,
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

# The dummy variable trap

One-hot encode all $k$ categories AND fit an intercept:

$$\sum_{j=1}^{k} \text{dummy}_j = 1 = \text{intercept}$$

::: notes
The k dummy columns always sum to the intercept column, so they are not
independent of it: the design matrix loses rank, and the normal equation has no
unique solution.

Stress that this is not a numerical inconvenience but an identifiability
problem. There are infinitely many coefficient vectors that fit equally well,
because you can add a constant to every dummy coefficient and subtract it from
the intercept without changing a single prediction.

Handout Section 6.1 has the proof; notebook 2 confirms it directly with
matrix_rank, which is the version they will believe. The next slide shows it
as arithmetic.
:::

# Rank deficient by exactly one

![](dummy_variable_trap.png)

::: notes
Point at the number: the rank is short by exactly one, not by an arbitrary
amount. That is the signature of a single linear dependence - the sum of the
dummies equalling the intercept - rather than of general collinearity.

The fix is OneHotEncoder(drop="first"). The dropped category becomes the
reference level, absorbed into the intercept: no information is lost, and full
rank is restored.

Two caveats worth saying. Tree-based models (Lesson 7) do not need this,
because they never invert a design matrix. And regularised models tolerate the
rank deficiency, because the penalty term picks one solution out of the
infinitely many - which is a good early illustration of what regularisation
actually does, and we return to it in Lesson 3.
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
- The dangerous one comes next
::::
:::

::: notes
Three encodings of one column, and each makes a different claim.

Ordinal claims the categories are equally spaced on a line - fine for small/medium/large, wrong for region. One-hot claims nothing at all, which is why it is the default, and costs k columns: 493 for zip code on 2000 rows. Target claims each category is summarised by its own churn rate, which is compact and, as the next section shows, computed from the labels.

The question to leave them with: which claim is true for THIS column? It is a domain question, not a technical one.
:::

# Three encodings, three claims

![](encoding_comparison.png)

::: notes
The same column under all three schemes, and each makes a different claim about
the world.

Ordinal claims the categories are equally spaced along a line - fine for
small/medium/large, wrong for region, and the model has no way to tell you it
was the wrong claim.

One-hot claims nothing at all, which is why it is the default, and pays for
that neutrality in columns: 493 of them for zip code, on 2000 rows.

Target claims each category is summarised by its own churn rate. Compact,
effective - and computed from the labels, which is where the next section
begins.

The question to leave with them: which claim is true for THIS column? That is a
domain question, not a technical one, and no cross-validation score will answer
it.
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

# zip_code: 493 levels

![](onehot_width.png)

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

$f$ includes **every** learned parameter, not just "the model part".

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
- Model (numeric + low-card categorical, no zip): **0.820** accuracy, **0.751** AUC (area under the receiver operating characteristic curve)

Modest accuracy gain. Why?

::: notes
Let the room answer why the accuracy gain is modest. The dataset is 80/20 imbalanced, so accuracy is dominated by the majority class - exactly as Lesson 1 warned, and the first time they meet it on data they prepared themselves.

The matrix shows where the errors sit: read the bottom-left cell, the churners the model missed. The area under the receiver operating characteristic curve - AUC, which Lesson 4 builds properly - at 0.751 is unaffected by the imbalance and says there is real signal that accuracy is hiding.

Had we reported only accuracy, this model would look barely better than the baseline and someone would reasonably conclude the features were useless.
:::

# Where the errors fall

![](churn_confusion_matrix.png)

::: notes
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
AUC: 0.751 → 0.754, a modest but real gain.

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
25 minutes. Let them work; circulate.

The gradient descent toy is worth running interactively - change the variance
ratio live and watch the iteration count move, per the handout's "try this".
It makes the condition number argument concrete in a way the derivation alone
does not.

If the room is moving fast, the ColumnTransformer section is where to slow
down: it is the piece they will reuse in every exercise and in the project,
and getting the column selectors right is fiddlier than it looks.
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

So a test row's missing value can be filled using its own neighbours - which
may include rows whose labels the model will later be scored on. The imputer
did nothing wrong; it was simply shown data it should not have seen.

Notebook 3 measures the gap. Connect it back to Lesson 1: this is the same
independence argument, applied to a step nobody thinks of as learning. The
imputer learns; therefore it belongs inside the training fold.
:::

# The smoking gun

**98 of 128** training rows with missing age had a **test-set row** among
their five nearest neighbours.

Nothing raised an error.

::: notes
98 of 128. Not an edge case - the substantial majority of imputed training rows borrowed a value from at least one row the model would later be scored on.

Stress that nothing in the code looks wrong. KNNImputer did exactly what it says: found the nearest neighbours in the data it was given. The error was in what it was given, and it happened one line earlier.

This is the number to put on the board if only one number from today survives.
:::

# Leakage, counted

![](smoking_gun.png)

::: notes
Ninety-eight of a hundred and twenty-eight, drawn so the proportion lands.

Say what was actually done: the imputer was fitted before the split, so
when it filled a missing age it was allowed to look at test rows to decide
what to fill it with. Three quarters of the affected training rows had a
test row among the five neighbours it consulted.

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

# Why small groups make it worse

The "leave-in" encoding differs from the honest leave-one-out encoding by:

$$\bar y_c - \bar y_c^{(-i)} = \frac{y_i - \bar y_c^{(-i)}}{n_c}$$

::: notes
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
Notebook 3 shows the n_c=1 case literally: zip code Z378, one customer,
churned=1, encoded value 1.000. Not correlated with the label. IS the label,
relabelled as an input.

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
20 minutes. This notebook is the payoff of the lesson: three leaks, each
measured against an honest pipeline on the same data.

The numbers matter less than the pattern - every leak inflates the score, none
of them raises an error, and all three are things a competent person does by
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

# Homework: due Friday 9 October

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
Set the exercise explicitly and give the deadline - Friday 9 October, at the
start of Lesson 3.

Point out that it uses the same churn dataset, so the exploration they did
today carries over, and that the marks are again on methodology: a pipeline
that is correct but modest beats a better score obtained by preparing the full
dataset before splitting.

Remind them the Lesson 1 exercise is due today if anyone has not handed it in,
and that the project topic must be confirmed by Lesson 4 - which is two weeks
away, so now is the time to be reading dataset descriptions.
:::

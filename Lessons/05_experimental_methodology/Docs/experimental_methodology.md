---
title: "Experimental Methodology"
subtitle: "Lesson 5 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "23 October 2026 · reading time about 85 minutes"
---

## Lesson plan

| Time | Minutes | Segment | Material |
|---|---|---|---|
| 0:00–0:10 | 10 | Exercise 4 returned; the promise we broke | Slides 2–4 |
| 0:10–0:30 | 20 | One split is a lottery | Slides 5–10 |
| 0:30–0:52 | 22 | Cross-validation, and when it needs care | Slides 11–18 |
| 0:52–1:14 | 22 | **Notebook 01** — the split lottery | Slide 19 |
| 1:14–1:26 | 12 | **Break** | Slide 20 |
| 1:26–1:50 | 24 | Bias, variance and the noise floor | Slides 21–27 |
| 1:50–2:04 | 14 | Learning curves and what they prescribe | Slides 28–29 |
| 2:04–2:24 | 20 | **Notebook 02** — measuring the decomposition | Slide 30 |
| 2:24–2:46 | 22 | Leakage that survives cross-validation | Slides 31–38 |
| 2:46–2:56 | 10 | The debt paid; seeds; reproducibility | Slides 39–43 |
| 2:56–3:00 | 4 | **Notebook 03**; summary; homework | Slides 44–46 |
| | **180** | **Total** | **46 slides, 3 notebooks** |

---

## 1. Why this lesson sits in the middle of the course

You can already build models. Lessons 3 and 4 gave you regression and
classification, and scikit-learn gives you a hundred more for one line each.
**That is precisely the problem this lesson exists to solve.**

A student who can program will produce a model that appears to work within an
afternoon. Producing a model that appears to work is not difficult and it is not
the job. The job is knowing whether it *does* work, and that turns out to be
considerably harder than fitting it — hard enough that published results in
several fields have not survived scrutiny.

Everything after today depends on this. Lessons 6 to 10 add more powerful and
more flexible methods, and every increase in flexibility increases the number of
ways a result can be wrong while looking right.

### 1.1 A promise we broke on purpose

Lesson 4 ended by choosing a decision threshold — and choosing it by sweeping
values on the **test set**, which lesson 1 forbade. We said at the time that
this was a demonstration of a method rather than a reportable result, and that
lesson 5 would fix it.

Section 8 fixes it. Keep the debt in mind while reading, because almost every
error in this lesson has the same shape: **a choice was made by looking at data
that was then used to report the result.**

---

## 2. A test score is a measurement

### 2.1 The picture first

You would not report the result of a single opinion poll of forty people to
three decimal places. You would say: this is an estimate, it has an error bar,
and the error bar is wide because forty people is not many.

**A test score is exactly the same kind of object.** It is computed on a finite
sample of held-out rows, and a different sample would have given a different
number. The rule from lesson 1 — hold data out, never look at it — makes the
estimate *unbiased*. It does nothing whatever to make it *precise*.

That distinction is the whole of section 2, and it is routinely missed, because
the phrase "test set accuracy" sounds like a property of the model rather than a
measurement with a standard error.

### 2.2 How wide the error bar actually is

Take the disk fleet from lesson 4 and cut it to **800 drives, 29 of which
fail**. That is not to exaggerate the effect — 800 records with 29 positives is
a far more ordinary situation than 8,000, and the numbers below are what the
honest version of that situation looks like.

Split it 75/25, fit a logistic regression, and score it with the area under
the ROC curve (AUC) of lesson 4. Then do it again with a different random seed.
Two hundred times:

| | Test AUC |
|---|---|
| worst split | 0.885 |
| best split | **1.000** |
| mean | 0.955 |
| standard deviation | 0.024 |

**The best split reports a perfect classifier.** Every failing drive ranked
above every healthy one, on a model whose honest performance is about 0.95.
Nothing was done wrong to obtain it: the split is legitimate, the model never
saw the test rows, the metric is computed correctly.

Ten of the two hundred splits scored above 0.99.

### 2.3 Why the flattering results are the dangerous ones

A disappointing score makes you frown and keep working. **A delightful score
makes you stop.** You write it down, put it in the report, and move on to the
next thing — and the splits that flatter you are precisely the ones you are
least likely to interrogate.

This is not a claim about dishonesty. It is a claim about attention: the errors
that survive are the ones nobody had a reason to look for.

### 2.4 The arithmetic behind the spread

Each test set here contains **seven failures**. Every AUC quoted above is
computed from seven positive examples against 193 negatives.

The rule of thumb worth carrying: **the precision of a classification metric is
governed by the count of the rarer class, not by the size of the dataset.** Eight
thousand drives with 3.8% failures gives a stable estimate; eight hundred does
not, and the difference is 306 positives against 29.

If you remember one diagnostic question from this lesson, make it: *how many
positive examples are in the test set?*

---

## 3. Selecting a model on one split selects noise

Reporting a noisy number is bad. **Choosing** with a noisy number is worse,
because noise does not average out when it is used to make a decision — it
decides.

Take three logistic regressions differing only in the strength of the penalty,
and let a single split pick the winner. Two hundred times:

| Model | Mean AUC | Standard deviation | Times declared best |
|---|---|---|---|
| C = 0.01 | 0.9549 | 0.0240 | 91 |
| C = 1 | 0.9549 | 0.0240 | 71 |
| C = 100 | 0.9544 | 0.0243 | 38 |

Read the first column, then the last.

**The three models are the same.** Their mean performance agrees to three
decimal places — a difference an order of magnitude smaller than the spread of a
single measurement.

**And yet a winner is declared every time.** The counts in the last column carry
no information about the models. They record which model happened to suit which
random quarter of the data.

Run this once, which is what real life allows, and you will report that one of
these three is best, with a number to support it. **That conclusion is noise
wearing the clothes of a result.**

---

## 4. Cross-validation

### 4.1 The idea

**The picture first.** If one measurement is noisy, take several and average
them. The difficulty is that you cannot afford several test sets — you have one
dataset, and every row spent on testing is a row not spent on training.

Cross-validation resolves the tension by **rotating the role of the test set**.
Cut the data into $k$ equal parts. Train on $k-1$ and test on the one left out.
Repeat until every part has been the test set exactly once.

The result is $k$ scores, and — this is the part that makes it legitimate rather
than a trick — **every row has been predicted exactly once, by a model that
never saw it.** It is not testing on training data with extra steps.

### 4.2 What it estimates, precisely

Let $A(S)$ denote the model that our training procedure produces from a dataset
$S$, and let $R(f)$ be the expected error of a fixed model $f$ on new data from
the distribution $\mathcal{D}$. Cross-validation computes

$$\hat{R}_{CV} = \frac{1}{k}\sum_{j=1}^{k} \hat{R}_{S_j}\!\left(A(S \setminus S_j)\right)$$

where $S_j$ is the $j$-th fold. Each term is an unbiased estimate of the risk of
a model trained on $m(1 - 1/k)$ examples.

Two consequences follow, and both are worth stating because both surprise
people.

**Cross-validation estimates the performance of the *procedure*, not of a
particular fitted model.** There are $k$ different fitted models in the
calculation and the reported number belongs to none of them. It answers: *if I
apply this training procedure to a dataset of about this size, what should I
expect?*

**The estimate is slightly pessimistic.** Every fold trains on a fraction
$1 - 1/k$ of the data, and less training data means a slightly worse model. So
$\hat{R}_{CV}$ estimates the risk of a model trained on 80% of your data (for
$k = 5$), while the model you eventually ship is trained on 100% of it. The bias
is in the safe direction, and it shrinks as $k$ grows — which is the argument
for leave-one-out cross-validation, where $k = m$.

### 4.3 Why the usual error bar is optimistic

Having $k$ scores, the temptation is to quote a standard error
$s/\sqrt{k}$ and treat it as a confidence interval. **Do not**, and the reason is
specific.

That formula assumes the $k$ scores are independent. They are not: any two of
the training sets share a fraction $(k-2)/(k-1)$ of their rows — for $k = 5$,
three quarters of the data is common to any pair. Positively correlated
measurements carry less information than independent ones, so

$$\operatorname{Var}\left(\hat{R}_{CV}\right)
  = \frac{\sigma^2}{k} + \frac{k-1}{k}\,\rho\,\sigma^2$$

where $\rho$ is the correlation between fold scores. The naive formula keeps
only the first term. With $\rho$ appreciably above zero the second term
dominates, and no unbiased estimator of this variance exists in general — a
result of Bengio and Grandvalet (2004).

**In practice:** report the mean and the spread of the folds, describe the
spread as a spread rather than a confidence interval, and treat small
differences between models as unresolved.

### 4.4 Stratification

With 29 positives among 800 rows, an unstratified 5-fold split can produce a
fold containing almost none of them. Measured on the fleet:

| Fold | Plain `KFold` | `StratifiedKFold` |
|---|---|---|
| 1 | 7 failures of 160 | 5 of 160 |
| 2 | **1 failure of 160** | 6 of 160 |
| 3 | 7 of 160 | 6 of 160 |
| 4 | 5 of 160 | 6 of 160 |
| 5 | 9 of 160 | 6 of 160 |

An AUC computed from a single positive example is not a measurement of anything.
**Stratify whenever a class is rare** — which, per lesson 4, is whenever the
problem is interesting.

### 4.5 How many folds

Larger $k$ means each fold trains on more data, so the pessimistic bias of
section 4.2 shrinks. It also means more fits, and training sets that overlap
even more heavily — so the fold scores are more strongly correlated and the
spread you observe understates the uncertainty by more.

**Five or ten**, in practice, and the difference between them rarely matters.
$k = m$ is **leave-one-out**: nearly unbiased, $m$ fits, and the estimate itself
is notoriously high-variance. If you have compute to spare, **repeated k-fold** —
the whole procedure run with several different shuffles and averaged — buys more
than moving from 5 to 10.

### 4.6 When the rows are not independent

Everything above assumed the rows are independent draws from $\mathcal{D}$. That
assumption is violated constantly, silently, and without any warning from the
code.

**The picture first.** Suppose the table holds ten telemetry readings per drive
rather than one. A random split puts nine of a drive's readings in the training
set and one in the test set. The model does not have to learn anything about
failure: it learns to recognise *that drive*, and the test row is very nearly a
duplicate of a training row.

The score is excellent. The model is worthless on a drive it has never seen —
which is the only situation anyone cares about.

The same structure appears whenever rows share a source: several images of the
same patient, several sentences from the same document, repeated measurements of
the same machine, any panel data.

**The rule: split along the axis you must generalise across.**

| Situation | Tool |
|---|---|
| Several rows per entity, must work on unseen entities | `GroupKFold` |
| Data ordered in time, must predict forward | `TimeSeriesSplit` |
| Both | group first, then respect time within groups |

Time deserves its own sentence, because it is stricter than it looks. With a
random split the model trains on Wednesday and is tested on Tuesday. No
deployment ever works that way — real use always predicts forward — so
validation must too. `TimeSeriesSplit` trains on a prefix and tests on what
comes next.

**The symptom of having needed one of these and not used it** is identical in
both cases: a validation score that is excellent and a production score that is
not. It is among the most common reasons a model fails after deployment.

### 4.7 How much it helps

Changing only the seed, forty times:

| | Single 75/25 split | 5-fold cross-validation |
|---|---|---|
| worst | 0.9119 | 0.9439 |
| best | 1.0000 | 0.9588 |
| spread | 0.0881 | 0.0149 |
| standard deviation | 0.0221 | 0.0032 |

**Cross-validation is about seven times more stable across seeds.** Note that
both centre in the same place — it is not more optimistic, it is less arbitrary.

**What it costs** is $k$ fits instead of one. For this course that is seconds;
for a large network it is a real decision, and the usual compromise is a single
validation set large enough that its own error bar is tolerable.

---

## 5. What the error is made of

### 5.1 The picture first

Imagine drawing many different training sets from the same source and fitting
the same model to each. You get many models, and they fail in two distinguishable
ways.

**They may agree with each other and all be wrong.** A straight line fitted to a
curve gives nearly the same straight line every time, and every one of them
misses the curvature. That is **bias**: being consistently, confidently wrong.

**They may disagree wildly with each other.** A twelfth-degree polynomial
fitted to twenty-five points gives a different wild curve every time. That is
**variance**: being unreliable, whatever the average may look like.

And underneath both sits the **noise** in the measurements, which no model
removes because it is not a property of the model.

### 5.2 The decomposition, derived

Let the data be generated as $y = f(x) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$
and $\operatorname{Var}(\varepsilon) = \sigma^2$. Fix a test point $x$, and write
$\hat{f}$ for the model fitted to a random training set. The expectation below is
over both the training set and the noise in the new observation $y$.

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right]
 = \mathbb{E}\left[(f(x) + \varepsilon - \hat{f}(x))^2\right]$$

Because $\varepsilon$ is independent of the training set and has mean zero, the
cross term vanishes:

$$= \sigma^2 + \mathbb{E}\left[(f(x) - \hat{f}(x))^2\right]$$

Now add and subtract $\bar{f}(x) = \mathbb{E}[\hat{f}(x)]$, the average model:

$$\mathbb{E}\left[(f - \bar{f} + \bar{f} - \hat{f})^2\right]
 = (f - \bar{f})^2 + \mathbb{E}\left[(\bar{f} - \hat{f})^2\right]
 + 2(f - \bar{f})\,\mathbb{E}\left[\bar{f} - \hat{f}\right]$$

The last term is zero, because $\mathbb{E}[\hat{f}] = \bar{f}$ by definition.
What remains is

$$\boxed{\;\mathbb{E}\left[(y - \hat{f}(x))^2\right]
 = \underbrace{(f(x) - \bar{f}(x))^2}_{\text{bias}^2}
 + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \bar{f}(x))^2\right]}_{\text{variance}}
 + \underbrace{\sigma^2}_{\text{noise}}\;}$$

**This is an identity, not an approximation.** Notebook 2 checks it numerically
on 300 independently drawn training sets, and the two sides agree to within
$3 \times 10^{-12}$ — which is floating-point equality.

### 5.3 What it looks like on real numbers

Energy against outdoor temperature, 25 training observations, 300 samples, noise
variance 484:

| Degree | bias² | Variance | Noise | Total |
|---|---|---|---|---|
| 1 | 5,250.8 | 723.8 | 484.0 | 6,458.6 |
| 2 | 59.6 | 70.8 | 484.0 | **614.3** |
| 3 | 44.5 | 125.8 | 484.0 | 654.4 |
| 5 | 8.9 | 368.9 | 484.0 | 861.8 |
| 9 | 80.1 | 65,186.0 | 484.0 | 65,750.1 |
| 12 | 153,537.3 | 32,174,344.9 | 484.0 | 32,328,366.2 |

**Degree 1 is a bias problem** — the largest bias² in the table and the smallest
variance. Stable and wrong.

**Degree 12 is a variance problem**, and the magnitude is worth pausing on: 32
million, four orders of magnitude larger than anything else. On 25 points a
degree-12 polynomial nearly interpolates, and an interpolating model is a
machine for amplifying noise.

Its bias² also looks terrible, which seems to contradict the claim that flexible
models have low bias. It does not: with variance this large, the *average* of
the fitted curves is not a meaningful curve, so the bias term inherits the
instability. **Read the two numbers together.**

**The noise column never moves.** 484, at every complexity, because it is a
property of the data. It is the floor: no model, no algorithm and no quantity of
data brings the expected error below it.

That has a practical corollary worth stating plainly. **A reported error below
the noise floor is evidence of contamination, not of excellence.** If you know
your measurement precision and your model beats it, something has leaked.

### 5.4 The optimum depends on how much data you have

The same experiment at three training-set sizes, showing total expected error:

| Degree | n = 25 | n = 60 | n = 150 |
|---|---|---|---|
| 1 | 6,458.6 | 5,998.6 | 5,861.5 |
| 2 | **614.3** | 570.6 | 555.3 |
| 3 | 654.4 | 564.6 | 544.0 |
| 5 | 861.8 | **540.9** | **506.9** |
| 9 | 65,750.1 | 756.6 | 514.0 |
| 12 | 32,328,366.2 | 7,626.9 | 899.0 |

The best degree moves from **2 to 5** as the sample grows. Degree 12 improves by
a factor of four thousand and remains the worst model in its column.

> **"Which model is best?" is not a question about models. It is a question
> about models *and* the quantity of data you have.**

This explains a pattern you will meet constantly: a paper reports that a large
model beat a small one on a dataset far bigger than yours, and the result does
not reproduce for you. Neither experiment was wrong.

---

## 6. Learning curves

The decomposition above required knowing $f$ and $\sigma$. Learning curves give
the same diagnosis from data you actually have.

**The construction.** Train on 15% of the data, then 25%, and so on to 100%,
plotting both the training score and the cross-validated score against the
number of examples used. The *shapes* carry the diagnosis.

Measured on the fleet, with two deliberately broken models:

| Model | Training | Validation | Gap | Validation trend |
|---|---|---|---|---|
| One feature only | 0.715 | 0.718 | −0.002 | 0.718 → 0.718, flat |
| 6 features + 150 noise columns | **1.000** | 0.847 | +0.153 | 0.760 → 0.847, rising |

### 6.1 Reading them

**High bias.** The curves meet, and they meet *low*. The model is equally
mediocre on data it has seen and data it has not, because it has extracted
everything available to it and that is not enough.

- *Helps:* a more flexible model, better features, less regularisation.
- *Does not help:* **more data.** The curve is already flat; more rows land on
  the same plateau. This is the expensive mistake, because collecting data is
  the slowest and costliest item on the list, and the plot says in advance that
  it will buy nothing.

**High variance.** A wide gap between a high training score and a lower
validation score, with the validation curve **still rising** at the right edge.

- *Helps:* more data — and the rising curve is the evidence that it will. Also
  stronger regularisation, fewer features, a simpler model.

### 6.2 The one-line diagnostic

> **Is the gap large, and is the validation curve still rising?** If yes, get
> more data. If the curves have met and levelled off, more data is money spent
> on nothing.

**The predictable mistake here** is prescribing data for a bias problem, and the
instinct behind it is entirely sound — more data almost always helps, it is the
one intervention that never makes a model worse, and it is what every textbook
recommends. It simply does not help *this* failure, and a learning curve is
fifteen lines of code that tells you which failure you have.

---

## 7. Leakage that survives cross-validation

Lesson 2 covered leakage in preprocessing. This section is about the leakage
cross-validation does **not** catch, which is more dangerous precisely because
you believe you are protected.

### 7.1 The catastrophe

Build a table of **2,000 columns of pure random noise**, independent of the
label and of each other. The honest performance of any model on it is 0.500.

Now write the code almost everyone writes:

```python
selector = SelectKBest(f_classif, k=10).fit(noise, y)   # all the data
chosen = noise.loc[:, selector.get_support()]
cross_val_score(model, chosen, y, cv=folds, scoring="roc_auc")
```

**Cross-validated AUC: 0.931.** Four of the five folds land between 0.93 and
0.98 — which is exactly what a trustworthy result looks like.

The error is on the first line. `SelectKBest` was shown every row, including
those that would later serve as test folds. It searched 2,000 columns for the
ones that best matched labels it had already seen, and handed the winners to
cross-validation.

**Cross-validation did not fail. It was lied to.** It faithfully measured a
procedure that had already peeked.

### 7.2 The fix, and the surprise

Put the selection inside the pipeline so it is refitted per fold, on training
rows only. Over 20 different fold seeds the estimate becomes **0.658** — mean of
20, ranging 0.576 to 0.759.

Better. **And not 0.500.**

This is not a bug, and understanding why is the most useful idea in the lesson.

**With 2,000 columns and 800 rows, some columns correlate with the label by
accident across the whole dataset.** That accident is a property of this
particular sample. It is present in every subset of it — in each training fold
and in each test fold alike. A selector fitted honestly on four fifths of the
data finds those columns, and they still "work" on the remaining fifth, because
the spurious correlation was never fold-specific.

> Cross-validation protects you from leaking **between folds**. It cannot
> protect you from having searched a large space of possibilities on a small
> sample. No rearrangement of the same 800 rows removes a pattern that is
> genuinely present in those 800 rows and absent from the world.

**What does help:** a test set held out before any of this began and used
exactly once; fewer candidates; or more rows. Nothing else.

### 7.3 Hyperparameter search is the same problem

Choosing between configurations is choosing, and choosing on data costs the same
honesty. A grid of 25 combinations on that signal-free table:

| | AUC |
|---|---|
| best of 25 candidates (`best_score_`) | **0.7999** |
| average candidate | 0.7265 |
| worst candidate | 0.6840 |
| nested cross-validation | 0.6699 |
| the truth | 0.500 |

The number a practitioner reports is `best_score_`, and it is the **maximum of
25 noisy estimates**. The maximum of a set of noisy numbers is biased upward
even when every one measures the same quantity. **0.7999 is not the performance
of the chosen configuration; it is the performance of whichever configuration
got luckiest.**

The optimism here is **+0.13** — larger than most of the differences anyone
reports between methods.

### 7.4 Nested cross-validation

Measure the search, not the winner.

- **Outer loop:** hold out a fifth of the data.
- **Inner loop:** run the entire search on the remaining four fifths.
- Score the search's chosen model on the held-out fifth. Repeat five times.

The inner loop may overfit its own data as much as it likes; the outer fold was
never part of it. Nested cross-validation is what you use when you need an
honest estimate of *"how well does my whole procedure, tuning included, do?"*

**It is not a way to choose hyperparameters** — it produces $k$ possibly
different winners. It is a way to estimate what choosing costs. Choose on all
the data afterwards, and report the nested figure.

---

## 8. Doing lesson 4's threshold honestly

The debt from section 1.1, now paid. Split three ways: **train** to fit,
**validation** to choose, **test** to report once.

| Policy | Cost on the test set |
|---|---|
| threshold 0.50 (the default) | 10,540 EUR |
| threshold 0.050, chosen on the **validation** set | **4,700 EUR** |
| threshold 0.171, chosen on the **test** set | 3,300 EUR |

The honest threshold is worse on the test set than the cheating one. **It is
supposed to be.** The cheating figure describes a threshold tuned to this
particular test set and there is no reason it survives contact with the next two
hundred drives; the honest figure is an unbiased estimate of what happens next.

Note the cost of doing it properly: we now train on 56% of the data instead of
75%, because the validation set has to come from somewhere. On a small dataset
that hurts — which is the argument for choosing the threshold by
cross-validation on the training portion instead, and keeping the test set whole.

---

## 9. Reproducibility

### 9.1 Seeds

Fix every seed, and write them down:

```python
train_test_split(..., random_state=0)
StratifiedKFold(..., shuffle=True, random_state=0)
LogisticRegression(..., random_state=0)     # solvers that sample
np.random.default_rng(0)                    # your own generation
```

**But a fixed seed buys repeatability, not stability.** Thirty seeds on the
fleet gave scores from 0.913 to a perfect 1.000 — and every one of those runs is
exactly reproducible by anyone with the same code.

`AUC 1.000 (random_state=3)` is fully reproducible and thoroughly misleading.
`AUC 0.951 ± 0.019 over 5-fold cross-validation, seed 0` is reproducible and
honest.

### 9.2 The rest of it

- **Record library versions.** `numpy`, `pandas` and `scikit-learn` change
  defaults between releases; a result nobody can reproduce in two years was not
  reproducible.
- **Name what you could not fix** — thread scheduling, GPU non-determinism.
- **Commit the code that produced the number**, not a cleaned-up version of it.

The course container exists for this reason: it pins the environment so that
your result and your marker's result are the same result.

---

## 10. Summary

- **A test score is a measurement with an error bar.** Holding data out makes
  it unbiased, not precise. On 800 drives the same model scored **0.885 to a
  perfect 1.000** depending only on the seed.
- **The precision is governed by the count of the rarer class.** Ask how many
  positives are in the test set.
- **Selecting on one split selects noise.** Three identical models each won a
  share of 200 splits: 91, 71, 38.
- **Cross-validation** rotates the test set, estimates a *procedure*, is
  slightly pessimistic, and its folds are correlated — so quote the spread, not
  a confidence interval.
- **bias² + variance + noise is an identity**, verified to $3 \times 10^{-12}$.
  The noise floor never moves, and beating it means contamination.
- **The best model depends on how much data you have.**
- **Learning curves prescribe the fix.** Curves meeting low mean bias, and more
  data will not help.
- **Cross-validation protects against fold leakage, not against searching a
  large space on a small sample.** Selection from 2,000 noise columns still
  scored 0.658 done fold-honestly.
- **`best_score_` is the maximum of many noisy estimates.** Nested
  cross-validation measures the search.
- **Split along the axis you must generalise across** — `GroupKFold` when rows
  share an entity, `TimeSeriesSplit` when you must predict forward.
- **Choose on validation data, report on test data, touch the test set once.**

### What belongs in a report

The metric, **the spread**, and how it was estimated. How many **positives** the
test set held. Every **seed** and every **library version**. And which choices
were made on which data.

Those four items map onto the four failures in this lesson: a number without a
spread hides the lottery; a metric without a positive count hides why the spread
is wide; a missing seed makes it unreproducible; and an unstated selection
procedure hides the optimism.

Read papers with the same list. A single figure with no spread, no fold count
and no statement about how the hyperparameters were chosen leaves all four
questions open — and section 7.3 suggests how large the correction tends to be.

### Homework

`Exercises/05_experimental_methodology.md`, due **Friday 30 October 2026**.

### Notation used in this lesson

| Symbol | Meaning |
|---|---|
| $S$, $S_j$ | the dataset; its $j$-th fold |
| $A(S)$ | the model our procedure produces from dataset $S$ |
| $f$, $\hat{f}$, $\bar{f}$ | the truth; a fitted model; the average fitted model |
| $R$, $\hat{R}$ | expected risk; its empirical estimate |
| $\hat{R}_{CV}$ | the cross-validated estimate |
| $\mathcal{D}$ | the unknown distribution the data comes from |
| $\sigma^2$ | the irreducible noise variance |
| $\rho$ | the correlation between fold scores |
| $m$, $k$ | number of examples; number of folds |

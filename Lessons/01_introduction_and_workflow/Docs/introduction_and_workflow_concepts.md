---
title: "Introduction and the Machine Learning Workflow — Key Concepts"
subtitle: "Lesson 1 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "25 September 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly. Use this to find your way back into the text, or to check
before an exercise that there is no word here you could not define.

---

## What learning is

**Learning from data.** Supplying examples and letting a procedure search for a
rule consistent with them, instead of stating the rule yourself. The method of
choice when you can recognise the answer but cannot articulate it. → § 2

**Input and output space.** The set of things you observe and the set of answers
you want, written $\mathcal{X}$ and $\mathcal{Y}$. → § 2.1

**Loss function.** $L(\hat{y}, y)$: the cost of answering $\hat{y}$ when the truth
is $y$. Where you state what counts as a bad mistake, before any model exists.
→ § 2.1

**Expected risk.** The average loss over all data the world might produce,
including data that does not exist yet. The quantity we want small, and the one
we cannot compute. → § 2.1

**Empirical risk.** The same average taken over the finite sample you hold. What
you can compute, and what fitting actually minimises. → § 2.1

**Empirical risk minimisation.** Choosing the model that makes the empirical risk
small. Nearly every method in this course is an instance of it. → § 2.1

---

## Why it can go wrong

**Overfitting.** Driving the empirical risk to zero by memorising the sample
rather than learning the pattern. The central difficulty of the field. → § 2.2

**Generalisation.** How a model performs on data it has never seen — the only
question that matters, and the one a training score cannot answer. → § 2.2

**Train/test split.** Holding data out, never letting the fitting procedure touch
it, and measuring there. The entire reason a test set exists. → § 2.2

**Unbiased estimation.** A test set independent of the fitted model gives a test
score that is right *on average*. A statement about the procedure, not a
guarantee about your one number. → § 2.3

**Independence of the test set.** The condition the whole argument rests on. It
breaks the moment a test row influences any decision — a mean, a ranking, a
comparison. → § 2.3

**Standard error of a proportion.** Roughly $\sqrt{p(1-p)/m}$: how much a measured
accuracy would move if you drew a different test set of the same size. → § 2.4

**Validation set.** The third split, used for choosing between models and
settings, so that the test set stays untouched until the end. → § 2.5

---

## When not to reach for it

**Known rules.** If you can state the rule, write it. A classifier for something
you can specify is a worse program than the specification. → § 3

**Unrepresentative data.** A model learns the distribution it was trained on;
measured performance says little if deployment draws from another one. → § 3

**Unrecoverable, unreviewed errors.** Errors are certain; whether they can be
undone, and whether a human sees the output first, are properties of the
application and the deployment, decidable before any model exists. → § 3

---

## The three kinds of learning

**Supervised learning.** Each example carries a label somebody produced, so the
model can be checked against ground truth. Classification when the target is a
category, regression when it is continuous. → § 5.1

**Unsupervised learning.** No targets: the goal is structure — groups, a
lower-dimensional description, unusual points. There is still a loss; what is
missing is something to check the answer against. → § 5.2

**Self-supervised learning.** The target is manufactured from the input by hiding
part of it. Nobody annotates anything and the supervision is still genuine; this
is how modern large models are trained. → § 5.3

---

## The workflow, and its order

**Framing.** Deciding what is predicted, from what, and which mistake is worse —
before modelling, because a metric chosen after seeing results is chosen to
flatter them. → § 6

**Baseline.** The simplest thing that could work: the majority class, the mean, a
threshold on one variable. Without one, a score has no meaning. → § 6

**Pipeline.** An object that binds preprocessing to the model, so that every step
which learns from data is fitted on training data alone — structurally, not by
remembering to. → § 6

**Error analysis.** Asking where a model fails and whether the failure is
systematic. Worth more than another decimal place of accuracy. → § 6

---

## Four ways a result misleads

**Data leakage.** Information from the test set reaching the training procedure —
through a scaler, a feature selection, or a column recorded after the outcome. No
error is raised. → § 7

**Class imbalance.** When the interesting class is rare, accuracy measures the
class you are not interested in. → § 7

**Shortcut feature.** A column that is a consequence of the label rather than a
predictor of it, and which will not exist when a prediction is needed. No metric
detects it; knowing how the data was recorded does. → § 7

**Single-split noise.** One split is one draw from a distribution. A result quoted
with no indication of variability is incomplete. → § 7

---

## Limits worth stating on day one

**Inherited bias.** A model is a compressed summary of its training data,
injustices included, and it reproduces them with an appearance of objectivity
that makes them harder to contest. → § 8

**Correlation, not cause.** Nothing in empirical risk minimisation separates a
cause from a coincidence that predicts well in this sample. → § 8

**Beyond accuracy.** Explainability, recoverability of errors, and who bears their
cost are engineering requirements that belong in the framing step. → § 8

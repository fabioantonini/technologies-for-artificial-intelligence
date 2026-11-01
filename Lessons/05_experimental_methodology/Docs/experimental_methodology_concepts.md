---
title: "Experimental Methodology — Key Concepts"
subtitle: "Lesson 5 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "6 November 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly. Use this to find your way back into the text, or to check
before an exercise that there is no word here you could not define.


## A score is a measurement

**Unbiased is not precise.** Holding data out makes a test score right on average,
over the test sets one might have drawn, for the model that was fitted.
It does nothing whatever to make it repeatable, and the two are routinely
confused. → § 2.1

**Error bar on a test score.** A score is computed on a finite sample; a different
sample gives a different number, and how different is a property of the sample's
size. → § 2.2

**The count of the rarer class.** What governs the precision of a metric
computed on the rare class — recall, and in practice the area under the curve (AUC) — not the size of the
dataset. The diagnostic question is
*how many positives are in the test set?* → § 2.4

**Why it is the rarer class.** A metric computed over the positives is a proportion
over $m_+$ draws, so its standard error carries $m_+$ in the denominator and the
healthy rows do not appear in it at all. → § 2.4

**Why flattering results are dangerous.** A disappointing score makes you keep
working; a delightful one makes you stop. The errors that survive are the ones
nobody had reason to look for. → § 2.3

**Selection on noise.** Choosing between models with a noisy number is worse than
reporting one, because noise does not average out when it is used to decide — it
decides. → § 3


**Parameters and hyperparameters.** The coefficients and intercept a fit estimates
from the data are parameters. What it does not estimate — penalty strength, number
of neighbours, tree depth, and the decision threshold applied after the fit — are
hyperparameters: they govern the fitting or the decision, and are chosen from
outside by comparison. Not "fixed before the fit": the threshold is chosen after
it. → § 3

**`C` in scikit-learn.** The reciprocal of a penalty strength, so small `C` is heavy
regularisation. It runs opposite to lesson 3's $\lambda$. → § 3

**Two recipes for choosing a hyperparameter.** One fixed before the fit needs its
own fits for every candidate, then a refit of the winner; one chosen after the fit,
like a threshold, is tried on the probabilities of a single fit. Either way the test
set is scored once, at the end. → § 3


## Cross-validation

**k-fold cross-validation.** Cut the data into $k$ parts, train on $k-1$ and score
on the one left out, the validation fold, rotating until every row has been predicted exactly once by a
model that never saw it. → § 4.1

**Fold and round.** A fold is a part of the data; a round is one fit and one
score, so one model per round. scikit-learn calls a round a split. → § 4.1

**What it estimates.** The performance of a *procedure* applied to a dataset of
about this size — not of any one fitted model, of which there are $k$. → § 4.2

**Its pessimism.** Each round's model trains on a fraction of the data, so the estimate
describes a slightly smaller training set than the one you will ship. The bias is
in the safe direction. → § 4.2

**Confidence interval.** An estimate plus or minus about two standard errors, a
range built to contain the true value in 95 of 100 repeated experiments. → § 4.3

**Why the usual error bar is optimistic.** $s/\sqrt{k}$ assumes independent
scores. Overlapping training sets can correlate them, and a standard deviation
computed from a handful of scores is itself rough, too small more often than
not. → § 4.3

**The variance of a mean of correlated measurements.** The variance of a sum
collects the covariances as well as the variances, which is the term
$s/\sqrt{k}$ silently sets to zero. → § 4.3

**Stratified folds.** Folds that keep each class in its overall proportion, so a
rare class cannot land almost entirely in one of them. → § 4.4

**Repeated k-fold.** The whole cross-validation run with several shuffles and
averaged; it removes the dependence on how the folds were cut, not the uncertainty
about new data. → § 4.5

**Grouped folds.** Splitting along the axis you must generalise across, when
several rows share a source — a patient, a document, a machine. The group label
is passed apart from the features and is never one of them. → § 4.6

**Time-ordered splits.** Training on a prefix and testing on what comes next,
because real use always predicts forward. → § 4.6


## What error is made of

**Bias.** Being consistently wrong: models fitted to different samples all miss
the truth in the same way. Not the bias of an estimate, which sections 2 and 7
use. → § 5.1

**Variance.** Being unreliable: models fitted to different samples disagree wildly,
worst at the edges of the range, where all the points lie on one side. → § 5.1

**Irreducible noise.** The part of the error that is a property of the data, not of
the model. No method and no quantity of data goes below it. → § 5.1

**The decomposition.** Expected squared error equals bias squared plus variance
plus noise. An identity, not an approximation. → § 5.2

**The noise floor.** The value that decomposition's last term sets. A test error
*well below* it is evidence of leakage, not of excellence (a small test set can
dip under it by luck); a training error can go
below it by fitting the noise. → § 5.3

**Model choice depends on sample size.** The complexity that minimises total error
moves as data grows — which is why a paper's result on a larger dataset may
honestly fail to reproduce on yours. → § 5.4

**Training error against validation error.** The diagnosis without the truth:
training error high means bias; validation far above training means variance.
Both come from cross-validation inside the training data, never from the test
set. → § 5.5


## Learning curves

**Learning curve.** Training and cross-validated scores plotted against the number
of examples used. The *shapes* carry the diagnosis. → § 6

**High bias.** The curves meet, and meet low. A more flexible model or better
features help; more data does not. → § 6.1

**High variance.** A wide gap between a high training score and a lower validation
one. What to do depends on the *slope*: still rising by more than its spread means
collecting more is likely to pay, flattened means regularise or simplify. → § 6.1


## Leakage cross-validation cannot catch

**Fold leakage.** A step fitted on all the data before the folds are cut, so the
selection has already seen every validation row. Cross-validation does not fail here; it
is lied to. → § 7.1

**Searching a large space on a small sample.** The residual optimism that survives
even an honest fold-wise fit: a spurious correlation present in *these* rows is
present in every subset of them. → § 7.2

**Optimism of `best_score_`.** The maximum of many noisy estimates is biased upward
even when every estimate measures the same quantity. → § 7.3

**Nested cross-validation.** Run the entire search inside an outer training fold and
score on the outer test fold. It measures what *searching* costs; it is not a way to
choose. → § 7.4

**Train, validation, test.** Fit on the first, choose on the second, report once on
the third. The honest number is expected to be worse than the cheating one. → § 8


## Reproducibility

**Seeds.** Fix them and write them down — but a fixed seed buys repeatability, not
stability: a perfectly reproducible run can still be a lucky one. → § 9.1

**Library versions.** Defaults change between releases, so a result nobody can
reproduce in two years was never reproducible. → § 9.2

**What belongs in a report.** The metric, the spread and how it was estimated; how
many positives the test set held; every seed and version; and which choices were
made on which data. → § 10

---
title: "Experimental Methodology — Key Concepts"
subtitle: "Lesson 5 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "23 October 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly. Use this to find your way back into the text, or to check
before an exercise that there is no word here you could not define.


## A score is a measurement

**Unbiased is not precise.** Holding data out makes a test score right on average.
It does nothing whatever to make it repeatable, and the two are routinely
confused. → § 2.1

**Error bar on a test score.** A score is computed on a finite sample; a different
sample gives a different number, and how different is a property of the sample's
size. → § 2.2

**The count of the rarer class.** What actually governs the precision of a
classification metric — not the size of the dataset. The diagnostic question is
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


## Cross-validation

**k-fold cross-validation.** Cut the data into $k$ parts, train on $k-1$ and test
on the one left out, rotating until every row has been predicted exactly once by a
model that never saw it. → § 4.1

**What it estimates.** The performance of a *procedure* applied to a dataset of
about this size — not of any one fitted model, of which there are $k$. → § 4.2

**Its pessimism.** Each fold trains on a fraction of the data, so the estimate
describes a slightly smaller training set than the one you will ship. The bias is
in the safe direction. → § 4.2

**Why the usual error bar is optimistic.** The fold scores are not independent: any
two training sets share most of their rows, and positively correlated measurements
carry less information than independent ones. → § 4.3

**The variance of a mean of correlated measurements.** The variance of a sum
collects the covariances as well as the variances, which is the term
$s/\sqrt{k}$ silently sets to zero. → § 4.3

**Stratified folds.** Folds that keep each class in its overall proportion, so a
rare class cannot land almost entirely in one of them. → § 4.4

**Grouped folds.** Splitting along the axis you must generalise across, when
several rows share a source — a patient, a document, a machine. → § 4.6

**Time-ordered splits.** Training on a prefix and testing on what comes next,
because real use always predicts forward. → § 4.6


## What error is made of

**Bias.** Being consistently, confidently wrong: models fitted to different samples
agree with each other and all miss the truth. → § 5.1

**Variance.** Being unreliable: models fitted to different samples disagree wildly,
worst where the data runs out. → § 5.1

**Irreducible noise.** The part of the error that is a property of the data, not of
the model. No method and no quantity of data goes below it. → § 5.1

**The decomposition.** Expected squared error equals bias squared plus variance
plus noise. An identity, not an approximation. → § 5.2

**The noise floor.** The value that decomposition's last term sets. A reported error
*below* it is evidence of contamination, not of excellence. → § 5.3

**Model choice depends on sample size.** The complexity that minimises total error
moves as data grows — which is why a paper's result on a larger dataset may
honestly fail to reproduce on yours. → § 5.4


## Learning curves

**Learning curve.** Training and cross-validated scores plotted against the number
of examples used. The *shapes* carry the diagnosis. → § 6

**High bias.** The curves meet, and meet low. A more flexible model or better
features help; more data does not. → § 6.1

**High variance.** A wide gap between a high training score and a lower validation
one. What to do depends on the *slope*: still rising means collect more, flattened
means regularise or simplify. → § 6.1


## Leakage cross-validation cannot catch

**Fold leakage.** A step fitted on all the data before the folds are cut, so the
selection has already seen every test row. Cross-validation does not fail here; it
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

---
title: "Classification and Evaluation Metrics — Key Concepts"
subtitle: "Lesson 4 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "16 October 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly; the handout's own Section 11 is where the *findings* are
collected. Use this to find your way back into the text, or to check before an
exercise that there is no word here you could not define.

---

## The model

**Logistic regression.** A linear model of the log-odds, squashed into a
probability by the sigmoid. Same features and coefficients as Lesson 3, a
different question. → § 2

**Sigmoid.** The function $\sigma(z) = 1/(1 + e^{-z})$: takes any real number and
returns one between 0 and 1, steep in the middle and flat at both ends. → § 2.2

**Odds and log-odds.** The odds are the probability of an event divided by the
probability of its complement; the log-odds are their logarithm, and they range
over the whole real line, which is what a linear model can safely predict. → § 2.3

**Odds multiplier.** $e^{w_j}$: what a one-unit increase in a feature does to the
odds. It multiplies the odds, not the probability — the commonest misreading in
the lesson. → § 2.4

---

## The cost function

**Cross-entropy, or log loss.** The cost that maximum likelihood produces for a
Bernoulli outcome: the negative logarithm of the probability the model gave to
what actually happened. → § 3.3

**Maximum likelihood.** Choose the parameters that make the observed data as
unsurprising as possible. Cross-entropy is what that principle yields here rather
than something anyone designed. → § 3.1

**Why not squared error.** Its gradient carries a factor $\sigma'(z)$ that
vanishes exactly where the model is most confidently wrong, so the optimiser asks
for the smallest correction where the largest is needed. → § 4.2

**Convexity.** The cross-entropy cost has no local minima to be trapped in — but
convex does not mean solvable in closed form, and there is no normal equation for
logistic regression. → § 4.3

---

## Reading a classifier

**The accuracy trap.** When one class is rare, answering "the majority class"
every time scores well and detects nothing. This is the number to remember from
the lesson. → § 5.1

**Confusion matrix.** The four counts — true and false, positive and negative —
that every metric below is built from. Count before you summarise. → § 6.1

**Precision.** Of the cases the model flagged, how many really were positive. The
denominator is the alarms raised. → § 6.2

**Recall.** Of the cases that really were positive, how many the model found. The
denominator is the failures that existed. → § 6.2

**F1 and F-beta.** The harmonic mean of precision and recall, so a perfect score
on one axis cannot buy a pass on the other; $\beta > 1$ weights recall more
heavily. → § 6.3

**Macro against weighted averaging.** Macro gives every class an equal say;
weighted gives each class its share of the data, which on an imbalanced problem
means the majority class talking. → § 6.4

---

## Turning a score into a decision

**The threshold.** The line between "leave it" and "act on it". Nothing about the
fitted model changes as it moves — only which side of it each case falls. → § 7.1

**Cost-optimal threshold.** $t^{*} = C_{FP} / (C_{FP} + C_{FN})$, a function of
the two error costs alone: not of the model, the dataset, or the class balance.
→ § 7.2

**Receiver operating characteristic (ROC) curve.** True positive rate against
false positive rate, swept over every threshold. Its shape says how fast the
model catches positives relative to how fast it raises false alarms. → § 8.1

**Area under the curve (AUC).** The probability that the model scores a randomly
chosen positive above a randomly chosen negative. It measures ranking, and is
blind to whether the probabilities themselves mean anything. → § 8.2

**Where AUC misleads.** It is unmoved by class imbalance, because its denominator
is the whole negative class; precision is not. On a rare-event problem report the
precision-recall curve beside it. → § 8.3

---

## Working with imbalance

**Class weights.** Make each rare example count as several during training. Note
that this moves the operating point rather than teaching the model anything new —
the threshold by another name. → § 9

**Resampling.** Oversampling the minority class, or synthesising new minority
examples. Same caution, plus one more: it must happen inside the
cross-validation fold. → § 9

---

## More than two classes

**Softmax.** One score per class, exponentiated and normalised to a probability
distribution. For two classes it reduces to the sigmoid. → § 10.1

**Categorical cross-entropy.** The same idea as the binary loss, generalised: the
negative logarithm of the probability assigned to the true class. → § 10.1

**One-against-rest metrics.** Precision and recall computed one class at a time,
each in turn treated as the positive one. → § 10.2

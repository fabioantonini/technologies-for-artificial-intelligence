---
title: "k-NN, Naive Bayes and Support Vector Machines — Key Concepts"
subtitle: "Lesson 6 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "30 October 2026 · a one-page index"
---

An index, not a summary. Each entry says what a term means and where the handout
treats it properly. Use this to find your way back into the text, or to check
before an exercise that there is no word here you could not define.


## k-nearest neighbours

**The algorithm.** To classify a new point, find the $k$ training points closest to
it and take the majority vote. There is no training step. → § 2.1

**Lazy learner.** A method whose "fitting" is storing the data. An unusually honest
name for an algorithm. → § 2.1

**Euclidean distance.** The square root of the summed squared differences per
feature — which is why every feature must be on a comparable scale before it is
used. → § 2.1

**k as the bias-variance dial.** Small $k$ follows every point including the
mislabelled ones; large $k$ smooths until it stops following real structure. At
$k=1$ the training accuracy is exactly 1 on any dataset. → § 2.2

**Prediction cost.** No training time, paid back at every prediction: a naive
implementation compares the query against every training row. → § 2.3

**The model is the dataset.** You cannot ship the model without shipping the
training data — a legal question as much as a practical one. → § 2.3


## The curse of dimensionality

**Distance concentration.** As dimensions grow, every distance converges on the same
value, and "nearest" stops being a meaningful claim. → § 3.2

**What the curse is.** Not that high-dimensional problems are unlearnable — that
methods built on *distance* lose their footing, because the quantity they depend on
stops varying. → § 3.3

**Why adding features is not free here.** With a linear model an irrelevant feature
costs a coefficient near zero. With k-nearest neighbours it costs a dimension in
the distance, and dimensions are what the method is made of. → § 3.3


## Naive Bayes

**Bayes' rule for a class.** Turn the probability of a class given the data into a
likelihood times a prior, dropping the denominator because it is the same for every
class. → § 4.1

**Where the difficulty is.** Estimating the probability of an exact combination of
readings — a density in as many dimensions as you have features. → § 4.1

**The naive assumption.** Given the class, the features are independent of one
another. → § 4.2

**What it buys.** One $n$-dimensional estimation problem becomes $n$ one-dimensional
ones, trained in a single pass, needing very little data per feature. → § 4.2

**Independence *given the class*.** Not the same as independence overall, and the
distinction is what makes the assumption testable on real data. → § 4.3

**Where it fails.** When the signal is an *interaction* — a rule about the two
features together that neither shows alone. No quantity of data repairs it, because
the model cannot represent what is being asked. → § 4.4

**Uncalibrated probabilities.** Its scores can be useful as a ranking while being
worthless as probabilities, in either direction. → § 4.5

**When to use it anyway.** Very high dimensions with little data, and as a baseline
that trains in one pass. → § 4.6


## Support vector machines

**The margin.** The width of the slab you can push out from the boundary before it
touches the nearest point of either class. → § 5.1

**Why maximise it.** A boundary close to a training point is one perturbation away
from being wrong. The widest margin tolerates the most movement in the data — a
statement about generalisation, not about fit. → § 5.1

**Support vectors.** The points touching the slab. They alone determine the answer;
move any other point and nothing changes. → § 5.1

**Soft margin.** Allowing violations and charging for them, because real data is not
separable. → § 5.2

**$C$, the price of an error.** Large $C$ makes violations expensive and the model
contort; small $C$ buys a wider, calmer boundary. Note the direction: **large $C$
means less regularisation**. → § 5.2

**A high support-vector fraction.** A free warning that the model is struggling to
find room — that no slab separates anything. → § 5.3

**The kernel trick.** Map into a space where a linear boundary works, and compute
only the inner products there, never the map itself. → § 5.4

**Radial basis function (RBF) kernel.** The standard choice, corresponding to an
infinite-dimensional feature space at the cost of one exponential per pair. → § 5.4

**$\gamma$.** How far a single training point's influence reaches. Large $\gamma$
lets the boundary dissolve into islands around individual points, including the
mislabelled ones. → § 5.5


## Choosing

**Family before tuning.** On the same data the gap between the best and worst family
here is larger than any difference this course has shown between a model and a tuned
version of itself. → § 6

**How to choose.** Few features and an odd-shaped boundary, many features and little
data, a need for calibrated probabilities, a need for a small fast model — each
points at a different one of the three. → § 6

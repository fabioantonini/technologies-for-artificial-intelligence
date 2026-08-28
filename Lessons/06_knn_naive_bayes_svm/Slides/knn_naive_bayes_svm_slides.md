---
title: "Lesson 6: k-NN, Naive Bayes and Support Vector Machines"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "30 October 2026"
---

# Before we start

- Exercise 5 was due today
- The score that was too good, and what the honest number cost you

::: notes
Collect exercise 5. The marks were for the diagnosis, not for the final figure,
so say one sentence about what the class found and one about what it missed.

There was more than one problem planted. Ask out loud how many people found the
second one, and how many stopped after the first - stopping after the first is
the realistic failure, because once a score drops from excellent to plausible
the pressure to keep looking disappears.

Then the sentence that links last week to today: everything from now on is
evaluated with the apparatus you built in lesson 5. Today adds three model
families, and each of them offers new ways to be wrong while looking right.
:::

# Today: the first lesson that offers a choice

- **k-nearest neighbours (k-NN)**: remember, and vote
- The **curse of dimensionality**: the bill for that
- **Naive Bayes**: one strong assumption, bought cheap
- **Support vector machines (SVM)**: margins, and kernels
- Which family to reach for, and why that beats tuning

::: notes
Agenda, and one framing sentence that is worth saying deliberately.

Lessons 3 and 4 built linear models and spent most of their effort on fitting
them honestly. Lesson 5 built the apparatus for telling whether a model works.
None of that told them WHICH MODEL TO REACH FOR. This is the first lesson that
offers a choice, and the choosing is the content.

Stress that these are not three ways of doing the same thing. They attack
classification from three unrelated directions: one makes no assumptions and
does no fitting at all, one makes an assumption that is usually false and wins
anyway, one changes the criterion and then changes the space. The differences
between them are the point, not the accuracy figures.

Handout section 1 sets this out in a paragraph each.
:::

# 1,200 pumps, two readings each

- Vibration in Hz, pressure in bar
- A pump is faulty when its readings fall **outside the design envelope**,
  too low as readily as too high
- **61.3%** faulty; 4% of the labels are deliberately flipped

::: notes
Describe the physical fact first, before showing the picture, and let them
predict what it implies. A pump has a design operating envelope. It can fail by
running too slow just as easily as by running too hot.

Ask the room what that does to the shape of the two classes before you advance.
Somebody will get it: healthy in the middle, faulty all around.

The 4% flipped labels are lesson 5's noise floor arriving in a classification
problem. No model here can exceed roughly **0.96**, and that is the number every
score today should be compared against - not against 1.000, and not against
each other in isolation.

Handout section 1.1.
:::

# The shape of the problem

![](pump_scatter.png)

::: notes
Give them a moment before saying anything. Then ask for a straight line that
separates these two classes, and let them try.

The healthy pumps form a disc around the design point; the faulty ones form the
annulus around them. There is no line. Not a badly chosen line - no line at all,
because the class you want to isolate is completely surrounded.

Worth naming what is unusual here, since most textbook pictures are blobs side
by side: the geometry comes from the physics, not from an attempt to be awkward.
Any "within tolerance versus out of tolerance" problem has this shape, and that
covers a large fraction of industrial classification.

The scattered wrong-coloured points inside each region are the 4% flipped
labels. Point at two of them now - they come back at k = 1.
:::

# What a straight boundary costs

| Model | Cross-validated accuracy |
|---|---|
| Always predict the majority class | 0.613 |
| Logistic regression (lesson 4) | **0.613 ± 0.000** |

::: notes
Read the two rows and let the equality sit there.

The thing to say clearly, because it is easy to under-read: logistic regression
has not been narrowly beaten by the baseline. It has learned NOTHING AT ALL. It
predicts "faulty" for every pump, because with a straight boundary that is
genuinely the best answer available to it.

Then point at the standard deviation, which is the real tell: **± 0.000**. A
model that gives everything the same answer is perfectly consistent across
folds. Give them that as a diagnostic to keep - a suspiciously stable
cross-validation score often means a model that is not using its input.

Connect back to lesson 5's habit: quote the baseline first, always. Without the
0.613 in the row above, 0.613 looks like a mediocre result rather than an absent
one.
:::

# k-nearest neighbours: the whole algorithm

- To classify a new point, find the **k closest** training points and take the
  majority vote
- There is no training step: "fitting" means **storing the data**
- Called a **lazy learner**: an unusually honest name for an algorithm

::: notes
The whole method is one sentence, and it is worth pausing on how strange that
is after three weeks of optimisation. There is no cost function, no gradient, no
parameters. Nothing is estimated.

Ask the room what could possibly go wrong with a method that makes no
assumptions. The answer is the next fifteen minutes: making no assumptions means
having no way to ignore anything, and that turns out to be expensive.

Two decisions hide inside that sentence - what "closest" means, and what k is  - 
and both are consequential. Take them in that order.

Handout section 2.1.
:::

# Distance is all it has

$$d(x, x') = \sqrt{\sum_{j=1}^{n} (x_j - x'_j)^2}$$

::: notes
Euclidean distance, almost always. Do not dwell on the formula; dwell on what it
implies, which is the next slide.

The observation that matters: every feature contributes to that sum through its
own units. There is nothing in the expression that knows Hz from bar. The method
has no other channel through which the data reaches it - no coefficients, no
weights, no notion that one column might matter more than another.

Say that plainly: for k-NN, the distance IS the model. Anything that distorts
the distance distorts everything.

Handout section 2.1 has it with the scaling argument attached.
:::

# Scale first, or vibration decides everything

- Vibration runs over **tens of Hz**; pressure over **a couple of bar**
- Unscaled, a one-bar pressure difference is invisible beside a ten-Hz one
- Lesson 2's scaling argument, arriving with **immediate consequences**

::: notes
This is where lesson 2 stops being hygiene and starts being the difference
between a working model and a broken one.

Work the arithmetic out loud. Two pumps differing by 10 Hz and 0 bar are 10
apart. Two differing by 0 Hz and 1 bar are 1 apart. So the algorithm considers
the second pair ten times more similar - for no reason other than the units
somebody chose when the sensors were installed.

The consequence: without standardisation, vibration picks every neighbour by
itself and pressure is decorative. Change the pressure sensor to millibar and
the model's answers change. That should feel unacceptable, and it is.

Practical instruction: k-NN goes in a Pipeline with StandardScaler, always, and
the scaler is fitted inside the fold - which is last week's lesson, not a new
one.
:::

# k is the bias-variance dial, made visible

- **Small k**: the boundary follows every point, mislabelled ones included.
  Low bias, high variance
- **Large k**: the vote is taken over a wide neighbourhood, and eventually
  stops following real structure. High bias, low variance
- At k = 1 the training accuracy is **exactly 1.000**. Always, on any dataset

::: notes
Lesson 5 decomposed error into bias and variance and had to build 300 parallel
universes to show it. Here you watch the same trade-off by turning one integer,
and that is why this slide is worth more time than its size suggests.

The k = 1 claim is the one to make them prove to themselves. Ask why the
training accuracy is exactly 1.000 at k = 1, and wait. The answer is that every
training point is its own nearest neighbour, at distance zero. So it votes for
itself and wins.

That is the purest illustration in the whole course of lesson 5's point: a
training score can be perfect and measure absolutely nothing. Not
approximately nothing - nothing, by construction, independent of the data.

Handout section 2.2.
:::

# Measured on the pumps

| k | Training | Cross-validated |
|---|---|---|
| 1 | **1.000** | 0.912 |
| 5 | 0.953 | **0.944** |
| 15 | 0.948 | 0.938 |
| 51 | 0.940 | 0.933 |
| 401 | 0.828 | 0.708 |

::: notes
Read the first row, then the last, then the middle.

Row one: training 1.000, honest 0.912. The gap is the whole of lesson 5 in two
numbers on one line.

The best value is k = 5, at 0.944 - wide enough to average out the flipped
labels, narrow enough to still follow the boundary. Note that it is not
dramatically better than 15 or 51: the choice of k is forgiving over a wide
range, which is worth saying because they will otherwise grid-search it to death.

The last row is the interesting failure. At k = 401 the training score has
collapsed too, which distinguishes it from overfitting: this model is bad
everywhere, not just on data it has not seen. That is what bias looks like.

Compare 0.944 against the ceiling of 0.96, not against 1.000.
:::

# The two curves, and the two lines that bound them

![](knn_choosing_k.png)

::: notes
Point at the left edge first. The training curve starts at exactly 1.000 and
falls; that is the k = 1 identity, drawn.

Then the two horizontal lines, which are the honest frame for reading any score
in this lesson: the noise ceiling at 0.96, which nothing can exceed, and the
majority baseline at 0.613, which everything should.

The gap between the two curves is variance, and watch it close as k grows.
Where they meet and then both fall together, you have run out of variance and
started buying bias. The minimum of the honest curve sits just before that.

Ask them which end of this plot they would have picked by looking at the
training curve alone. The answer is the far left, which is the worst honest
model on the plot bar one.
:::

# The same data at three values of k

![](knn_boundaries.png)

::: notes
Three panels, and each says something different.

k = 1: the boundary is ragged, with islands around individual points. Those
islands are the mislabelled 4% - the model has carved out a small territory for
each one. That is variance made visible.

k = 15: a clean disc, close to the envelope that actually generated the data.
This is roughly the truth.

k = 401 is the one worth the time, because it does not do what people expect.
It has not collapsed to a single class. The boundary has INFLATED past the true
envelope and now swallows faulty pumps. With 401 votes taken over a large
neighbourhood, the majority class wins in regions where it should not. The model
has stopped following the boundary and started averaging over it.

Handout section 2.2 carries the same three panels.
:::

# What k-NN costs

- No training time at all: you pay at **every prediction** instead
- A naive implementation compares the query against **every** training row:
  `O(mn)` for m rows and n features
- **The model is the dataset.** You cannot ship one without the other

::: notes
The first cost is engineering. For 1,200 pumps it is nothing. For ten million
rows answering a thousand queries a second it is the entire problem, and it is
why approximate nearest-neighbour indexes are a small industry. Mention that the
scikit-learn default uses a k-d tree or ball tree, which helps in low dimensions
and stops helping in high ones - for reasons that are the next segment.

The second cost is easy to miss and matters more often. Every other model in
this course compresses its training data into parameters and then throws the
data away. k-NN cannot: to make a prediction it must hold the training set.

Spell out what that means when the training set is medical records or customer
transactions. Shipping the model means shipping the data - to a phone, to a
customer's server, to a third party. That is a legal question before it is a
technical one. This lesson's Resources document and lesson 2's cover it.
:::

# Now the price: add columns of nothing

- Extra columns of **pure noise**, drawn from a normal distribution
- The two real readings are **untouched**: the problem is exactly as solvable
- Only the number of columns changes

::: notes
Set the experiment up carefully before showing the result, because the whole
force of it depends on them believing that nothing was taken away.

Say it twice if necessary: the signal is still there. Both original columns are
present, unmodified, and still sufficient to solve the problem perfectly. We are
adding columns that contain nothing - no relationship to the label, no
relationship to each other.

Ask for a prediction before you advance. Most of the room will say performance
degrades gently, or not at all, because the useful information is intact. That
instinct is exactly right for a linear model and exactly wrong here, and the
next slide is why.

Handout section 3.1.
:::

# The signal never left

| Total columns | Accuracy | Above baseline |
|---|---|---|
| 2 | 0.938 | +0.325 |
| 12 | 0.762 | +0.149 |
| 27 | 0.662 | +0.049 |
| 52 | 0.602 | **−0.011** |
| 102 | 0.578 | −0.035 |

::: notes
Read down the last column, slowly, and stop at the fourth row.

At 52 columns - that is 2 real and 50 empty - k-NN is BELOW the majority
baseline. A model that ignored the data entirely and always answered "faulty"
would now do better than a model that looked at it.

Let that be uncomfortable before explaining. Nothing was removed. Fifty columns
of nothing were added, and they destroyed a method that a moment ago was near
the noise ceiling.

Point out that the damage is already serious at 12 columns: 0.938 down to 0.762
for ten empty columns. Twelve features is not a large table. Nobody would look
at a twelve-column dataset and think "high-dimensional".

The mechanism is on the next three slides, and it is geometry, not statistics.
:::

# Why: distances stop varying

- Scatter points in a cube, pick one, measure its **nearest** and its
  **farthest** neighbour
- In two dimensions those are very different numbers, which is what makes
  "nearest" a meaningful word
- Each new dimension adds its own squared difference, and the contributions
  **average out**

::: notes
Give the picture in words before any number appears.

In two dimensions, the nearest point is genuinely near and the farthest is
genuinely far. That difference is the entire basis on which k-NN operates: it
assumes that "the five nearest" is a meaningfully different set from "five
chosen at random".

Now add dimensions. Every new coordinate contributes its own squared difference
to every pairwise distance. Those contributions are independent draws from the
same distribution, so by the law of large numbers their average concentrates,
and every distance converges on the same value.

Say explicitly that this has nothing to do with k-NN, or with machine learning
at all. It is a fact about high-dimensional Euclidean space. k-NN is simply the
first method we have met that depends on it.

Handout section 3.2 does it properly.
:::

# The nearest point stops being near

![](distance_concentration.png)

::: notes
The ratio of the nearest distance to the farthest, plotted against dimension. It
climbs towards 1, and 1 is the point at which every point is the same distance
from every other point.

The shape is the thing to notice: the damage is done early. Most of the rise
happens in the first fifty dimensions, well below anything anyone would call
high-dimensional. By the time you are worrying about a thousand features it is
long over.

Connect it back to the previous table so nobody treats this as a separate
curiosity. The reason 52 columns took k-NN below the baseline is drawn on this
plot: at that width, "nearest" has already stopped meaning much.
:::

# Nearest divided by farthest

| Dimensions | nearest ÷ farthest |
|---|---|
| 2 | 0.016 |
| 10 | 0.263 |
| 50 | 0.592 |
| 100 | **0.701** |
| 500 | 0.855 |

::: notes
The same curve as numbers, because these are the ones worth carrying out of the
room.

Start at the top. In two dimensions the nearest point is about 2% as far away as
the farthest. "Nearest" is a strong claim there - it picks out something
genuinely special.

Then walk down. By ten dimensions it is a quarter. By fifty it is more than half.

Give them the ten-dimension row as the practical warning, because it is the one
that applies to their work: at a size that feels completely ordinary, the
nearest neighbour is already a quarter of the way to being an arbitrary point.
:::

# In 100 dimensions the nearest point is 70% as far as the farthest

- In **two** dimensions it is **2%**: "nearest" picks out something special
- In **one hundred** it is **70%**: the nearest point is barely nearer than a
  random one
- A vote among "the five nearest" becomes a vote among five taken at random

::: notes
This is the number to carry out of today, so slow down and say it twice.

Seventy per cent. The closest point in your entire dataset is seven tenths of
the way to the furthest one. There is almost nothing left of the idea of a
neighbourhood.

And then the sentence that makes it operational: if the nearest five points are
barely nearer than five chosen at random, a majority vote among them is barely
different from a majority vote among five random labels - which is the base
rate, which is exactly what the 52-column row showed.

Ask them to write the number down. Derivations fade; this one tends not to. It
comes back at the close.

Handout section 3.2.
:::

# What the curse is, and is not

- **Not** that high-dimensional problems are unlearnable: lesson 9's networks
  work in thousands of dimensions
- That **methods built on distance lose their footing**, because the quantity
  they depend on stops varying
- k-NN, k-means (lesson 8), and radial basis function (RBF) kernels, all of
  them

::: notes
Correct the overstatement before it forms, because "the curse of dimensionality"
gets remembered as "high dimensions are bad" and that is not the claim.

Neural networks work happily in thousands of dimensions. So do linear models
with regularisation. What breaks is specifically the family of methods whose
mechanism is a distance, because a quantity that no longer varies can no longer
discriminate. Name the three that appear in this course: k-NN today, k-means in
lesson 8, and the RBF kernel later this afternoon - which is why the RBF SVM
also degraded in that table, though more slowly.

**Now the predictable mistake, and defend the instinct first.** Adding features
because they might help is good practice with a linear model: an irrelevant
feature earns a coefficient near zero and costs you almost nothing. That
reasoning is sound, and it is what they have been taught for three lessons. With
k-NN the same feature costs you a dimension in the distance, and dimensions are
what the method is made of. The habit is right; transferring it one lesson later
does real damage. Say that the fix is to select features BEFORE the distance,
inside the pipeline - which is last week's rule again.
:::

# Notebook 1, live

- k-NN on the pumps, choosing k, and the curse measured

::: notes
Run notebooks/01. Twenty-two minutes.

Have them run the unscaled version first, before the Pipeline appears, so they
watch the score drop and can attribute it to the units rather than to the
method.

The cell to protect if time runs short is the noise-column sweep. Let them add
the columns themselves and watch the score cross the baseline - it changes
behaviour in a way that reading the table does not.

If somebody finishes early, the interesting extension is to ask what happens
with 50 noise columns and 12,000 rows instead of 1,200. The curse is partly a
sample-size problem, and more data does push it back - just far more slowly than
anyone expects.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. The second half is two more model families, so they need to come
back awake.
:::

# Naive Bayes: turn the question around

- We want the probability of the class **given** the readings
- The prior is a count; the denominator is identical across classes and cancels
- Everything hard sits in one term: the probability of **this exact combination
  of readings** among examples of that class

::: notes
Say the strategy in words before the symbols appear. We want a quantity we
cannot estimate directly. Bayes' rule trades it for quantities we might be able
to estimate. That is the whole move.

Walk the right-hand side once. The prior is the class frequency - a count, no
difficulty. The denominator is the same number for every class, so it cannot
change which class wins, and it can be dropped entirely.

That leaves one term, and it carries all the trouble: how likely is THIS
combination of readings, among pumps of this class. With two features that is a
two-dimensional density and we could estimate it from 1,200 rows. With twenty
features it is a twenty-dimensional density, and no quantity of data populates a
twenty-dimensional space.

Point out that this is the curse again, arriving from a completely different
direction - density estimation rather than distance. Handout section 4.1.
:::

# Bayes' rule, applied to a class

$$P(y = c \mid x) = \frac{P(x \mid y = c)\,P(y = c)}{P(x)}$$

::: notes
Left, what we want: the probability of the class given this pump's
readings. Right, what the data can supply: how often pumps of that class
produce readings like these, times how common the class is.

Point at the denominator and say it out loud once: it does not depend on c,
so it is the same number for healthy and for faulty. We are choosing the
larger of two quantities, and dividing both by the same thing changes
nothing. It can be dropped, and scikit-learn drops it.

What remains is the numerator, and only the first factor is hard. Handout
section 3.1 works the same rule through with counts if anyone wants it in
numbers rather than symbols.
:::

# The assumption: independent, given the class

- **Given the class, the features are independent of one another**
- The joint density then factorises into n one-dimensional densities
- Each factor is estimated from the rows of that class alone

::: notes
One assumption, stated in one line, and everything follows from it.

Emphasise the words "given the class". This is not a claim that the features are
independent - it is a claim that whatever dependence they have is entirely
explained by which class they belong to. Those are different claims and the
distinction matters in about ten minutes, when we measure it.

What it buys: one n-dimensional estimation problem becomes n one-dimensional
ones. A one-dimensional density needs very little data. Training becomes a
single pass computing a mean and a variance per feature per class - there is no
iteration, no optimisation, nothing to converge.

Mention that in practice this is computed as a sum of logarithms, for the same
underflow reason as lesson 4's log-likelihood: a product of several hundred
small probabilities is exactly zero in floating point. Handout section 4.2 has
the log form.
:::

# The factorisation the assumption buys

$$P(x \mid y = c) = \prod_{j=1}^{n} P(x_j \mid y = c)$$

::: notes
This is the naive assumption, written down. One n-dimensional density on
the left; n one-dimensional densities on the right.

Why that matters is a counting argument, and it is worth making concrete.
To estimate the left-hand side directly you would need enough pumps to fill
an n-dimensional space - the curse of dimensionality from an hour ago,
arriving in a new disguise. To estimate the right-hand side you need enough
pumps to fit n separate histograms, which is a completely different demand.

That is the whole bargain: an assumption that is usually false, traded for
an estimation problem that is actually solvable. Notebook 02 measures what
the assumption costs when it fails.
:::

# A wonderful bargain, and almost never true

- Training is **a single pass**; adding features costs almost nothing
- But vibration and pressure are both driven by the operating point
- "New" is not independent of "York" given the topic. Symptoms co-occur
- So the question is not whether it holds: it is **when being wrong about it
  costs you nothing**

::: notes
Both halves matter, and students tend to hear only one of them.

The bargain is real. Nothing else in this course trains in one pass, and nothing
else tolerates ten thousand features on a few thousand rows.

The assumption is false almost everywhere, and give concrete cases rather than a
general warning. Two sensors on the same machine are both driven by the
operating point, so of course they move together. In text, "New" and "York"
co-occur far more than independence would allow. In medicine, symptoms of the
same disease arrive in clusters.

Then the framing that makes the rest of this segment worth doing, and put it as
a question to the room: given that the assumption is essentially always false,
why does anybody use this? The answer is that the assumption being false and the
CLASSIFIER being wrong are different things. Argmax only needs the ordering
right, not the probabilities. Handout section 4.2 closes on exactly this.
:::

# On the pumps, it scores 0.933

| Model | Accuracy |
|---|---|
| Majority baseline | 0.613 |
| Logistic regression | 0.613 |
| **Gaussian Naive Bayes** | **0.933** |
| k-NN, k = 5 | 0.944 |

::: notes
A point behind k-NN, and an enormous distance ahead of the linear model - from a
method that assumes something false and trains in a single pass over the data.

Resist the temptation to stop here, because a score alone tells you nothing
about when it will hold. Ask the room the useful question instead: WHY did that
work? If the assumption is false, what happened?

Then be explicit that we are about to check, rather than speculate. This is the
habit worth transferring: when a model does better than you expected, find out
which of your assumptions was accidentally satisfied. That tells you whether the
result will survive on the next dataset.

Handout section 4.3.
:::

# Because here the assumption is true

| Correlation between the two readings | Value |
|---|---|
| overall | −0.046 |
| within healthy pumps | **−0.006** |
| within faulty pumps | **−0.049** |

::: notes
Measured directly, and remember the assumption concerns independence GIVEN THE
CLASS - so the second and third rows are the ones that count, not the first.

Within each class the two readings are essentially uncorrelated: −0.006 among
healthy pumps is as close to zero as a sample of this size will ever produce.
The assumption is not approximately satisfied here; it is satisfied.

Say why that is more useful than the 0.933. A score tells you what happened
once. Knowing WHY it happened tells you when to expect it again - and it gives
them a check they can run in two lines before trusting Naive Bayes on anything.

The obvious caution, since somebody will raise it: zero correlation is not
independence. Correlation only detects linear dependence. For Gaussian Naive
Bayes with Gaussian-ish features it is the right check; in general it is
necessary and not sufficient.
:::

# One step away: when the signal is an interaction

- A second pair of sensors on the same fleet
- Faulty when **exactly one** of the two readings is high
- Both high is the designed high-load mode; both low is idle; one without the
  other is a mismatch between demand and delivery

::: notes
Change one thing about the problem and watch the method fall apart. Describe the
physics before the geometry, because the rule is completely reasonable
engineering.

High demand and high delivery: fine, that is the machine working hard. Low and
low: fine, that is idle. High demand with low delivery, or the reverse: something
is wrong between them. Exactly one high means faulty.

Ask them, before the picture, what each sensor tells you ON ITS OWN under that
rule. Give it a moment. The answer is: nothing at all. Every value of sensor A
appears in both classes about equally often, because whether it means "faulty"
depends entirely on sensor B.

That is what an interaction is, and it is precisely the structure the
independence assumption is unable to represent. Handout section 4.4.
:::

# What Naive Bayes gets to see

![](interaction_marginals.png)

::: notes
Left panel: the two sensors together. Four clean groups, and a perfectly
learnable rule - any method that can draw a non-linear boundary will find it.

Middle and right: each sensor on its own, which is ALL Naive Bayes ever gets.
The two class distributions sit almost exactly on top of one another.

That is the slide in one sentence: the information is real, it is entirely in
the relationship between the columns, and factorising the density throws that
relationship away before the model ever sees it.

Worth saying that no amount of data repairs this. It is not an estimation
problem - with a million rows the marginals still overlap. The model cannot
represent what is being asked of it.
:::

# 0.404: below the baseline, and below chance

| Model | Accuracy |
|---|---|
| Majority baseline | 0.523 |
| **Gaussian Naive Bayes** | **0.404** |
| Logistic regression | 0.393 |
| SVM, linear kernel | 0.606 |
| k-NN, k = 5 | 0.967 |
| SVM, RBF kernel | 0.972 |

::: notes
Two things to draw out, and the second is the more interesting.

First: k-NN and the RBF SVM are at 0.967 and 0.972 on this data. The problem is
not hard. It is only hard for a model that looks at one feature at a time.

Second, and take the time: 0.404 is BELOW CHANCE. Below the majority baseline of
0.523, and below what you would get by flipping a coin. Ask how that is even
possible - a model with no information should surely sit at 50%.

The explanation is worth having in full. The class means on each sensor differ
by about 0.17, against a spread near 1, purely as an artefact of a finite sample.
That accident is the ONLY per-feature evidence available. Naive Bayes has
nothing else to multiply, so it follows it - and in this sample it points the
wrong way.

The sentence to land: a model with no signal does not sit politely at 50%. It
follows whatever spurious structure it can find, with complete confidence.
:::

# Its probabilities are not probabilities

- Mean confidence when **correct: 0.567**. When **wrong: 0.555**
- It cannot tell the difference between the two situations
- With correlated features it fails the other way: 0.999 reported, with an
  accuracy nothing like that
- **The ranking may be useful while the probabilities are not**

::: notes
Two numbers, four thousandths apart. The model is exactly as confident when it
is wrong as when it is right, which means `predict_proba` carries no usable
information about reliability here.

The more common complaint runs in the opposite direction, so give it too: when
features ARE correlated, multiplying their probabilities counts the same
evidence repeatedly. Ten correlated words in a document are treated as ten
independent pieces of evidence, and the posterior saturates. Naive Bayes is
famous for reporting 0.999 on problems it gets wrong a fifth of the time.

Then lesson 5's distinction, which this is the textbook case of: ranking and
calibration are different properties. A model can order examples usefully and
still report numbers that mean nothing.

The practical consequence, and it is concrete: do not use Naive Bayes anywhere a
calibrated probability enters a decision - including lesson 4's cost calculation
in section 7.2, where the threshold depends on the probability being real.
:::

# Notebook 2, live

- Where the assumption holds, and one step away where it does not

::: notes
Run notebooks/02. Eighteen minutes.

The cell to protect is the one that prints 0.404 next to the 0.523 baseline. Let
them see a below-chance score appear from correct code on solvable data before
any explanation arrives.

Have them compute the within-class correlations themselves on both datasets. Two
lines, and it is the diagnostic they should carry: measure the assumption, do
not assume it.

If time allows, the interesting exercise is to hand Naive Bayes the product of
the two sensors as a third column and watch it recover. That is the whole story
of feature engineering versus model choice in one cell.
:::

# The margin: a criterion that is not the error

- On separable data there are **infinitely many** separating lines, and every
  one has **zero training error**
- Minimising the error therefore cannot choose between them
- The SVM picks the widest **slab**: push it out until it touches the nearest
  point of each class
- A boundary passing close to a point is one small perturbation from getting it
  wrong

::: notes
Start with the problem rather than the solution. Draw two well-separated blobs
on the board and ask for the boundary. Somebody will draw a line. Ask whether
another line would have been equally good, and keep asking until the room sees
that there are infinitely many, all with identical training error.

So the error cannot decide. Something else has to. Logistic regression breaks the
tie with log loss - a perfectly respectable answer, and one answer among several.
The SVM breaks it differently: take the boundary with the most room around it.

The intuition, and this is the sentence to say slowly: maximising the distance to
the closest points chooses the boundary that tolerates the most movement in the
data before it changes its mind. That is a statement about GENERALISATION, not
about fit - which is unusual, because almost everything else in this course
optimises fit and controls generalisation indirectly.

Handout section 5.1.
:::

# Three points decide everything

![](svm_margin.png)

::: notes
The solid line is the boundary, the dashed lines are the edges of the slab, and
the circled points are the support vectors touching it.

The number to say out loud: of eighty points, THREE determine the answer. Move
any of the other seventy-seven - anywhere, as far as you like, as long as they
stay outside the slab - and the boundary does not shift by a millimetre.

That is where the name comes from, and it is a genuinely different idea from
anything so far. Every linear model in lessons 3 and 4 used every row: each one
contributed to the gradient in proportion to its error. Here most of the data is
irrelevant once the fit is done.

Two consequences worth naming. The model is small - you store the support
vectors, not the dataset, which is the contrast with k-NN. And it is sensitive
in a specific way: an outlier near the boundary matters enormously, while an
outlier far from it matters not at all.
:::

# Not separable? Charge for the violations

- Our labels are 4% flipped, so the strict problem has **no solution at all**
- Let each point violate the margin by ξᵢ, and charge C for every unit of it
- The constraint says: be on the correct side, or pay

$$\min_{w, b, \xi} \ \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i$$

::: notes
Do not derive this. State what each piece is doing and move on - handout section
5.2 has the derivation, including why maximising the margin is the same as
minimising the norm of w.

The first term wants a wide slab. The second term wants few violations. C sets
the exchange rate between them, and that is the entire content of the expression.

The point worth stressing is why the soft margin is not a patch or a
convenience. With even one mislabelled point inside the other class, the strict
problem is INFEASIBLE - there is no solution, not a bad one. Real data always
contains such points. So the soft margin is the only version anyone ever runs;
the hard-margin problem is a teaching device.

If someone asks about the half in front of the norm: it is there to make the
derivative tidy, exactly as in lesson 3's cost function.
:::

# C is the price of a training error

- **Large C**: violations are expensive, so the model contorts to classify
  everything. Narrow margin, low bias, high variance
- **Small C**: a wider, calmer boundary, at the cost of some errors
- The same dial as k in k-NN and λ in lesson 3, in a third costume
- Note the direction: **large C means less regularisation**

::: notes
Third appearance of the same idea, and say so - that repetition is the point.
Lesson 3 had λ, this morning had k, now C. Every model has one knob that trades
fitting this data against surviving the next, and recognising it in an unfamiliar
model is worth more than memorising three names.

Then the direction, which catches people out constantly and is worth writing on
the board. Large λ means MORE regularisation. Large C means LESS. They run
opposite ways, and scikit-learn's `C` parameter in `LogisticRegression` has the
same inverted convention - it is the reciprocal of the penalty there too.

Ask them which way they would expect C to run before you tell them. Most guess
wrong, and having guessed wrong once is what makes it stick.
:::

# The best straight line is still a straight line

| Model | Accuracy | Support vectors |
|---|---|---|
| SVM, linear kernel | **0.613 ± 0.000** | 947 of 1,200 (**79%**) |
| SVM, RBF kernel | 0.947 ± 0.005 | 278 of 1,200 (23%) |

::: notes
The linear kernel scores 0.613 - the base rate, exactly as logistic regression
did this morning, with the same zero variance across folds.

Make the point explicitly, because it is the one students most often get
backwards: the margin criterion is a better way of CHOOSING among straight
lines. It does not give you anything other than a straight line. When no straight
line works, choosing the best one optimally still gets you nothing. A better
criterion cannot rescue an inadequate hypothesis class.

Then the RBF row, which is the same algorithm with one argument changed, at
0.947 - the best number in the whole lesson, and within a hair of the 0.96
ceiling.

The support-vector column says the same thing in a second language, and that is
the next slide.
:::

# A high support-vector fraction is a free warning

- The linear model needs **79%** of the training set to define its boundary
- The RBF model needs **23%**
- Almost every point sits on or inside the margin: there is no slab that
  separates anything
- The count falls out of `fit` at no cost, and you should look at it

::: notes
Explain what 79% actually means geometrically. A support vector is a point on
the margin or inside it. If 79% of your data is in that position, there is no
region of empty space between the classes at all - the model has been forced to
place a boundary through a crowd.

Contrast with 23%, where the model found genuine room and only the points near
the edge matter.

Give them the diagnostic to keep, since it costs nothing: after fitting an SVM,
look at `len(model.support_)` over the number of training rows. High means the
model is struggling to find room, which usually means the kernel is wrong for
the geometry. It is a warning available BEFORE you cross-validate anything.

Worth adding that it also predicts prediction cost - every support vector is one
kernel evaluation per query, so a model needing 79% of the data has thrown away
most of the speed advantage over k-NN.
:::

# The kernel trick: they needed different coordinates

- Healthy pumps in a disc, faulty ones around it: no line works **in the plane**
- Add a third coordinate: the **distance from the design point**
- Healthy pumps rise a little, faulty ones a lot, and a flat plane separates
  them perfectly
- The classes were always separable. They were measured in the wrong coordinates

::: notes
Build the picture in words first and let them see it before the figure appears.
Take the scatter plot from the start of the lesson, lift each point off the page
by its distance from the centre, and look from the side.

Then the sentence that reframes the whole afternoon: the classes were never
inseparable. Separability is not a property of the data - it is a property of
the data AND the coordinates you happen to have measured it in.

Ask what is unsatisfying about what we just did, and wait for it. We chose that
lift by already knowing the answer. We knew the fault mode was radial. In any
real problem you do not, and inventing the right coordinate is the hard part  - 
which is exactly the difficulty the kernel trick removes.

Handout section 5.4.
:::

# The same pumps, lifted

![](kernel_lift.png)

::: notes
The same 1,200 pumps twice: on the left in the plane where they were measured,
on the right lifted by their distance from the design point. The gold plane does
what no line could.

Have them look at the right panel and confirm for themselves that a flat
horizontal cut separates the colours cleanly. Nothing was added - no new
measurement, no extra sensor. The third coordinate is computed from the two they
already had.

Then set up the objection that the next slide answers. This worked because the
lift was three-dimensional and we could guess it. The spaces that work in general
are enormous - the useful ones are sometimes infinite-dimensional - and
computing the coordinates of every point in such a space is impossible.
:::

# And the map is never computed

- The solution depends on the data **only through inner products** between pairs
  of points
- So we need a function returning the inner product **in the new space**, while
  computing only with the original one
- The radial basis function is scikit-learn's default: one exponential per pair,
  for an **infinite-dimensional** space

::: notes
The trick in one sentence: the SVM never needs the coordinates, only the angles
and lengths between points, and those can be obtained without ever going there.

Do not prove it - handout section 5.4 shows where the inner products come from
in the dual formulation. The claim to state and let land is that a function of
two ordinary two-dimensional vectors returns their inner product in a space of
infinite dimension, at the cost of one exponential.

And close the loop from the previous slide: we chose the lift in the picture by
knowing the answer. The RBF kernel does something equivalent without being told,
which is exactly why it works on problems where nobody could guess the right
coordinates.
:::

# The radial basis function, written out

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

::: notes
One line, and it is the whole of what scikit-learn does when you leave `kernel`
at its default.

Read it aloud as a similarity rather than as a formula: the closer two points
are, the nearer the exponent is to zero and the nearer the kernel is to one; far
apart, it falls away to nothing. That is all an RBF kernel says - how much does
this point look like that one.

Now say what γ means, because the next two slides turn on it: it sets how far a
single training point's influence reaches. Small γ, wide reach, smooth boundary.
Large γ, each point influences only its immediate neighbourhood - and a boundary
that can afford an island around every mislabelled pump.

Worth naming the cost, since this is an engineering course: one exponential per
pair of points, so the kernel matrix is m × m. That is why support vector
machines are a poor fit above a hundred thousand rows, and it is a property of
the method rather than of the implementation.
:::

# The highest training score, the lowest honest one

| Setting | Training | Cross-validated |
|---|---|---|
| γ = 0.1, C = 1 | 0.936 | 0.929 |
| γ = 1, C = 1 | 0.950 | **0.944** |
| γ = 50, C = 1000 | **0.995** | 0.902 |

::: notes
Read the two columns against each other, exactly as lesson 5 taught, and let the
last row do the work.

The bottom row has the best training score on the slide - 0.995, nearly perfect  - 
and the worst honest one. That is the signature of overfitting, and here it is
not inferred from theory but measured on the same data by two different
procedures.

**Now the second predictable mistake, and defend the instinct.** Choosing between
these three on the training score selects the WORST model, with complete
confidence and a very impressive number to report. And the instinct behind it is
not stupid: a higher score on the data you have is, in every other part of
engineering, evidence of a better system. Fitting the data well is what we spent
lessons 3 and 4 doing. What breaks the analogy is that a flexible model can fit
data it has already been shown without learning anything transferable.

The reason this table is on the slide rather than in the handout is that the
next slide shows the same three settings as pictures - so the failure is visible
as well as measurable, which is rare.
:::

# Overfitting you can see

![](svm_gamma_c.png)

::: notes
Three settings, with training and cross-validated scores in each title. Read them
together.

Left, γ = 0.1: a smooth boundary, slightly too smooth, and the two scores agree.
Middle, γ = 1: close to the true envelope, and the best honest score.

Right is the one to sit on. The boundary has broken into bubbles - small islands
around individual points, including the mislabelled ones. That is what a 0.995
training score looks like from the outside: the model has drawn a private
territory around each flipped label so that it can get it right.

Ask them what happens to a new pump landing in the gap between two bubbles. That
is the 0.902.

The reason this slide matters more than the table: they have seen overfitting as
a curve on a plot since lesson 5. Here it is a shape in the input space, and the
shape is obviously wrong in a way no number is. When you can plot the boundary,
plot it.
:::

# Notebook 3, live

- Margins, support vectors, and what γ does to the boundary

::: notes
Run notebooks/03. Eighteen minutes.

The cell to protect is the γ sweep with the boundary drawn at each setting. Let
them turn γ up themselves and watch the bubbles form - it is the most direct
experience of overfitting available in this course.

Have them print the support-vector fraction for the linear and RBF kernels side
by side. Two numbers, and it makes the diagnostic real rather than a claim on a
slide.

If time runs short, skip the C sweep and set it as reading: the table is in
handout section 5.5.
:::

# The three compared, on identical data

| Model | Accuracy |
|---|---|
| Majority baseline | 0.613 |
| Logistic regression (lesson 4) | 0.613 |
| SVM, linear kernel | 0.613 |
| Gaussian Naive Bayes | 0.933 |
| k-NN, k = 5 | 0.944 |
| SVM, RBF kernel | **0.947** |
| *noise ceiling* | *≈ 0.96* |

::: notes
Everything in the lesson on the same 1,200 pumps, cross-validated the same way.

Read the top three rows together: all exactly 0.613, and all three are linear.
Then the bottom three, all near the ceiling, reached by three completely
unrelated routes - remembering the neighbourhood, assuming independence, bending
the space. There is no single idea shared between them.

Then the number that is the lesson: the gap between best and worst is **0.334**.
Larger than any difference this course has shown between a good model and a
carefully tuned one - lesson 5's entire hyperparameter search moved things by a
few hundredths.

Say the conclusion plainly, and it is the practical takeaway of the whole
lesson: choosing the right family matters far more than tuning the wrong one.
Nobody in the top three rows could have tuned their way out. And the way to tell
which family you need is to look at the data first, which is what lesson 2 was
for.
:::

# How to choose, in practice

| Situation | Reach for |
|---|---|
| Few features, plenty of data, odd-shaped boundary | k-NN, or an RBF SVM |
| Very many features, little data | Naive Bayes |
| Need a calibrated probability | **Not** Naive Bayes |
| Model must be small, or fast at prediction | SVM, not k-NN |
| The signal is an interaction between features | k-NN or a kernel |

::: notes
A table to photograph. Work down it and attach today's evidence to each row, so
it is remembered as a set of conclusions rather than as advice.

Row two deserves a sentence on its own, because it is the one that sounds wrong
after the last hour. Text classification is the canonical case: tens of thousands
of word-count features, an assumption that is transparently false, and a method
that works anyway - because being roughly right in ten thousand dimensions beats
being unable to estimate anything at all. Also mention Naive Bayes as a baseline:
it trains in one pass, so if your tuned model does not beat it you have learned
something quickly and cheaply.

One row that did not fit and should be said out loud: when you need to explain
the decision to the person affected, none of today's three will do it. A support
vector machine in a lifted space has no explanation a customer would accept.
Lesson 3's Resources document covers why that is sometimes a legal requirement
rather than a preference.
:::

# What to take away

- **No line separates the pumps**: 0.613, the base rate
- **k is the bias-variance dial**, made visible
- **In 100 dimensions the nearest point is 70% as far as the farthest**
- **Naive Bayes**: 0.933 where independence holds, **0.404** where it does not
- **Choosing the family beats tuning the wrong one**: 0.613 to 0.947

::: notes
Close on the number they opened the second segment with, and say it one more
time: seventy per cent. In a hundred dimensions the nearest point in the entire
dataset is seven tenths of the way to the farthest, and every method built on
distance is operating on a quantity that has stopped varying.

If they remember one habit from today, make it the diagnostic pair: before
reaching for a model, look at the shape of the data; after fitting one, check
whether the assumption it depends on is actually satisfied on your data. Two
lines of code gave us the −0.006 that explained the 0.933.

And the correction to the instinct that this lesson exists to install: more
features is not free. It is nearly free for a linear model and it is expensive
for anything built on a distance.

Handout section 7 lists all of this with the section numbers.
:::

# Homework

- **Exercise 6**, due **Friday 6 November 2026**
- A dataset, three families, and a defence of which one you chose

::: notes
Set it explicitly and say the deadline out loud: next Friday, 6 November.

The marks are for the reasoning, not for the accuracy. A well-argued choice that
scores slightly worse beats a lucky winner with no justification - and the
argument has to reference the data, in the way that the −0.006 justified Naive
Bayes on the pumps this afternoon.

Remind them that everything must be cross-validated and reported with a spread,
in the format lesson 5 set. Scaling goes inside the pipeline; that will be
checked.

Next week: decision trees and ensembles. A fourth family, with a different
answer again to the same question - how do you draw a boundary that is not a
line.
:::

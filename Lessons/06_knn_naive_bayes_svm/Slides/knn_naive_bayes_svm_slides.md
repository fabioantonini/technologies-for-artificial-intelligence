---
title: "Lesson 6: k-NN, Naive Bayes and Support Vector Machines"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "13 November 2026"
---

# Before we start

- Exercise 4: which threshold did you choose for the classifier, and how
  did you defend it?
- Exercise 5, set at the end of lesson 5, is discussed next Friday,
  20 November

::: notes
Nothing is collected. Only exercise 4 today: it was set on 6 November.
Exercise 5 was set earlier today, when lesson 5 finished, so it gets its own
week and is discussed on 20 November together with exercise 6.

Ask two people to defend their threshold out loud - that is the rehearsal for
the exam - and name the recurring mistake if it appeared: choosing the
threshold by sweeping it on the test set.

Then the sentence that links last week to today: everything from now on is evaluated with the tools built in lesson 5:
cross-validation, baselines, training scores read against validation scores.
Today adds three families of models, and each of them can produce a result that
looks right and is not, in a way of its own.
:::

# Today: the first lesson that offers a choice

- **k-nearest neighbours (k-NN)**, and the **curse of dimensionality**
  that limits it
- **Naive Bayes**: a probability model, very cheap to train
- **Support vector machines (SVM)**: the widest margin, then kernels
- How to choose a model family, and why that matters more than tuning

::: notes
The agenda, and one sentence about the whole lesson that is worth saying slowly.

Lessons 3 and 4 built linear models and spent most of their effort on fitting
them honestly. Lesson 5 built the apparatus for telling whether a model works.
None of that told them WHICH KIND OF MODEL TO USE. This is the first lesson that
offers a choice, and learning how to choose is what the lesson is about.

Stress that these are not three ways of doing the same thing. They attack
classification from three unrelated directions: k-NN makes no assumptions and does no fitting at all; Naive Bayes makes an
assumption that is usually false and works well anyway; the SVM changes the
criterion for choosing a boundary, and then changes the coordinates the data is
described in. The differences
between them are the point, not the accuracy figures.

Handout section 1 sets this out in a paragraph each.
:::

# 1,200 pumps, two readings each

- 1,200 simulated pumps, two sensor readings each: vibration in hertz (Hz),
  pressure in bar
- A pump is faulty when its two readings, taken together, are too far from
  the **design point, 42 Hz and 5.6 bar**: too low as readily as too high
- 735 of the 1,200 are faulty: **61.3%**
- 4% of the labels are flipped on purpose, as sensor errors: no model can
  score above about **0.96** (the noise ceiling)

::: notes
Describe the physical fact first, before showing the picture, and let them
predict what it implies. A pump has a design operating envelope. It can fail by
running too slow just as easily as by running too hot.

"Taken together" is the point: each reading is measured against its own scale,
3.5 Hz for vibration and 0.45 bar for pressure, and the pump is faulty when the
combined distance from the design point is more than 1. That boundary is an
ellipse, not two separate ranges, so a pump with each reading inside its own
range can still be faulty, in the corners.

Draw it on the board before you advance: vibration on the horizontal axis,
pressure on the vertical, the design point at 42 Hz and 5.6 bar, an ellipse
around it 3.5 Hz wide on each side and 0.45 bar high. Then ask the room what that
does to the shape of the two classes. Somebody will get it: healthy in the
middle, faulty all around.

The 4% flipped labels play the part of lesson 5's irreducible noise: a model
that learned the true rule perfectly would still get those pumps wrong. So no
model here can exceed roughly **0.96**, and that is the number every
score today should be compared against - not against 1.000, and not against
each other in isolation.

Handout section 1.1.
:::

# The shape of the problem: healthy pumps in the middle, faulty all around

![](pump_scatter.png)

::: notes
Give them a moment before saying anything. Then ask for a straight line that
separates these two classes, and let them try.

The healthy pumps form a disc around the design point; the faulty ones form a ring around them. There is no line. Not a badly chosen line - no line at all,
because the class you want to isolate is completely surrounded.

Then say what is unusual about this picture. The usual textbook picture of
classification is two clouds of points side by side, one class on the left and
one on the right, with a line between them. Here one class surrounds the other.
That shape was not chosen to make the problem hard: it follows from how a pump
fails, by running too low as well as too high. Every problem of the form "inside
the tolerance is good, outside is bad" has it - a machined shaft whose diameter
must be 20 mm plus or minus 0.1 mm, a supply voltage, a body temperature - and
pass/fail inspection of this kind is a large share of classification in
industry.

The scattered wrong-coloured points inside each region are the 4% flipped
labels. Point at two of them now - they come back when k-NN uses a single neighbour,
k = 1.
:::

# A straight boundary calls all 1,200 pumps faulty

| Model | Accuracy, 5-fold cross-validation (mean ± standard deviation over the folds) |
|---|---|
| Always predict the majority class: 735 ÷ 1,200 | 0.613 |
| Logistic regression (lesson 4) | **0.613 ± 0.000** |

::: notes
Read the two rows and let the equality sit there.

The thing to say clearly, because it is easy to under-read: logistic regression
has not been narrowly beaten by the baseline. It has learned NOTHING AT ALL. It
predicts "faulty" for every pump, because with a straight boundary that is
genuinely the best answer available to it.

Then point at the standard deviation, which is the clearest sign: **± 0.000**.
Why exactly zero: the folds are stratified, so each holds 240 pumps of which 147
are faulty, and a model that calls every pump faulty scores 147 ÷ 240 = 0.6125 on
every one of the five. It is not confident either: its probability of "faulty"
runs only from 0.58 to 0.64 across the 1,200 pumps, close to the base rate for
every pump - it does not tell pumps apart (notebook 1 prints both). Give them
the diagnostic to keep - a suspiciously stable cross-validation score often
means a model that is not using its input.

Connect back to lesson 5's habit: quote the baseline first, always. Without the 0.613 in the row above, the model's 0.613 would look like a
mediocre result rather than what it is: no learning at all.
:::

# k-nearest neighbours: the whole algorithm

- A line fails, so try a method that assumes no shape: to classify a new
  pump, take the majority vote of the **k closest** training pumps
- There is no training step: "fitting" only **stores the data** (a **lazy
  learner**: the work waits until a prediction is asked for)
- In code: `KNeighborsClassifier(n_neighbors=5)`

::: notes
A historical note: the method was described by Evelyn Fix and Joseph Hodges in
a 1951 report for the US Air Force, and its error was analysed by Thomas Cover
and Peter Hart in 1967.

The whole method is one sentence, and it is worth pausing on how strange that
is after three weeks of optimisation. There is no cost function, no gradient, no
parameters. Nothing is estimated.

Ask the room what could possibly go wrong with a method that makes no
assumptions. Slides 16 to 23 answer it: a method with no assumptions
has no way to ignore a useless column, and that turns out to be expensive.

The sentence hides two decisions - what "closest" means, and how many
neighbours k to use - and both change the result. Take them in that order.

Handout section 2.1.
:::

# k-NN, drawn: one new pump and its five nearest

![](knn_one_vote.png)

::: notes
The four steps of the previous slide, done once, on real data: the 900
stored pumps of notebook 1 (the other 300 are held out), standardised by a
scaler fitted on those 900, and a new pump at 46 Hz and 5.8 bar - the gold star.
Rust is faulty and teal is healthy, here and on every figure today.

Left: every training pump, and the box that the right panel enlarges. Right:
the five closest, circled in their class colour; the dotted circle reaches the
fifth. Four are faulty and one is healthy, so the vote says faulty.

Point at the healthy one. It is one of the 4% flipped labels: by the rule that
generated the data it is faulty, 46.6 Hz and 5.83 bar, well outside the envelope
(notebook 1 prints each neighbour's label against the rule). The star itself is
truly faulty too. A single dissenting neighbour does not change the verdict, and
that is how a vote over several neighbours outvotes the flipped labels. The
first step, standardising, we come back to in two slides.
Notebook 1, section 2.
:::

# Euclidean distance between two pumps x and x′, over their n readings

$$d(x, x') = \sqrt{\sum_{j=1}^{n} (x_j - x'_j)^2}$$

::: notes
"Closest" means the smallest Euclidean distance, almost always. x and x′ are two
pumps - here the new pump and one stored pump - and n is the number of readings
each carries, two for now. For each reading take the difference, square it, add
the squares over the n readings, take the square root. scikit-learn's default,
metric="minkowski" with p=2, is exactly this. Do not dwell on the formula; dwell
on what it implies, which is the next slide.

The observation that matters: every feature contributes to that sum through its
own units. There is nothing in the expression that knows Hz from bar. Distance is the only way the data reaches the method: there are no
coefficients, no weights, nothing that could say one column matters more than
another.

Say that plainly: for k-NN, the distance IS the model. Anything that distorts
the distance distorts everything.

Handout section 2.1 has it with the scaling argument attached.
:::

# Scale first, or vibration decides everything

- Vibration spans **18 Hz** (33 to 51), pressure **2.3 bar** (4.5 to 6.8)
- Unscaled, 10 Hz apart is distance 10 and 1 bar apart is distance 1:
  vibration picks the neighbours
- The fix (lesson 2): **standardise** each column: subtract its mean,
  divide by its standard deviation
- In code: `make_pipeline(StandardScaler(), KNeighborsClassifier(5))`

::: notes
In lesson 2 scaling looked like good housekeeping; here it is the difference
between a working model and a broken one.

Work the arithmetic out loud. Two pumps differing by 10 Hz and 0 bar are 10
apart. Two differing by 0 Hz and 1 bar are 1 apart. So the algorithm considers
the second pair ten times more similar - for no reason other than the units
somebody chose when the sensors were installed.

The consequence: without standardisation, vibration alone decides which pumps are neighbours,
and pressure has almost no effect. Change the pressure sensor to millibar and
the model's answers change. That should feel unacceptable, and it is.

Practical instruction: k-NN goes in a Pipeline with StandardScaler, always, and the scaler is fitted on the training part of each fold only, as lesson 5
required. Which also says what fit() does for this pipeline: the scaler computes
each column's mean and standard deviation - the only numbers estimated - and
k-NN stores the standardised pumps.
:::

# k sets the balance between bias and variance

- **Small k**: the vote follows each stored pump, those with flipped labels
  included. Low bias, high variance (lesson 5)
- **Large k**: the vote reaches pumps across the edge of the envelope, and
  near the edge the larger class wins. High bias, low variance
- At k = 1 the training accuracy is **exactly 1.000**: each stored pump is
  its own nearest neighbour

::: notes
Lesson 5 split the error into bias and variance, and had to train the model on
300 different training sets to show it. Here you watch the same trade-off by turning one integer,
and that is why this slide is worth more time than its size suggests.

The k = 1 claim is the one to make them prove to themselves. Ask why the
training accuracy is exactly 1.000 at k = 1, and wait. The answer is that every
stored pump is its own nearest neighbour, at distance zero. So it votes for
itself and wins - provided no two stored pumps have identical readings and
different labels, which with continuous readings does not happen.

What "high variance" means here: draw another set of pumps from the same
machines and the prediction at a given point changes a lot, because it rests on
one or two pumps, and the flipped labels are what makes it move. At the other
end, with k equal to every stored pump, every pump gets the majority class: the
0.613 baseline.

That is the purest illustration in the whole course of lesson 5's point: a training score can be perfect and say nothing about how the model does on new
data. Here the perfect score is guaranteed by how k = 1 works, whatever the data.

Handout section 2.2.
:::

# Training against cross-validated accuracy, by k

| k | Training accuracy (on each fold's 960 stored pumps, mean over 5 folds) | Cross-validated accuracy, 5 folds |
|---|---|---|
| 1 | **1.000** | 0.912 |
| 5 | 0.954 | **0.944** |
| 15 | 0.944 | 0.938 |
| 51 | 0.941 | 0.933 |
| 401 | 0.713 | 0.708 |

::: notes
Read the first row, then the last, then the middle.

What the two columns are. Cross-validation: five stratified folds of 240 pumps;
in each round the scaler and k-NN are fitted on the other 960 and score the 240
held out, so every pump is validated exactly once, and the number is the mean of
the five scores. Training accuracy: in the same rounds, the model scores the 960
pumps it stores - each of them is among its own k nearest, which is why the
training score is optimistic for every k, not only for k = 1.

Row one: training 1.000, cross-validated 0.912. The difference between them is
lesson 5's warning about training scores, in a single row.

The best value is k = 5, at 0.944 - wide enough to average out the flipped
labels, narrow enough to still follow the boundary. Note that it is not
dramatically better than 15 or 51: the choice of k is forgiving over a wide
range, which is worth saying because otherwise they will spend a long search on a
choice that hardly matters.

The last row is the interesting failure. At k = 401 the training score has
collapsed too, to 0.713 against 0.708, which distinguishes it from overfitting:
this model is bad everywhere, not just on data it has not seen. That is what high
bias looks like: a model too simple to follow the data.

Compare 0.944 against the ceiling of 0.96, not against 1.000.
:::

# Accuracy against k, between the baseline and the noise ceiling

![](knn_choosing_k.png)

::: notes
Point at the left edge first. The training curve starts at exactly 1.000 and
falls; that is the k = 1 identity, drawn. The k axis is logarithmic: 1, 3, 5, 9,
15, 25, 51, 101, 201, 401. And the k = 1 training point sits above the 0.96
ceiling: a score above the ceiling can only come from memorising the flipped
labels.

Then the two horizontal lines, which are the honest frame for reading any score
in this lesson: the noise ceiling at 0.96, which nothing can exceed, and the
majority baseline at 0.613, which everything should.

A large gap between the two curves is a symptom of high variance; watch it
close as k grows. Where the curves meet and then fall together, the variance is
gone and bias has taken over: the model has become too smooth. The maximum of the cross-validated curve sits just before that.

Ask them which end of this plot they would have picked by looking at the
training curve alone. The answer is the far left, k = 1, one of the worst models on the plot by its
cross-validated score.
:::

# The boundary k-NN draws at k = 1, 15 and 401

![](knn_boundaries.png)

::: notes
Three panels, and each says something different.

k = 1: the boundary is ragged, with islands around individual points, both
ways: teal patches in the faulty ring and rust holes in the healthy region.
Those islands are the 4% with flipped labels - the model has carved out a small
territory for each one. That is variance made visible.

k = 15: a clean ellipse, close to the envelope that actually generated the
data, 42 ± 3.5 Hz by 5.6 ± 0.45 bar.

k = 401 is the one worth the time, because it does not do what people expect.
It has not collapsed to a single class. The boundary has INFLATED past the true
envelope and now swallows faulty pumps. Why it grows rather than shrinks: with 401 votes, a pump just outside the
envelope counts nearly the whole healthy disc among its neighbours - the healthy
pumps are packed tightly in the middle - while the faulty pumps are spread thinly
round the ring. So the healthy votes win outside the envelope too. The model no
longer follows the boundary; it averages the votes across it.

Handout section 2.2 carries the same three panels.
:::

# What k-NN costs

- Storing the pumps costs nothing; the cost comes at **every prediction**
  instead
- A direct implementation compares each new pump with **every** stored
  pump: about m × n operations for m pumps and n features
- To predict, the model needs every stored pump: **deploying the model
  means handing over the data**

::: notes
The first cost is engineering. For 1,200 pumps it is nothing. For ten million
rows answering a thousand queries a second it is the entire problem, and it is why there is a whole family of software for finding approximate nearest
neighbours quickly. Mention that scikit-learn by default uses a k-d tree or a
ball tree (data structures that avoid comparing the query with every row), which helps in low dimensions
and stops helping in high ones - for reasons that are the next segment.

The second cost is easy to miss and matters more often. Every other model in
this course compresses its training data into parameters and then throws the
data away. k-NN cannot: to make a prediction it must hold the training set.

Spell out what that means when the training set is medical records or customer
transactions. Shipping the model means shipping the data - to a phone, to a
customer's server, to a third party. That is a legal question before it is a
technical one. This lesson's Resources document and lesson 2's cover it. The
flip side, worth one sentence: erasing one person from a k-NN model is
immediate - delete the row - where a fitted model would need retraining.
:::

# An experiment: add columns of pure noise

- Keep the 1,200 pumps and their two real readings **unchanged**
- Add columns of **pure noise**: normal random numbers (mean 0, standard
  deviation 1), unrelated to the label
- Only the number of columns changes: the problem is as solvable as before
- Same model each time: standardised k-NN, k = 15, 5-fold cross-validation

::: notes
Set the experiment up carefully before showing the result, because the whole
force of it depends on them believing that nothing was taken away.

The model is k-NN with k = 15, not the best k = 5: on the two real columns
it scores 0.938 against 0.944 (slide 12), almost the same, and a vote over 15
pumps is less thrown by one bad neighbour than a vote over 5 - so if accuracy
falls, the noise columns did it, not a small k.

After standardising, each noise column weighs in the distance exactly as much
as each real reading: the "same weight to every column" of slide 10.

Say it twice if necessary: the signal is still there. Both original columns are
present, unmodified, and still enough to solve the problem as well as before,
up to the 0.96 ceiling. We are
adding columns that contain nothing - no relationship to the label, no
relationship to each other.

Ask for a prediction before you advance. Most of the room will say performance
degrades gently, or not at all, because the useful information is intact. That
instinct is exactly right for a linear model and exactly wrong here, and the
result, after the drawing on the next slide, is why.

Handout section 3.1.
:::

# The experiment, drawn

![](noise_experiment.png)

::: notes
The previous slide's experiment as a picture, before its result. Left: the
table k-NN receives with 10 noise columns - the first 14 of the 1,200 pumps,
every column standardised, the colour giving the value. The two columns left of
the black line are the real readings, the same in every run; everything to the
right is random numbers.

The thing to make the room see: after standardising, the two real columns look
exactly like the noise ones - the same colours, the same range. Nothing in this
table tells k-NN which columns matter.

Right: the five steps, repeated with 0 to 100 noise columns. Only step 2
changes between runs. Ask for the prediction now, before the next slide shows
the result. Notebook 1, section 4.
:::

# The two real readings are still there, and accuracy still falls

| Columns: 2 real + noise | Cross-validated accuracy | Accuracy − baseline 0.613 |
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

Why below the baseline, and not just down to it: the 15 "nearest" pumps are by
now close to a random draw, and a vote among random pumps still says "healthy"
some of the time, for no reason - always saying "faulty" never makes that
mistake. Handout section 3.1 checks the number.

The reason is on the next slides, and it is a property of distances in many
dimensions, not of this dataset.
:::

# Why: distances stop varying

- A separate experiment, no pumps: 1,000 random points with n columns,
  each uniform on [0, 1] (n = 2 a square, n = 3 a cube)
- For a new point, its **nearest** and **farthest** distance: in two
  dimensions they differ a lot
- Each extra column adds a squared difference to every distance: over
  many columns **all distances come out about the same**

::: notes
Say first that this is a different experiment: no pumps, no labels, no model,
only distances between random points. It measures why the pumps' accuracy fell.
The cube has no real and no noise columns - every coordinate is random - so
what it shows holds for any data.

Give the picture in words before showing it - the figure is on the next slide,
and it lands better once they have pictured it themselves.

In two dimensions, the nearest point is genuinely near and the farthest is
genuinely far. That difference is the entire basis on which k-NN operates: it
assumes that "the five nearest" is a meaningfully different set from "five
chosen at random".

Now add dimensions. Every new coordinate contributes its own squared difference
to every pairwise distance. Those contributions are independent draws from the
same distribution, so by the law of large numbers their sum grows while its spread, relative to its
size, shrinks: every distance ends up close to the same value.

The sentence to say, because it is the intuition that sticks: to be near in 100
dimensions a point must be near in all 100 coordinates at once, and almost no
point is. Every pair is near in some coordinates and far in others, and all
pairs end up equally mixed.

The name is Richard Bellman's, from his 1957 book on dynamic programming,
where adding variables made his computations explode.

Say explicitly that this has nothing to do with k-NN, or with machine learning
at all. It is a fact about high-dimensional Euclidean space. k-NN is simply the
first method we have met that depends on it.

Handout section 3.2 has the derivation.
:::

# With n random columns, the nearest point stops being near

![](distance_concentration.png)

::: notes
The ratio of the nearest distance to the farthest, plotted against n, the
number of random columns; each point is the mean over 60 new points, each
measured against the same 1,000. It climbs towards 1, and 1 is the point at
which every point is the same distance from every other point.

The shape is the thing to notice: the damage is done early. Most of the rise
happens in the first fifty dimensions, well below anything anyone would call
high-dimensional. By the time you are worrying about a thousand features, most of the damage has
already happened.

Connect it back to the pumps so nobody treats this as a separate curiosity.
This is the same effect, measured on random points: at 52 columns, "nearest" has
already stopped meaning much, and that is what took k-NN below the baseline.
:::

# Nearest divided by farthest

| n, random columns | nearest distance ÷ farthest distance, in n dimensions, mean over 60 new points |
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

- In **two** dimensions the ratio is 0.016: the nearest point is at **2%**
  of the farthest distance
- In **one hundred** it is 0.701, **70%**: the nearest point is barely
  nearer than a random one
- Back to the pumps: the 15 nearest are almost 15 pumps at random, so with
  52 columns accuracy fell below the baseline

::: notes
This is the number to carry out of this part, so slow down and say it twice.

Seventy per cent. The closest point in your entire dataset is seven tenths of
the way to the furthest one. There is almost nothing left of the idea of a
neighbourhood.

And then the sentence that makes it operational: if the nearest 15 pumps are
barely nearer than 15 chosen at random, a majority vote among them is barely
different from a vote among 15 random labels - and a random vote scores below
the base rate, about 0.57, which is what the 52- and 102-column rows showed.

Ask them to write the number down. Derivations fade; this one tends not to. It
comes back at the close.

Handout section 3.2.
:::

# What the curse is, and is not

- **Not** that many columns cannot be learned: a linear model ignores a
  useless one
- It is that **distance-based methods stop working**: all distances come
  out alike
- For k-NN: choose the columns first, inside the pipeline
- In code: `SelectKBest(mutual_info_classif, k=2)` before k-NN: back
  to **0.938**

::: notes
Correct the overstatement before it forms, because "the curse of dimensionality"
gets remembered as "high dimensions are bad" and that is not the claim.

Neural networks work happily in thousands of dimensions. So do linear models
with regularisation. What breaks is specifically the family of methods whose
mechanism is a distance, because a quantity that no longer varies can no longer
discriminate. Name the three that appear in this course: k-NN today, k-means in
lesson 8, and the RBF kernel later today, which measures similarity through the
same Euclidean distance.

**Now the predictable mistake, and defend the instinct first.** Adding features
because they might help is good practice with a linear model: an irrelevant
feature earns a coefficient near zero and costs you almost nothing. That
reasoning is sound, and it is what they have been taught for three lessons. With
k-NN the same feature costs you a dimension in the distance, and dimensions are
what the method is made of. The habit is right; transferring it one lesson later
does real damage. Say that the fix is to choose the features before any distance is computed, as
a step inside the pipeline, so that the choice is made on the training part of
each fold only - lesson 5's rule again.

Then the trap in choosing them, measured in notebook 1 on 52 columns (50 noise),
k = 15, the same folds. With no selection, 0.602. The obvious tool,
SelectKBest(f_classif, k=2), scores each column by how far apart the class means
are - and on the ring healthy and faulty pumps have almost the same mean
vibration and pressure, so it keeps two noise columns and scores 0.562, worse
than doing nothing. mutual_info_classif detects any dependence between a column
and the label, not only a shift of the mean: it keeps vibration and pressure and
scores 0.938, the score without noise.

One condition on the linear model's "coefficient near zero": it needs many more
examples than columns, as here, 1,200 against 102. With few examples even a
linear model overfits noise columns, which is why lesson 3 regularised.
:::

# Notebook 1, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| the straight line, 4-6 | `LogisticRegression`, `cross_val_score`, `StratifiedKFold` → 0.613; `cross_val_predict` → all faulty |
| k-NN, scaled first, 7-10 | `make_pipeline(StandardScaler(), KNeighborsClassifier(k))` |
| choosing k, 11-14 | `cross_validate` with training scores → 0.944 at k = 5 |
| columns of noise, 16-18 | k = 15 on `load_with_noise(50)` → 0.602 |
| distances stop varying, 19-22 | NumPy: nearest ÷ farthest → 0.701 at n = 100 |
| choosing the columns, 23 | `SelectKBest`: `f_classif` 0.562, `mutual_info_classif` 0.938 |

::: notes
k-NN on the pumps, choosing k, and the curse of dimensionality measured. Run
notebook 01. Twenty-two minutes.

Have them run the unscaled version first, before the pipeline appears, so they
watch the score drop and can attribute it to the units rather than to the
method.

If time runs short, do not skip the noise-column sweep. Let them add the columns
themselves and watch the score cross the baseline - seeing it happen convinces
more than reading the table.

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

# Naive Bayes: ask how likely the readings are in each class

- A second family: we want P(y = c | x), the probability of class c
  **given** the readings x
- Bayes' rule uses how common each class is, and which readings are
  common in each (healthy near 42 Hz and 5.6 bar, faulty all around)
- The hard part: how often a class shows **this exact combination** of
  readings - with many readings, almost no pump of that class is near x

::: notes
The rule is Thomas Bayes's, an English minister; his essay was published in
1763, after his death, by his friend Richard Price.

Say the strategy in words before the symbols appear. We want a quantity we
cannot estimate directly. Bayes' rule trades it for quantities we might be able
to estimate. That exchange is the whole idea: the question we want answered is
"given these readings, which class?"; the rule swaps it for the reverse
question, "given the class, how likely are these readings?", which the data can
answer - take the faulty pumps alone and look at where their readings fall,
then the healthy ones.

Walk the right-hand side once. The prior is the class frequency - a count, no
difficulty. The denominator is the same number for every class, so it cannot
change which class wins, and it can be dropped entirely.

That leaves one term, and it carries all the trouble: how likely is THIS
combination of readings, among pumps of this class. With two features that is a
two-dimensional density and we could estimate it from 1,200 rows. With twenty
features it is a twenty-dimensional density, and no quantity of data populates a
twenty-dimensional space.

Point out that this is the curse of dimensionality again, from a different
direction: before the break it broke
distances, here it breaks the estimation of a
density. Handout section 4.1.
:::

# Bayes' rule, applied to a class

$$P(y = c \mid x) = \frac{p(x \mid y = c)\,P(y = c)}{p(x)}$$

::: notes
Left, what we want: the probability of the class given this pump's
readings. Right, what the data can supply: how often pumps of that class
produce readings like these, times how common the class is.

Say once why one letter is small. The readings are continuous, so the
probability of one exact value is zero: p(x | y = c) and p(x) are densities,
written with a lowercase p, while P(y = c) and the left-hand side stay
probabilities - the density units cancel between top and bottom. The
consequence they will meet later: a density can exceed 1.

P(y = c | x) is what any probabilistic classifier wants, not only this one. The
difference is the route: logistic regression estimates it directly, one formula
from readings to probability (lesson 4); Naive Bayes reaches it indirectly,
through the reverse probability and the class frequencies.

Point at the denominator and say it out loud once: it does not depend on c,
so it is the same number for healthy and for faulty. We are choosing the
larger of two quantities, and dividing both by the same thing changes
nothing. It can be dropped, and scikit-learn drops it.

What remains is the numerator, and only the first factor is hard. Handout
section 4.1 works the same rule through with the pumps' counts, 465 and 735, if
anyone wants it in numbers rather than symbols.
:::

# Drop the denominator: pick the class with the largest numerator (arg max)

$$\hat{y} = \mathrm{arg\,max}_c \; p(x \mid y = c)\,P(y = c)$$

::: notes
This is the classifier, before any assumption: score each class by the
numerator and pick the largest. The denominator has gone because it is the
same for every class.

Read "arg max" out, because it appears here for the first time: arg max over c
is the class c that makes the score largest - the class, not the score.
np.argmax in code. And the score is not a probability: it is a density times a
probability.

Expect someone to object that the number is then no longer a probability. The
instinct is sound - a term has been thrown away - and the answer is that it is
recoverable. The denominator is the sum of the numerators over the classes, so
dividing each score by the sum of the scores gives the probability back
exactly. That is what predict_proba does, in logarithms. Dropping it does not make the probabilities less reliable. Where Naive Bayes's
probabilities do go wrong is inside the numerator, because of the assumption,
and slide 45 shows it.

Then point at the first factor: this is where all the difficulty now sits, and
the next slide is the assumption that makes it estimable. Handout section 4.1
carries a worked example with the pumps' priors.
:::

# The assumption: independent, given the class

- To make p(x | y = c) estimable, Naive Bayes assumes: **given the class,
  the features are independent of one another**
- Then the density of all n readings together is the product of n
  one-dimensional densities, one per feature
- Each one-dimensional density is estimated from the pumps of that class
  alone. In code: `GaussianNB()`

::: notes
One assumption, stated in one line, and everything follows from it.

Emphasise the words "given the class". This is not a claim that the features are
independent - it is a claim that whatever dependence they have is entirely
explained by which class they belong to. Those are different claims and the
distinction matters on slide 40, when we measure it. An everyday case: ice-cream
sales and drownings rise together, because both rise in summer; within one
season they are unrelated - dependent overall, independent given the season.

Why it helps, on the pumps: instead of counting faulty pumps near 48 Hz AND
5.6 bar together, count faulty pumps near 48 Hz ignoring pressure, then near
5.6 bar ignoring vibration - plenty of pumps for each. The product is the
assumption's consequence; the bell shape of each factor is a second, separate
choice, two slides on.

What it buys: one n-dimensional estimation problem becomes n one-dimensional
ones. A one-dimensional density needs very little data. Training becomes a
single pass computing a mean and a variance per feature per class - there is no
iteration, no optimisation, nothing to converge.

Mention that in practice this is computed as a sum of logarithms, for the same reason lesson 4 used the log-likelihood: a product of several
hundred small densities is too small for the computer to store, and becomes
exactly zero (this is called underflow). Handout section 4.2 has
the log form.
:::

# The assumption, written: a product of n densities

$$p(x \mid y = c) = \prod_{j=1}^{n} p(x_j \mid y = c)$$

::: notes
This is the naive assumption, written down. One n-dimensional density on
the left; n one-dimensional densities on the right.

Why that matters is a counting argument, and it is worth making concrete.
To estimate the left-hand side directly you would need enough pumps to fill
an n-dimensional space - the curse of dimensionality from before the break, in a new form. To estimate the right-hand side you need enough
pumps to fit n separate histograms, which is a completely different demand.

That is the whole bargain: an assumption that is usually false, traded for
an estimation problem that is actually solvable. Notebook 02 measures what
the assumption costs when it fails.
:::

# Gaussian: each factor is a bell curve

$$p(x_j \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}}\exp\left[-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right]$$

::: notes
The factorisation leaves one choice open: what shape each one-dimensional
density has. "Gaussian" answers it. Every feature, in every class, gets its own
bell curve with its own centre mu and its own width sigma.

So say plainly what training is, because students expect an optimisation and
there is none. Training computes three kinds of average: the class priors, and
a mean and a variance per feature per class. One pass over the data. No
gradient descent, no learning rate, no iterations, nothing to converge.

Ask the room how many numbers that is for the pumps - two features, two classes.
Let them count: two priors, four means, four variances. Ten. The next slide
shows all ten. Handout section 4.2.
:::

# Training computes ten numbers: the widths separate the classes

| Fitted on the 1,200 pumps | Healthy | Faulty |
|---|---|---|
| Prior: the class's share of the pumps | 465 ÷ 1,200 = 0.3875 | 735 ÷ 1,200 = 0.6125 |
| Vibration mean | 41.986 Hz | 42.015 Hz |
| Vibration standard deviation | **1.695 Hz** | **4.636 Hz** |
| Pressure mean | 5.617 bar | 5.616 bar |
| Pressure standard deviation | **0.219 bar** | **0.576 bar** |

::: notes
Let them read the means first and find the surprise themselves: the two
classes have the same centre, to a few hundredths of a hertz. What differs is
the width. Healthy pumps are a narrow bell around the design point, faulty ones
a wide bell - the disc and the ring, one feature at a time.

This explains the result they are about to see. A linear model separates
classes by putting a boundary between their means, and here there is nothing
between the means to find, so logistic regression will sit at the baseline.
Naive Bayes compares widths. The same equal means are why f_classif picked two
noise columns on slide 23: it compares class means, and here they do not differ.

One pump makes it concrete. At 48 Hz and 5.6 bar, 6 Hz off the design point,
that is three and a half healthy standard deviations but barely more than one
faulty one. The vibration term decides it: P(faulty) = 0.981. At the design
point itself, 42 Hz and 5.6 bar, P(healthy) = 0.820. The next slide draws it.

If someone spots in the handout that one log-density is positive, that is the
moment to say a density is not a probability: it is a probability per unit of
pressure, and a bell with a standard deviation of 0.219 bar has to peak at about
1.82 for its total area to be 1. Handout section 4.2 works the pump through term by term.
:::

# A pump at 48 Hz, 5.6 bar: there the faulty bell is 86 times higher

![](nb_bells_one_pump.png)

::: notes
The previous slide's numbers, drawn. Left, vibration: both bells are centred on
42 Hz, and the only difference is width. Follow the dashed line at 48 Hz up. The
narrow healthy bell has fallen almost to nothing there - 6 Hz is three and a
half of its standard deviations - while the wide faulty bell, for which 6 Hz is
barely more than one, is still well up: 86 times higher.

Then the other dashed line, at 42 Hz, the design point: there the narrow bell is
on top, and that pump comes out healthy, 0.820.

Right, pressure: 5.6 bar is central for both classes, so this reading barely
discriminates - and the narrow healthy bell peaks at about 1.82, which is why
its log-density comes out positive on the next slide. A density is not a
probability.

The verdict can be read here already, by multiplying three ratios, faulty
over healthy. Vibration: 0.0374 ÷ 0.000433 = 86.4. Pressure: the healthy bell is
the higher, 1.818 ÷ 0.692 = 2.63, so divide by 2.63. Class shares:
0.6125 ÷ 0.3875 = 1.58. Together 86.4 × 1.58 ÷ 2.63 = 52: odds of 52 to 1 for
faulty, so P(faulty) = 52 ÷ 53 = 0.981. Say this even when the log slides are
skipped - it is the answer they lead to.

The next slide puts these heights, as logarithms, in a table.
:::

# Logs turn the product into a sum, and keep the same winner

$$\ln\left[P(y = c) \prod_{j} p(x_j \mid y = c)\right] \\ = \ln P(y = c) + \sum_{j} \ln p(x_j \mid y = c)$$

::: notes
Inside the bracket on the left is the classifier as slides 28 and 30 built it:
prior times one density per reading, the class with the largest product
winning. The equation says its log is a sum, and the title says why that is
allowed: ln is increasing, so the class with the largest product is also the
class with the largest log. The value changes - −8.095 is not 0.000305 - but the
winner does not. The logs are natural logs - base e - which matters two slides
on, when we undo them with e.

Say why nobody computes the left-hand side directly. With hundreds of features
the product is hundreds of small densities multiplied together - a density can
exceed 1, but far from a class's centre it is small - and the product becomes
too small to store. A double cannot hold anything below about 10 to the minus
308: two hundred factors of 0.01 make 10 to the minus 400, stored as exactly
zero (underflow), while the sum of their logs is −921, an ordinary number. Every
class then scores zero and nothing can be compared.

Two facts fix it. The logarithm is increasing, so whichever class has the
larger product also has the larger log: the arg max does not change. And the
log of a product is the sum of the logs. So: add one log per reading to the log
prior, and compare the sums. This is lesson 4's log-likelihood, for the same
reason.

Show it on the 48 Hz pump before the table, so the table is not a surprise.
Healthy: 0.3875 times 0.000433 times 1.818 = 0.000305; in logs, −0.948 − 7.745 +
0.598 = −8.095, and e^(−8.095) is the same 0.000305. Faulty: 0.6125 times 0.0374
times 0.692 = 0.01585; in logs, −4.144. The next slide is exactly those numbers.
Handout section 4.2.
:::

# 48 Hz, 5.6 bar: scores and their logs

| Class | Row | Prior | Vibration density | Pressure density | Product of the three, or sum of their logs |
|---|---|---|---|---|---|
| Healthy | value | 0.3875 | 0.000433 | 1.818 | 0.000305 |
| Healthy | ln | −0.948 | −7.745 | +0.598 | −8.095 |
| Faulty | value | 0.6125 | 0.0374 | 0.692 | 0.01585 |
| Faulty | ln | −0.490 | −3.286 | −0.368 | **−4.144** |

::: notes
This table is the whole of prediction (inference), and it is worth saying how
little computation it needs. Read each class as
two rows. The value row is the product form: the prior from slide 32, and the
height of that class's bell at each reading - the dots on slide 33 - each
density conditional on the class. Multiply across and you get the score. The ln
row is the same numbers logged, and the sum across is the log score. The last
column is the point of the previous slide: e^(−8.095) is 0.000305, so the sum of
logs and the product carry the same information. Compare the totals: that is
the prediction. No
training row is consulted - only the ten stored numbers - which is the exact
opposite of k-NN, where prediction is the whole cost.

Walk the pump across. Vibration decides it: 48 Hz is 6 Hz from the centre, three
and a half standard deviations for the narrow healthy bell, so its log density
is −7.745; for the wide faulty bell it is barely more than one, −3.286.
Pressure is at the centre, so it favours the narrow bell slightly - and its
healthy log density is positive, +0.598, because a density is not a
probability and a narrow bell peaks above 1.

Faulty has the larger total, so faulty is the prediction, and the gap between
the totals, −4.144 − (−8.095) = 3.951, is ln 52: the 52 to 1 of slide 33, in
logs. How the two totals become probabilities is the next slide. Handout
section 4.2 lists the four steps and works one cell out in full.
:::

# From the two totals to probabilities

| Step | Healthy | Faulty |
|---|---|---|
| 1. Log total, previous slide | −8.095 | −4.144 |
| 2. Undo the log: e^total = prior × densities | 0.000305 | 0.01585 |
| 3. Divide each by their sum, 0.000305 + 0.01585 = 0.01616 = p(x) | **0.019** | **0.981** |

::: notes
Three steps, one row each, and both classes side by side so they can see the
two probabilities come out of the same arithmetic.

Row 1 is the table they just read. Row 2 undoes the logarithm: e to the total
gives back the Bayes numerator, prior times the two densities - the product
from two slides ago, 0.3875 times 0.000433 times 1.818 = 0.000305 for healthy,
and 0.01585 for faulty. These are the unnormalised scores of slide 28.

Row 3 is slide 28's normalisation, now with numbers. Why the sum of the two
scores is p(x): a pump is either healthy or faulty, so its readings come from
one of the two - the law of total probability, the step students tend to take on
faith - and p(x) = P(healthy) p(x | healthy) + P(faulty) p(x | faulty), the two
scores added. It is the denominator of slide
27's Bayes' rule, the one we dropped to choose the class. Dividing each score by it gives probabilities that add to
one: 0.000305 / 0.01616 = 0.019 for healthy, 0.01585 / 0.01616 = 0.981 for
faulty. Ask them to check the two add to one before moving on.

0.981 is the model's number, not the truth. For this pump the truth is known:
it is (48 − 42) / 3.5 = 1.71 scales from the design point, so faulty by the
rule, and labels are flipped at random 4% of the time, so the probability of
seeing "faulty" is 0.96. The model is a little more sure than the data allow.

The next slide shows the same numbers computed from the gap alone, and why
that is the form software uses. Handout section 4.2.
:::

# Software uses only the gap between the logs

- Many features: e^total rounds to 0 for both classes, and 0 ÷ 0
- The gap stays ordinary: −4.144 − (−8.095) = 3.951
- Divide top and bottom by e^(faulty total): P(faulty) =
  1 / (1 + e^(healthy total − faulty total))
- P(faulty) = 1 / (1 + e^(−3.951)) = **0.981**: odds of e^3.951 ≈ **52 to 1**

::: notes
Same result as the previous slide, reached without ever computing the two
scores. Start with the reason, the first bullet: with two features the scores
are ordinary numbers and the previous slide works fine. With hundreds of
features the totals reach −800, e^(−800) is exactly zero in floating point,
and the fraction of the previous slide becomes 0/0.

Then where the formula in the third bullet comes from. Divide the numerator
and the denominator of 0.01585 / (0.01585 + 0.000305) by 0.01585: the fraction
is unchanged and becomes 1 / (1 + 0.000305 / 0.01585). The two scores are e to
their totals, and a ratio of exponentials is the exponential of the
difference, e^a / e^b = e^(a - b). So 0.000305 / 0.01585 = e^(−8.095 + 4.144)
= e^(−3.951) = 0.0192. Neither total matters on its own - only how far apart
they are, and that gap is an ordinary number however many features there are.

Then the reading that makes it intuitive. The gap is the log-odds, the logarithm
of how many times more likely faulty is than healthy: e^3.951 is
about 52, so faulty is 52 times as likely as healthy. Odds of 52 to 1 split
the probability 52/53 = 0.981 and 1/53 = 0.019, the way anyone would for a bet.
1 / (1 + e^(−Δ)) is lesson 4's sigmoid - odds over odds plus one, rewritten.
Handout section 4.2, "From log scores to a probability".
:::

# A cheap model, built on an assumption that is almost never true

- The gain: training is **one pass** over the data; extra features cost
  almost nothing
- The assumption rarely holds: on a real pump, vibration and pressure both
  depend on how hard it runs
- In a text, "New" and "York" appear together far more than independence
  allows
- So the useful question: **when does being wrong about it cost nothing?**

::: notes
Both halves matter, and students tend to hear only one of them.

The bargain is real. Nothing else in this course trains in one pass, and few methods
tolerate ten thousand features on a few thousand rows as cheaply. Training is
priors, means and standard deviations - counts, sums and sums of squares,
accumulated row by row - and an extra feature adds two numbers per class, its
mean and standard deviation, with nothing linking it to the others.

The assumption is false almost everywhere, and give concrete cases rather than a
general warning. Two sensors on the same machine are both driven by the
operating point, so of course they move together. In text, "New" and "York"
co-occur far more than independence would allow. In medicine, symptoms of the
same disease arrive in clusters.

Then the framing that makes the rest of this segment worth doing, and put it as
a question to the room: given that the assumption is essentially always false,
why does anybody use this? The answer is that the assumption being false and the
CLASSIFIER being wrong are different things. To choose a class, the model only needs the right class to get the highest
score, not the probabilities to be right. Handout section 4.2 closes on exactly this.
:::

# Naive Bayes on the pumps: 0.933

| Model | Accuracy, 5-fold cross-validation, same folds | Standard deviation over the 5 folds |
|---|---|---|
| Majority baseline | 0.613 | 0.000 |
| Logistic regression | 0.613 | 0.000 |
| **Gaussian Naive Bayes** | **0.933** | 0.010 |
| k-NN, k = 5 | 0.944 | 0.006 |

::: notes
A point behind k-NN, and an enormous distance ahead of the linear model - from a
method that assumes something false and trains in a single pass over the data.

Read "a point behind" with lesson 5's eyes. 0.011 is about 13 pumps of 1,200, and
the fold standard deviations are 0.010 and 0.006: the gap is about one fold
standard deviation, so call the two comparable rather than ranked. And k = 5 was
chosen as the best k on these same folds, so its 0.944 is slightly optimistic
(lesson 5, nested cross-validation); Naive Bayes had nothing to tune.

Resist the temptation to stop here, because a score alone tells you nothing
about when it will hold. Ask the room the useful question instead: WHY did that
work? If the assumption is false, what happened?

Then be explicit that we are about to check, rather than speculate. This is the
habit worth transferring: when a model does better than you expected, find out
whether it worked because its assumption held or despite it failing. That tells
you whether the result will survive on the next dataset.

Handout section 4.3.
:::

# Uncorrelated within each class, but not independent

| Correlation, within one class | between vibration and pressure | between the two standardised readings, squared (how far from the centre) |
|---|---|---|
| healthy pumps | −0.006 | +0.168 |
| faulty pumps | −0.049 | **−0.420** |

::: notes
Show the middle column first and let the room draw the conclusion everyone
draws: within each class the readings are uncorrelated, so the assumption
holds. It is the natural check, and for bell-shaped data it would settle it.

If someone asks why the square and not the cube - and someone should - the
answer is symmetry. The dependence is about how far from the centre, not which
side, so the transformation must map far-left and far-right alike: an even
function. The square does (−2 and +2 both give 4); the cube is odd, keeps the
sign, and its correlation among the faulty pumps is −0.022, as blind as the raw
readings. The absolute value works too, −0.454. The square is the natural pick
because the disc and the ring are defined by z1² + z2², and it is the coordinate
the kernel lift will use later. Handout section 4.3.

How the right column is computed, in three steps: standardise each reading with
the mean and standard deviation of all 1,200 pumps (42.004 Hz and 3.78 Hz; 5.616
bar and 0.471 bar); square it, which keeps "how far from the centre" and forgets
the side; then the ordinary correlation of the two squared columns, within one
class. One faulty pump, at 35.561 Hz and 5.233 bar: z = (35.561 − 42.004) / 3.78 =
−1.70, squared 2.90; z = (5.233 − 5.616) / 0.471 = −0.81, squared 0.66. Read the
−0.420 by quarters: faulty pumps with vibration near the centre (squared below
0.32) have a mean pressure square of 2.38 - far out, to escape the hole - and
those with vibration far out (above 2.34) have 0.64. One large, the other small:
negative.

The +0.168 of the healthy pumps is not the disc's dependence. It comes from the
28 pumps labelled healthy whose label was flipped - truly faulty, far out in both
readings. Leave them out and the 437 left give −0.036; the faulty −0.420 becomes
−0.460 without its 21. The faulty number carries the argument.

Then show why that conclusion is wrong. Correlation only sees straight-line
dependence. Healthy pumps
fill a disc and faulty ones a ring; on a ring, a pump far out in vibration must
be central in pressure. That dependence is symmetric, so the straight-line
summary averages it to zero. Square each reading's distance from the design
point and it appears: −0.420 among the faulty pumps. Or say it as a spread. Take the faulty pumps whose vibration is most extreme -
the fifth of them furthest from the centre, more than 6.2 Hz away - and their
pressure has a standard deviation of 0.37 bar; for the other faulty pumps it is
0.62 bar. Same average pressure, very different spread, so knowing the vibration
tells you something about the pressure - statistically, not mechanically:
knowing one changes what to expect of the other. Put
the distinction on the board in two lines: uncorrelated, yes; independent, no.
Naive Bayes assumes the second. The assumption is false.

Someone will then ask why we used it at all. Answer it directly: nobody checks
the assumption before fitting. It is a simplification accepted in exchange for a model of ten numbers,
and the model is judged, like any other, by its validation score. That score asks for less than the assumption does. To classify correctly, the
right class only has to get the higher score - the sign of the gap between the
two log scores - and the densities do not have to be right. Naive Bayes can be optimal in that sense with independence badly violated
(Domingos and Pazzani, 1997).

So why 0.933? Because the densities are wrong but the boundary they imply is
right. With both classes centred on the same point, comparing a narrow bell
with a wide one gives an axis-aligned ellipse, and the true envelope is an
axis-aligned ellipse. The fitted model's edge sits at about 3.17 Hz by 0.41
bar against a true 3.5 by 0.45 - the same shape, about a tenth smaller. A false
assumption is cheap when it leaves the boundary the right shape. The next
slide is the case where it does not. Handout section 4.3.
:::

# The densities are wrong, but the boundary they draw is about right

![](nb_boundary_vs_envelope.png)

::: notes
The previous slide's argument, drawn. Dots are the pumps; the solid black
ellipse is where the fitted Naive Bayes model changes its mind; the dashed gold
one is the envelope the data was generated from.

Let them compare the two shapes before saying anything: same orientation, same
proportions, the model's about a tenth smaller - 3.17 Hz by 0.41 bar against 3.5
by 0.45. Why the shape comes out right: the gap between the two log scores is a
sum of terms a_j (x_j − μ_j)² plus a constant, every a_j positive because the
healthy bell is the narrower, so "gap = 0" is an ellipse with axes along the two
readings, centred on the design point. Only its size is off. Densities are not
"calibrated" or not - they are wrong in shape; calibration is about the
probabilities. The assumption is false, the densities it multiplies are wrong, and the
boundary they imply is nearly right. That is why 0.933 sits so close to k-NN's
0.944. Handout section 4.3.
:::

# A second dataset: the signal is an interaction

- 1,200 more simulated pumps, with two other sensors, A and B
- Each sensor is either high (around +1) or low (around −1), half the pumps
  each; faulty when **exactly one** is high - the exclusive or (XOR)
- Both high: high-load mode. Both low: idle. One alone: demand and
  delivery do not match
- So A alone says nothing: what a high A means depends on B (an
  **interaction**)

::: notes
Change one thing about the problem and watch the method fail. Describe the
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

How the data is made: each sensor has a hidden state, high or low, 50/50 and
independent of the other; the reading is +1 or −1 plus noise with a standard
deviation of 0.45. The label comes from the states, not the readings, and no
label is flipped here. 628 of the 1,200 are faulty.
:::

# Each sensor alone: the two classes look the same

![](interaction_marginals.png)

::: notes
Left panel: the two sensors together. Four clean groups, and a perfectly
learnable rule - any method that can draw a non-linear boundary will find it.

Colours first, because someone will ask: teal is healthy, rust is faulty,
everywhere. In the left panel each class owns two opposite quadrants - healthy
has both sensors high or both low, faulty one high and one low.

Middle and right: each sensor on its own, which is ALL Naive Bayes ever gets.
Solid outlines are the data. Each sensor is either low or high, so each outline
has two humps - and healthy and faulty have the SAME two humps, because each
class is about half low and half high on each sensor alone.

The dashed curves are what the model fits: one bell per sensor per class. A
single bell cannot follow two humps, so it sits in the valley between them, and
the two classes' bells almost coincide - centre near 0, standard deviation about
1.1 for both.
Contrast the pumps a few slides ago: bells with a shared centre but widths of 1.695 against 4.636 Hz, and the widths carried the class. Here nothing does.

That is the slide in one sentence: the information is real, it is entirely in
the relationship between the columns, and factorising the density throws that
relationship away before the model ever sees it.

Worth saying that no amount of data repairs this, and that the bell is not the
culprit: even the exact per-sensor densities, the two-hump outlines, are the same
for the two classes, so a better shape per sensor would not help. What fails is
the factorisation. With a million rows the marginals still overlap. The model
cannot represent what is being asked of it.
:::

# 0.404: below the baseline, and below chance

| Model | Accuracy, 5-fold cross-validation |
|---|---|
| Majority baseline: 628 faulty ÷ 1,200 | 0.523 |
| **Gaussian Naive Bayes** | **0.404** |
| Logistic regression | 0.393 |
| SVM, linear kernel | 0.606 |
| k-NN, k = 5 | 0.967 |
| SVM, radial basis function (RBF) kernel | 0.972 |

::: notes
Two things to draw out, and the second is the more interesting.

First: k-NN and the RBF SVM are at 0.967 and 0.972 on this data. The problem is
not hard. It is only hard for a model that looks at one feature at a time.

Second, and take the time: 0.404 is BELOW CHANCE. Below the majority baseline of
0.523, and below what you would get by flipping a coin. Ask how that is even
possible - a model with no information should surely sit at 50%.

The explanation is worth having in full. The class means differ by 0.18 on sensor
A and 0.14 on sensor B, against a spread near 1, only by chance, because the sample is finite.
That accident is the ONLY per-feature evidence available. Naive Bayes has
nothing else to multiply, so it follows it - and in this sample it points the
wrong way. What that looks like, quadrant by quadrant (notebook 2): the
boundary is close to a line. Of the 369 pumps with A high and B low it gets 359
right - every faulty one; of the 264 with A low and B high it gets 9 right -
every faulty one missed; and it calls 239 of the 282 pumps with both high, and
213 of the 285 with both low, faulty, when nearly all are healthy. 42 + 359 + 9 +
75 = 485 right of 1,200: 0.404.

The sentence to land: a model with no signal does not sit politely at 50%. It
follows whatever accidental structure it can find in the sample. It is not even
confident about it, which is the next slide.
:::

# Its probabilities cannot be trusted

- Interacting sensors: Naive Bayes fitted on 900 pumps, tested on the
  other 300
- For each pump it gives P(faulty) and predicts the larger class.
  Confidence = the probability of the class it predicts, from 0.5 (a coin)
  to 1 (certain)
- Mean confidence **0.567** when right, **0.555** when wrong: it cannot
  tell when it is wrong
- It says about 56% sure and is right on 133 of 300, 44%: its
  **probabilities are not calibrated**

::: notes
Two numbers, twelve thousandths apart. The model is exactly as confident when it
is wrong as when it is right, which means `predict_proba` carries no usable
information about reliability here. A useful model is surer when it is right,
so you can trust the sure answers and check the unsure ones; this one gives you
nothing to sort by. And the last bullet is what "not calibrated" means, in
numbers: a calibrated model that says 56% is right about 56% of the time. It may
still rank pumps usefully (lesson 4), but its probabilities mean nothing.

The more common complaint runs in the opposite direction - features that copy
each other push it towards 0 or 1 - so give it too: when
features carry the SAME evidence about the class - near-copies of one another -
multiplying their probabilities counts it repeatedly. Be precise here, because
they have just seen two kinds of dependence that are not this: the pumps' ring
and the XOR sensors. Ten correlated words in a document are treated as ten
independent pieces of evidence, and the probability is pushed all the way to 0
or 1. Neither dataset today shows this; handout section 4.5 describes the
four-line experiment that does (ten near-copies of one column).

Then lesson 4's distinction (its slide 49), which this is the textbook case of:
ranking and calibration - whether a reported 0.7 comes true 70% of the time - are
different properties. A model can order examples usefully and
still report numbers that mean nothing.

The practical consequence, and it is concrete: do not use Naive Bayes anywhere a
calibrated probability enters a decision - including lesson 4's cost calculation
in its handout section 7.2, where the threshold depends on the probability being real.
:::

# Notebook 2, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| on the pumps, 39 | `GaussianNB` beside `KNeighborsClassifier` and `LogisticRegression` → 0.933 |
| what training computed, 31-37 | `.theta_`, `.var_`, `.class_prior_`, `predict_proba` → 0.981 at 48 Hz |
| checking the assumption, 40-41 | `X[y == c].corr()`, within each class → −0.420 |
| the interacting sensors, 42-44 | the same models, plus `SVC` → 0.404 |
| confidence, right and wrong, 45 | `train_test_split`, then `predict_proba` → 0.567 against 0.555 |

::: notes
Where breaking the assumption is cheap, and one step away where it is fatal. Run
notebook 02. Sixteen minutes.

If time runs short, keep the cell that prints 0.404 next to the 0.523 baseline. Let
them see a below-chance score appear from correct code on solvable data before
any explanation arrives.

The notebook prints the within-class correlations on both datasets. Before that
cell runs, ask them to predict the signs for the interacting sensors: it is the
diagnostic they should carry - measure the assumption, do not assume it.

If time allows, the interesting exercise is to hand Naive Bayes the product of
the two sensors as a third column and watch it recover. That is the whole story
of feature engineering versus model choice in one cell.
:::

# SVM: among the lines that fit, take the widest margin

- When a line can separate two classes, **infinitely many** lines do, all
  with **zero training error**: the error cannot choose
- The SVM picks the line with the widest **margin**: an empty strip (the
  slab) around the line, widened until it touches a point of each class
- The reason: a line that passes close to a point gets it wrong after a small
  change in that point's readings

::: notes
The margin idea goes back to Vladimir Vapnik and Alexey Chervonenkis in the
Soviet Union in the 1960s; the version with slack that everyone runs is Corinna
Cortes and Vapnik's, 1995.

Start with the problem rather than the solution. Draw two well-separated blobs
on the board and ask for the boundary. Somebody will draw a line. Ask whether
another line would have been equally good, and keep asking until the room sees
that there are infinitely many, all with identical training error.

So the error cannot decide: the count of errors is flat across all these lines,
with no slope to follow. Something else has to. Each loss picks one line by its
own criterion: logistic regression breaks the tie with log loss - a perfectly
respectable answer, and one answer among several. The SVM breaks it differently:
take the boundary with the most room around it. Same kind of algorithm, a
different objective. Only an algorithm that stops at the first line with zero
errors, like the perceptron, lands on whichever line it happens to meet first.

A board example in one dimension, where the "line" is a threshold. Healthy pumps
at 1, 2 and 3; faulty ones at 7, 8 and 9. Every threshold between 3 and 7 has
zero errors. A threshold at 3.5 is 0.5 from the nearest healthy pump; one at 5 is
2 from both sides. A new healthy pump arrives at 4, one step from the others:
3.5 calls it faulty, 5 calls it healthy. The SVM picks 5 - the middle of the
empty gap, the threshold the data must move furthest to cross.

The intuition, and this is the sentence to say slowly: maximising the distance to
the closest points chooses the boundary that tolerates the most movement in the
data before it changes its mind. That is a statement about GENERALISATION, not about fit - which is unusual,
because almost everything else in this course fits the training data as well as
possible and protects generalisation only indirectly, through a penalty or a
validation score.

Handout section 5.1.
:::

# 80 made-up points: three of them fix the line (support vectors)

![](svm_margin.png)

::: notes
Made-up points, not pumps: two groups of 40 drawn around (−1.6, 0) and
(+1.6, 0), and a linear SVM with a very large C, so close to the strict version
of the next slide. The solid line is the boundary, the dashed lines are the edges
of the slab, and the circled points are the support vectors touching it.

The number to say out loud: of eighty points, THREE determine the answer. Move
any of the other seventy-seven - anywhere, as far as you like, as long as they
stay outside the slab - and the boundary does not shift by a millimetre.

That is where the name comes from, and it is a genuinely different idea from
anything so far. Every linear model in lessons 3 and 4 used every row: each one
contributed to the gradient in proportion to its error. Here most of the data is
irrelevant once the fit is done.

Training does not find the support vectors first: it solves for the line, and
the support vectors are the points that end up on the edge of the slab.

Two consequences worth naming. The model is small - for a linear SVM, w and b
are enough, three numbers here; a kernel SVM, later, has to keep its support
vectors, still not the whole dataset, which is the contrast with k-NN. And it is sensitive
in a specific way: an outlier near the boundary matters enormously, while an
outlier far from it matters not at all.
:::

# The strict problem: the widest slab with no point inside

- Labels −1 (healthy), +1 (faulty): y·f(x) > 0 means "classified right";
  in code y stays 0/1
- Boundary w·x + b = 0, scaled so the closest points have |w·x + b| = 1:
  the slab is 2/‖w‖ wide
- Widest slab = shortest w: minimise ½‖w‖², every point outside,
  yᵢ(w·xᵢ + b) ≥ 1
- No term for errors: the constraints forbid them

::: notes
This is the problem the previous slide solved, written down, so that the next
slide has something to relax.

The labels first. Lessons 3 and 4 used 0 and 1; here healthy is −1 and faulty
+1, a relabelling with no content. It buys one product for "right side": f(x) =
w·x + b is positive on the faulty side, so y·f(x) > 0 says the sign of the
prediction and the label agree, and y·f(x) ≥ 1 says the point is outside the
slab. scikit-learn does the relabelling itself: y stays 0/1 in code.

Then the scale. Multiplying w and b by ten does not move the boundary, so fix the
scale by a choice: the closest points have |w·x + b| = 1. Then the slab reaches
1/‖w‖ on each side, 2/‖w‖ in all, and maximising its width is minimising ‖w‖ -
written ½‖w‖² for a tidy derivative, as in lesson 3.

The parallel to make: in lesson 3, ridge regression minimised the mean squared
error plus a penalty on the coefficients. Here the objective is the penalty
alone, ½‖w‖², and the data enter only as constraints - every point outside the
slab, no errors allowed. The next slide adds the data term. Handout section 5.2
derives each step.
:::

# Not separable? Charge for the violations

- With 4% of the labels flipped, no slab separates the pumps: the strict
  problem has **no solution**
- Give each point i an allowance ξᵢ ≥ 0, its slack: how far it may fall
  short of its edge of the slab. Each unit of slack costs C
- ξᵢ = 0 outside the slab, 0 to 1 inside it, above 1 on the wrong side
- That charge is the **hinge loss** max(0, 1 − y·f(x)), labels ±1,
  f(x) = w·x + b

::: notes
The point worth stressing is why the soft margin is not a patch or a
convenience. With even one point with a flipped label inside the other class, the strict
problem is INFEASIBLE - there is no solution, not a bad one. Real data always
contains such points. So the soft margin is the only version anyone ever runs;
the strict (hard-margin) version is only a teaching device.

Our own labels are flipped at 4%, deliberately, so this is not a hypothetical
for this dataset: the hard-margin problem on these pumps has no answer.

How the slack enters: the constraint yᵢ(w·xᵢ + b) ≥ 1 becomes ≥ 1 − ξᵢ, and the
objective gains C Σ ξᵢ. At the best fit each ξᵢ is as small as allowed, which is
the hinge of the last bullet. Three cases with numbers: a point with y·f(x) = 2,
outside on its own side, has ξ = 0 and pays nothing; one with 0.4, inside the slab
but on the right side, has ξ = 0.6 and pays 0.6 C; one with −0.5, on the wrong
side, has ξ = 1.5 and pays 1.5 C. The trade: a wider slab lowers ½‖w‖² but lets
more points inside, raising C Σ ξᵢ; C is the exchange rate.

Then read the three cases of the slack aloud, because they come back twice: in
the weights the training produces, and in who gets a vote at prediction. Put it
next to lesson 4's log loss: that one shrinks for a well-classified point but
never reaches zero, so every row keeps pulling. The hinge reaches zero, and a
point with zero loss drops out of the answer entirely.

The next slide draws the three cases. Handout section 5.2 has the derivation.
:::

# Three kinds of point: outside the slab, inside it, on the wrong side

![](svm_slack.png)

::: notes
The previous slide's three cases, drawn. The 80 made-up points of slide 48 with
four more placed against their class, and a soft-margin linear SVM with C = 1.

Three points are circled (notebook 3 prints them). A teal point outside the slab
on its own side: y·f(x) = 1.22, slack 0, it pays nothing. A rust point inside
the slab but still on the right side: y·f(x) = 0.45, slack 0.55. A teal point
across the boundary: y·f(x) = −0.86, slack 1.86, more than 1 because it is on the
wrong side. With C = 1 the slab is wider than on slide 48: errors are cheap
enough that the model accepts a few points inside rather than squeezing the
slab. Handout section 5.2.
:::

# The hinge loss stops at zero

![](hinge_vs_log_loss.png)

::: notes
What each point pays, against y times f(x): large and positive when it is
safely on its own side, negative on the wrong side, 1 at the edge of the slab.
The axis is not "the margin": the margin is the slab's width; y·f(x) belongs to
one point.

Walk the three zones left to right. Wrong side: both losses are large. Inside the
slab: the hinge still charges, falling in a straight line. Right of 1, outside
the slab: the hinge is exactly zero, and the log loss from lesson 4 is still
charging - 0.127 at y·f(x) = 2, 0.049 at 3. The dashed curve is lesson 4's log
loss rewritten for labels ±1, ln(1 + e^(−y·f(x))): 0.693 at 0.

The sentence to land: a point with zero loss stops pulling on the answer - zero
loss means zero gradient, so a point beyond 1 does not move w or b at all. That
is where support vectors come from - and why logistic regression never has any.
Handout section 5.2.
:::

# The SVM objective: small w, plus C times the slack

$$\min_{w, b, \xi} \ \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i \\ \text{subject to } y_i(w^\top x_i + b) \geq 1 - \xi_i, \ \xi_i \geq 0$$

::: notes
The second line is the constraint: every point on its own side of the slab,
y_i(w'x_i + b) at least 1, or paying xi_i for the shortfall. Handout section
5.2 writes it out.

Do not derive this. State what each piece is doing and move on - handout section
5.2 has the derivation, including why maximising the margin is the same as
minimising the norm of w.

Read it against the strict problem of slide 49: the same ½‖w‖², plus
C Σ ξᵢ, with the constraint relaxed from ≥ 1 to ≥ 1 − ξᵢ.

The first term wants a wide slab. The second term wants few violations. C sets
the exchange rate between them, and that is the entire content of the
expression. Read it out in those words rather than reading the symbols.

Then the parallel with lesson 3. Each ξᵢ ends up equal to point i's hinge loss,
so divide the whole objective by C: the sum of the hinge losses plus
(1/(2C))‖w‖². That is lesson 3's shape, a loss on the data plus λ times a
penalty, with ½‖w‖² playing ridge's penalty, the hinge playing the mean squared
error, and λ = 1/(2C).

If someone asks about the half in front of the norm: it is there to make the
derivative tidy, exactly as in lesson 3's cost function.
:::

# C is the price of a training error

- **Large C**: slack is expensive, the slab narrows to classify more
  training points. Low bias, high variance
- **Small C**: slack is cheap, the slab stays wide, some training points
  stay wrong. Higher bias, lower variance
- C plays the role of k in k-NN and of λ in lesson 3, but **large C means
  less regularisation**: divided by C, the penalty weighs 1/(2C)

::: notes
Third appearance of the same idea, and say so - that repetition is the point.
Lesson 3 had λ, the first half of today had k, now C. Every model has one knob that trades
fitting this data against surviving the next, and recognising it in an unfamiliar
model is worth more than memorising three names.

Then the direction, which catches people out constantly and is worth writing on
the board. Large λ means MORE regularisation. Large C means LESS. They run
opposite ways, and scikit-learn's `C` parameter in `LogisticRegression` has the
same inverted convention - it is the reciprocal of the penalty there too.

Ask them which way they would expect C to run before you tell them. Most guess
wrong, and having guessed wrong once is what makes it stick.

Be precise about what a large C does to the boundary. With a linear kernel the
boundary and the two edges of the slab stay three parallel straight lines: a
larger C can only make the slab narrower, or shift and rotate it, to classify
more training points. It never bends it. With the RBF kernel the boundary can
also bend around individual points, and we will see on the pumps how far that
goes when C and γ are both large.

The two ends, worth one sentence each. C towards infinity: no slack is
affordable, and the strict problem comes back - infeasible on the pumps. C
towards 0: only ½‖w‖² is left, its minimum is w = 0, and the model ignores the
readings. And k runs like λ - a larger k is smoother - while C runs the other
way.
:::

# The best straight line is still a straight line

| Model, on the pumps (standardised, default C = 1) | Accuracy, 5-fold cross-validation | Support vectors, fitted on all 1,200 |
|---|---|---|
| SVM, linear kernel: a straight line | **0.613 ± 0.000** | 947 of 1,200 (**79%**) |
| SVM, RBF kernel: a curved boundary | 0.947 ± 0.005 | 278 of 1,200 (23%) |

::: notes
The linear kernel scores 0.613 - the base rate, exactly as logistic regression
did on slide 6, with the same zero variance across folds: a straight line
through a disc inside a ring cannot do better than calling every pump faulty.
Both models standardise first and keep scikit-learn's defaults, C = 1 and, for
the RBF kernel, γ = 0.5.

Make the point explicitly, because it is the one students most often get
backwards: the margin criterion is a better way of CHOOSING among straight
lines. It does not give you anything other than a straight line. When no straight
line works, choosing the best one optimally still gets you nothing. A better
criterion cannot rescue an inadequate hypothesis class.

Then the RBF row, which is the same algorithm with one argument changed, at
0.947 - the best score on the pumps in the whole lesson, and close to the 0.96
ceiling.

The support-vector column says the same thing in a different way, and that is
the next slide. With slack, a support vector is any pump on the edge of the
slab, inside it or on the wrong side.
:::

# On the pumps: a straight line leaves almost every pump in its slab

![](svm_support_vectors_pumps.png)

::: notes
The previous table's last column, drawn on the standardised pumps. Circled: the
support vectors - on the edge of the slab, inside it, or on the wrong side.

Left, the linear SVM: 947 of 1,200 circled. Its boundary lies off the picture,
because it calls every pump faulty; the dashed line is the far edge of its slab,
and almost every pump is inside. Right, the RBF SVM: 278 circled, in a thin
band around the ellipse it draws; the pumps deep in the disc and far out in the
ring are left uncircled, and the model forgets them. The next slide turns the
count into a diagnostic.
:::

# Many support vectors: a free warning

- The linear SVM, `SVC(kernel="linear")`, keeps **79%** of the pumps as
  support vectors; the RBF SVM keeps **23%**
- 79% means almost every pump sits on or inside the slab: no strip of empty
  space separates the classes
- The count comes free after `fit`, as `len(model.support_)`: a high share
  is a warning before any cross-validation

::: notes
Explain what 79% actually means geometrically. A support vector is a point on
the margin or inside it. If 79% of your data is in that position, there is no
region of empty space between the classes at all - the model has been forced to
place a boundary through a crowd.

Contrast with 23%, where the model found genuine room and only the points near
the edge matter.

Give them the diagnostic to keep, since it costs nothing: after fitting an SVM,
look at `len(model.support_)` over the number of training rows. High means the
model is struggling to find room, which usually means the kernel cannot draw the shape the classes need. It is a warning available BEFORE you cross-validate anything.

Worth adding that it also predicts prediction cost - every support vector is one
kernel evaluation per query, so a model needing 79% of the data has thrown away
most of the speed advantage over k-NN.
:::

# A new coordinate makes the classes separable

- In the plane, healthy pumps sit in a disc, faulty ones around it: no line
  separates them
- Add a third coordinate: each pump's **distance from the design point**,
  standardised
- Healthy pumps rise a little, faulty ones a lot: a flat plane separates
  them, flipped labels apart
- The classes were separable all along, in other coordinates

::: notes
Build the picture in words first and let them see it before the figure appears.
Take the scatter plot from the start of the lesson, lift each point off the page
by its distance from the centre, and look from the side.

Standardised, as on slide 10: in raw units the hertz would swamp the bar. Then
the sentence that reframes the whole lesson: the classes were never
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
horizontal cut separates the colours, apart from the flipped labels - the teal
dots high up among the rust. A flat cut up there is a circle down here: linear
in the new coordinates is curved in the old ones. Nothing was added - no new
measurement, no extra sensor. The third coordinate is computed from the two they
already had.

Then set up the objection that the next two slides answer. This worked because the
lift was three-dimensional and we could guess it. The spaces that work in general
are enormous - the useful ones are sometimes infinite-dimensional - and
computing the coordinates of every point in such a space is impossible.
:::

# The SVM's solution: a weighted sum of pumps

- Slide 53's objective: ½‖w‖² + C Σ max(0, 1 − yᵢ(w·xᵢ + b))
- Its slope in w: w − C Σ yᵢ xᵢ, over the pumps inside the slab or on the
  wrong side
- Zero slope: w = Σ aᵢ yᵢ xᵢ; aᵢ = C inside, 0 outside, 0 to C on the edge
- Only the support vectors have aᵢ > 0: the pegs of the tent

::: notes
This is the rewriting the kernel trick rests on, which is why it comes before
the trick, not after. Derive it in the one step on the slide: set the slope of
the objective to zero. The first term contributes w. Each hinge term contributes
−C yᵢ xᵢ while pump i is inside the slab or on the wrong side, nothing once it is
outside, where the hinge is flat, and something in between exactly on the edge,
where the hinge has a corner. Zero slope gives w as a sum of pumps.

The same result, reached through multipliers and the problem in the weights
alone, is called the **dual formulation**: handout section 5.4.

Say it in one sentence first: instead of solving for $w$ directly, the
optimisation turns out to hand back $w$ as a weighted sum of the training
points themselves, one weight $a_i$ per pump. The picture: the boundary is a
tent held up by the points pressing against it, and $a_i$ is how hard pump $i$
presses. A pump comfortably on its own side of the slab presses on nothing, so
its weight is exactly zero. Only the pumps on or inside the slab - the support
vectors of slide 48 and handout section 5.1 - get a positive weight, which is where that 79%
against 23% came from.

Now the payoff, and this is the sentence that makes the next slide make sense.
Substitute this $w$ back into the boundary, $w^\top x + b$, and every $x_i$ is
still there, but only ever multiplied against another point: $x_i \cdot x_j$
in training, $x_i \cdot x$ in prediction. Nowhere does a coordinate appear on
its own. That is the whole reason a kernel is enough - a function that returns
that one number, the inner product in the lifted space, is all either
computation ever asks for.

The counts match notebook 3 on the pumps with the RBF kernel: 268 pumps at the
ceiling C, 10 in between, 922 at zero. The longer route, from the constrained
optimisation to this formula, is handout section 5.4.
:::

# So the new coordinates are never computed

- φ(x): a pump's new coordinates. Slide 60 there: w = Σ aᵢ yᵢ φ(xᵢ)
- Boundary Σ aᵢ yᵢ φ(xᵢ)·φ(x) + b; training needs only φ(xᵢ)·φ(xⱼ): the
  pumps enter **only through inner products**
- A **kernel** K(x, x′) gives φ(x)·φ(x′) from the original readings: φ is
  never computed
- RBF, scikit-learn's default: one exponential per pair

::: notes
Kernels were joined to the margin by Bernhard Boser, Isabelle Guyon and Vapnik
in 1992, at Bell Labs; that paper is what made SVMs a practical tool.

The trick in one sentence: the SVM never needs the coordinates, only the inner products between points,
which carry their lengths and the angles between them, and those can be obtained without ever going there.

Walk the bullets. φ(x) is the lift - slide 58's (z₁, z₂, radius) is one choice of
it. Run the previous slide in those coordinates and w is a sum of lifted pumps.
Put that w into the boundary and every φ(xᵢ) meets φ(x) only in an inner
product; training is the same, because f at each training pump and ‖w‖² are
both sums of φ(xᵢ)·φ(xⱼ). So a function that returns that inner product
straight from the original two readings is all either computation needs.

Do not prove it here - the previous slide gave the one line that makes this claim
true, and handout section 5.4 has the full three-fact argument behind that
line. The claim to state and let land now is that a function of two ordinary
two-dimensional vectors returns their inner product in a space of infinite
dimension, at the cost of one exponential.

And close the loop from two slides back: we chose the lift in the picture by
knowing the answer. The RBF kernel does something equivalent without being told,
which is exactly why it works on problems where nobody could guess the right
coordinates.
:::

# The kernel trick by hand: 25 = 25

| Route, for u = (1, 2) and v = (3, 1) | Computation | Result |
|---|---|---|
| Lift with φ(x) = (x₁², √2·x₁x₂, x₂²), then multiply | φ(u) · φ(v) = (1, 2.83, 4) · (9, 4.24, 1) = 9 + 12 + 4 | **25** |
| Never lift: the kernel (u · v)² | (1×3 + 2×1)² = 5² | **25** |

::: notes
Make the trick concrete before the famous formula. The lift here lists every
product of two coordinates: phi(x) = (x1², √2·x1·x2, x2²). Two points, u = (1, 2)
and v = (3, 1) - not a and b, which are already the weights and the intercept.

First row: lift both into three dimensions and take the ordinary inner product -
9 + 12 + 4 = 25. Second row: never lift, take the inner product in the plane, 5,
and square it - 25. Same number. The middle term is exactly 2√2 × 3√2 = 12.

Why it is always the same, in three lines on the board: (u·v)² = (u₁v₁ + u₂v₂)²
= u₁²v₁² + 2u₁v₁u₂v₂ + u₂²v₂² = (u₁²)(v₁²) + (√2u₁u₂)(√2v₁v₂) + (u₂²)(v₂²) =
φ(u)·φ(v). The √2 in φ is there to produce the 2 of the cross term. The second
route never left two dimensions, so the function (u · v)² IS the inner product in
the lifted space. That is all a kernel is.

Then connect it to the figure they just saw: the lifted space contains x1² and
x2², and their sum is the squared distance from the centre - the coordinate we
chose by hand. A flat plane there is a circle down here. The kernel found the
lift we had to guess. Handout section 5.4.
:::

# The RBF kernel: similarity falls with distance

- Radial basis function (RBF) kernel: K(x, x′) = exp(−γ‖x − x′‖²)
- A similarity: 1 for identical pumps, towards 0 far apart. With γ = 0.5:
  0.61 at one standard deviation, 0.14 at two
- Why it: expand the exponential and it holds (x·x′)², (x·x′)³, … every
  degree at once - infinitely many new coordinates, no lift to guess
- Cost: one exponential per pair of pumps

::: notes
One line, and it is the whole of what scikit-learn does when you leave `kernel`
at its default.

Read it aloud as a similarity rather than as a formula: the closer two points
are, the nearer the exponent is to zero and the nearer the kernel is to one; far
apart, it falls away to nothing. That is all an RBF kernel says - how much does
this point look like that one.

Why this kernel: slide 62's kernel is the degree-2 term of its expansion, and the
exponential holds every degree at once, so its φ has infinitely many entries and
can draw the circle the pumps need, or any smooth shape, without being told
which. The link back to k-NN: with this kernel the boundary is a vote of the
support vectors weighted by similarity - near ones count, far ones barely.

Now say what γ means, because the last part of the lesson turns on it: it sets
how far a single training point's influence reaches. Small γ, wide reach, smooth boundary.
Large γ, each point influences only its immediate neighbourhood - and a boundary
that can afford an island around every pump with a flipped label.

A number to anchor it: the similarity halves at a distance of sqrt(ln 2 / γ).
For scikit-learn's default on the standardised pumps, γ = 0.5, that is about 1.2
standard deviations: exp(−0.5) = 0.61 at one standard deviation, exp(−2) = 0.14
at two, 0.011 at three.

Worth naming the cost, since this is an engineering course: one exponential per
pair of points, so the kernel matrix is m × m. That is why support vector
machines are a poor fit above a hundred thousand rows, and it is a property of
the method rather than of the implementation.
:::

# Training: one weight per pump

- fit minimises slide 53's objective in the weights aᵢ, two at a time: no
  learning rate, one answer (RBF, γ = 0.5, C = 1)
- **922** pumps get weight 0: outside the slab, they drop out
- **268** get C = 1 (inside the slab or wrong side), **10** sit on its
  edge: 278 support vectors
- The model kept: those 278 pumps, their weights, and b = +1.105

::: notes
The solver is SMO, sequential minimal optimisation, published by John Platt at
Microsoft Research in 1998; it runs inside libsvm, which is what scikit-learn's
SVC calls.

Say what "fit" does, because after lesson 3 they will expect gradient descent.
It is not. The SVM solves the dual with the kernel substituted in: one weight
per training point, found by a solver that picks two weights at a time - two,
because the weights times the labels must sum to zero, so one cannot move
alone - sets them to their best values with the rest fixed, and repeats until
nothing improves. Convex, so there is one answer and it is found every time.

Make explicit a distinction the earlier slides left open. The objective being solved
here only ever evaluates K(x_i, x_j) between pairs of training pumps - the
kernel matrix, m by m. The new pump, the one this model will eventually
classify, appears nowhere in training. K(x_i, x), with x standing for any pump, is the prediction formula on the next
slide: same function, but its second argument is
now the pump being classified rather than another training point.

Then the counts, which match the three cases of the slack from earlier. Weight
zero: the 922 pumps safely on their own side, which the model now forgets.
Weight at the ceiling C: the 268 inside the slab or on the wrong side - with
C = 1 that weight is exactly 1, so each of them simply votes with its label.
Between: the 10 sitting exactly on the edge.

The model you ship is 278 points and their weights. Handout section 5.5.
:::

# The SVM's score at 48 Hz, 5.6 bar: Σ aᵢ yᵢ K(xᵢ, x) + b = +1.883

| Term, for the pump x at 48 Hz, 5.6 bar | Value |
|---|---|
| Faulty support vectors: sum of aᵢ · K(xᵢ, x) | +37.455 |
| Healthy support vectors: minus the sum of aᵢ · K(xᵢ, x) | −36.677 |
| Intercept b | +1.105 |
| **Total: positive, so faulty** | **+1.883** |

::: notes
The title is the formula: the score sums, over the 278 support vectors, weight
times label times similarity, plus b; its sign is the class. Inference is a
vote, in four steps. Standardise the pump with the training scaler: (1.587,
−0.035). Compute its RBF similarity to each of the 278 support vectors. Add up
weight times label times similarity - label +1 for faulty, -1 for healthy - and
add b. Read the sign. Nothing else goes in: the total is exactly what
decision_function returns.

Make it concrete with the two nearest support vectors (notebook 3): a faulty
pump at 46.59 Hz and 5.58 bar, similarity 0.932, votes +0.932; a healthy one at
46.78 Hz and 5.73 bar, similarity 0.916, votes −0.916. With C = 1, 268 of the 278
weights are exactly 1, so most votes are just label times similarity.

b on its own is what a pump far from every support vector would get: +1.105,
faulty - sensible here, where far from the design point means faulty. And
+1.883 is above 1: by slide 52's reading the pump sits beyond the faulty edge of
the slab, faulty with room to spare.

The same pump Naive Bayes called faulty at 0.981, reached by a completely
different route.

The point to make: the neighbourhood decides. The 56 support vectors within
about 1.2 standard deviations contribute a net +1.141 towards faulty; the
other 222 only -0.363. But not one neighbour - the five nearest disagree, because
pumps just outside the envelope are faulty and some just inside carry flipped
labels. So the SVM is k-NN again, with a smooth similarity instead of a hard k,
only the support vectors voting, and learned weights. And the output is a score,
not a probability. Handout section 5.5.
:::

# One pump's vote, drawn

![](svm_vote_one_pump.png)

::: notes
The table on the previous slide, drawn. Every support vector is sized and
shaded by its similarity to the pump, the gold star at 48 Hz and 5.6 bar; the
dashed ellipse is where similarity falls to one half, about 1.2 standard
deviations; the black line is the SVM's boundary.

Why an ellipse: in standardised units the half-similarity line is a circle of
radius √(ln 2 / γ) = 1.18 standard deviations; on these raw axes it stretches by
each reading's standard deviation, 1.177 × 3.78 = 4.45 Hz across and
1.177 × 0.471 = 0.55 bar up and down.

Point out where the weight is: the 56 support vectors inside the dashed ellipse
carry most of it, the ones across the disc have faded out. Then point at the
star's immediate neighbours - teal and rust together, boundary pumps and flipped
labels. That is why the five nearest nearly cancel, and why it is the
neighbourhood's balance, not one neighbour, that makes it faulty. Handout
section 5.5.
:::

# Training against cross-validated, three settings

| RBF SVM on the pumps | Training accuracy (each fold's 960 pumps) | Cross-validated accuracy, 5 folds |
|---|---|---|
| γ = 0.1, C = 1 | 0.934 | 0.929 |
| γ = 1, C = 1 | 0.948 | **0.944** |
| γ = 50, C = 1000 | **0.996** | 0.902 |

::: notes
Read the two columns against each other, exactly as lesson 5 taught, and let the
last row do the work.

Both columns come from the same five folds: the training score is measured on
the 960 pumps each fold fits on, as for k on slide 12.

The bottom row has the best training score on the slide - 0.996, nearly perfect -
and the worst cross-validated one. That is the signature of overfitting, and here it is
not inferred from theory but measured on the same data by two different
procedures.

**Now the second predictable mistake, and defend the instinct.** Choosing between
these three on the training score selects the WORST model, with complete
confidence and a very impressive number to report. And the instinct behind it is
not stupid: a higher score on the data you have is, in every other part of
engineering, evidence of a better system. Fitting the data well is what we spent
lessons 3 and 4 doing. What breaks the analogy is that a flexible model can fit
data it has already been shown without learning anything transferable.

The bottom row moves two knobs at once, and each alone barely hurts (notebook
3): γ = 50 with C = 1 scores 0.960 training and 0.940 cross-validated; γ = 1 with
C = 1000, 0.953 and 0.948. Together they overfit: a narrow reach and expensive
errors make an island around every flipped pump. And "best in the middle" holds
among these three only - the default γ = 0.5, C = 1 scores 0.947. Finding the
best means a grid over both, cross-validated: lesson 5's GridSearchCV, and nested
cross-validation to report it honestly.

The reason this table is on the slide rather than in the handout is that the
next slides show the same three settings as pictures - so the failure is visible
as well as measurable, which is rare.
:::

# How far one support vector's vote reaches, for four values of γ

![](rbf_reach.png)

::: notes
Gamma as a distance rather than a symbol. Each curve is the RBF similarity
against the distance between two pumps; where it crosses the dotted line is the
distance at which a support vector's vote has halved: 2.63 standard deviations
at γ 0.1, 1.18 at the default 0.5, 0.83 at 1, and 0.12 at 50. The reach goes as
1/√γ: multiply γ by 100 and the reach shrinks tenfold. The link to k-NN: a small
γ behaves like a large k, many pumps voting and a smooth boundary; a large γ like
k = 1, only the nearest speaking.

Ask them to predict the next slide from this one: at gamma 50 a point speaks only
for a neighbourhood a tenth of a standard deviation wide. What will the boundary
look like? Islands. Handout section 5.6.
:::

# Overfitting you can see: the boundary at the three settings

![](svm_gamma_c.png)

::: notes
Three settings, with training and cross-validated scores in each title. Read them
together.

Left, γ = 0.1: a smooth boundary, slightly too smooth, and the two scores agree.
Middle, γ = 1: close to the true envelope, and the best honest score.

All 1,200 pumps are drawn, so every island has its pump. Spend time on the right
panel. The boundary has broken into bubbles - small islands around individual
points, both ways: teal islands in the faulty ring around healthy-labelled pumps
far out, and rust holes in the healthy disc around faulty-labelled pumps near the
centre - mostly the flipped labels. That is what a 0.996 training score looks
like from the outside: the model has drawn a private
territory around each flipped label so that it can get it right.

Ask them what happens to a new pump landing in the gap between two bubbles: it
is judged by whichever stray point happens to be nearest, and that is where the
0.902 comes from.

The reason this slide matters more than the table: they have seen overfitting as
a curve on a plot since lesson 5. Here it is a shape in the input space, and the
shape is obviously wrong in a way no number is. When you can plot the boundary,
plot it.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| the widest slab, 47-52 | `SVC(kernel="linear")`, `.n_support_`, `decision_function` → 3 of 80 points |
| a straight line still fails, 55-57 | `cross_val_score`, linear against RBF kernel → 947 against 278 support vectors |
| the kernel trick, 58-63 | NumPy only: (u · v)² against the lifted product → 25 = 25 |
| training and one prediction, 64-66 | `.support_`, `.dual_coef_`, `.intercept_` → +1.883 |
| γ and C, 67-69 | `SVC(gamma=..., C=...)`, `cross_validate` → 0.944 at γ = 1 |

::: notes
Margins, support vectors, and what γ does to the boundary. Run notebook 03.
Sixteen minutes.

If time runs short, keep the γ sweep with the boundary drawn at each setting, and
skip the C sweep instead, setting it as reading (the table is in handout section
5.6). Let them turn γ up themselves and watch the bubbles form - it is the most
direct experience of overfitting available in this course.

Have them print the fraction of support vectors for the linear and RBF kernels
side by side. Two numbers, and it makes the diagnostic real rather than a claim on a
slide.
:::

# The three families compared, on the same pumps and the same folds

| Model | Accuracy, 5-fold cross-validation |
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
That is larger than anything tuning has bought in this course - lesson 5's whole
hyperparameter search moved scores by a few hundredths.

Say the conclusion plainly, and it is the practical takeaway of the whole
lesson: choosing the right family matters far more than tuning the wrong one.
Nobody in the top three rows could have tuned their way out. And the way to tell
which family you need is to look at the data first, which is what lesson 2 was
for.
:::

# How to choose, in practice

| Situation | Reach for |
|---|---|
| Few features, plenty of data, a curved boundary | k-NN, or an RBF SVM |
| Very many features, little data | Naive Bayes |
| You need probabilities you can trust | **Not** Naive Bayes |
| The model must be small, or fast at prediction | SVM, not k-NN |
| The signal is an interaction between features | k-NN or an RBF SVM, not Naive Bayes |

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

- **No line separates the pumps**: 0.613, the majority share
- **k trades bias for variance**
- **In 100 dimensions the nearest point is 70% as far as the farthest**
- **Naive Bayes**: 0.933 on the pumps, **0.404** with interacting sensors
- **Choosing the family beats tuning**: 0.613 to 0.947

::: notes
Close on the number they opened the second segment with, and say it one more
time: seventy per cent. In a hundred dimensions the nearest point in the entire
dataset is seven tenths of the way to the farthest, and every method built on
distance is operating on a quantity that has stopped varying.

If they remember one habit from today, make it the diagnostic pair: before
reaching for a model, look at the shape of the data; after fitting one, check
whether the assumption it depends on is actually satisfied on your data - and
check it properly. The −0.006 said yes; the squared distances said no; and the
ellipse the model drew explained the 0.933.

And the correction to the instinct that this lesson exists to install: more
features is not free. It is nearly free for a linear model and it is expensive
for anything built on a distance.

Handout section 7 lists all of this with the section numbers.
:::

# Homework

- **Exercise 6**, discussed at the start of **Friday 20 November**
- You get a dataset, fit the three families, and argue which one you would
  choose, and why

::: notes
Set it explicitly and say out loud when it comes back: the first ten
minutes of next Friday, 20 November.

If this exercise is drawn at the exam, what counts is the reasoning, not the
accuracy. A well-argued choice that scores slightly worse beats a lucky winner
with no justification - and the
argument has to reference the data, in the way that the squared distances and
the fitted ellipse explained Naive Bayes on the pumps today.

Remind them that everything must be cross-validated and reported with a spread,
in the format lesson 5 set. Scaling goes inside the pipeline; at the exam that is one of the first things
looked at.

Next week: decision trees and ensembles. A fourth family, with a different
answer again to the same question - how do you draw a boundary that is not a
line.
:::

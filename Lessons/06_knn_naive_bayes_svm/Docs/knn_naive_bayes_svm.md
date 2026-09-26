---
title: "k-NN, Naive Bayes and Support Vector Machines"
subtitle: "Lesson 6 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "6 November 2026 · reading time about 100 minutes"
---

## Lesson plan

| Time | Minutes | Segment | Material |
|---|---|---|---|
| 0:00–0:10 | 10 | Exercise 5 discussed; a problem no line solves | Slides 2–6 |
| 0:10–0:30 | 20 | k-nearest neighbours, and choosing k | Slides 7–14 |
| 0:30–0:52 | 22 | The curse of dimensionality | Slides 15–21 |
| 0:52–1:14 | 22 | **Notebook 01** — k-NN and the curse | Slide 22 |
| 1:14–1:26 | 12 | **Break** | Slide 23 |
| 1:26–1:48 | 22 | Naive Bayes, and when its assumption holds | Slides 24–35 |
| 1:48–2:06 | 18 | **Notebook 02** — where it fails | Slide 36 |
| 2:06–2:32 | 26 | Margins, support vectors, the kernel trick | Slides 37–49 |
| 2:32–2:50 | 18 | **Notebook 03** — kernels in practice | Slide 50 |
| 2:50–3:00 | 10 | The three compared; homework | Slides 51–54 |
| | **180** | **Total** | **53 slides, 3 notebooks** |

---

## 1. Why this lesson exists

Lessons 3 and 4 built linear models and spent most of their effort on how to fit
them honestly. Lesson 5 built the apparatus for telling whether a model works.

None of that told you **which model to reach for**, and this lesson is the first
that offers a choice. Three families, each attacking classification from a
completely different direction:

- **k-nearest neighbours** makes no assumptions and does no fitting. It
  remembers the data and votes.
- **Naive Bayes** makes one very strong assumption and, in exchange, trains in a
  single pass and works in thousands of dimensions.
- **Support vector machines** pick a boundary by a criterion nothing else in
  this course uses, and then change coordinates when no boundary exists.

They are not three ways of doing the same thing, and the differences between
them are the content of the lesson.

### 1.1 The dataset, and the point it makes immediately

1,200 industrial pumps, two readings each: vibration in Hz and pressure in bar.
A pump has a design operating envelope, and it is **faulty when its readings
fall outside that envelope — whether too low or too high**.

That one physical fact decides everything that follows. The healthy pumps form a
disc around the design point; the faulty ones form the annulus around them.

![](pump_scatter.png)

*1,200 pumps. The healthy ones are surrounded, because a pump can fail by
running too slow as well as too fast. No straight line separates these classes.*

4% of the labels are deliberately flipped, so no model can exceed roughly 0.96.
That ceiling is lesson 5's noise floor arriving in a classification problem, and
it is the number to compare every score below against.

Before building anything, it is worth measuring what a straight boundary costs:

| Model | Cross-validated accuracy |
|---|---|
| Always predict the majority class | 0.613 |
| Logistic regression (lesson 4) | **0.613 ± 0.000** |

**Logistic regression has not been narrowly beaten. It has learned nothing at
all**, and it predicts "faulty" for every pump because with a straight boundary
that is genuinely its best available answer. The zero standard deviation across
folds is the giveaway: a model that gives everything the same answer is
perfectly consistent.

---

## 2. k-nearest neighbours

### 2.1 The whole algorithm

> To classify a new point, find the $k$ training points closest to it, and take
> the majority vote.

There is no training step. "Fitting" means storing the data, which is why k-NN
is called a **lazy learner** — an unusually honest name for an algorithm.

Two decisions hide in that sentence, and both matter.

**What "closest" means.** Almost always Euclidean distance:

$$d(x, x') = \sqrt{\sum_{j=1}^{n} (x_j - x'_j)^2}$$

Because every feature contributes to that sum through its own units, the
features must be on comparable scales. Vibration ranges over tens of Hz and
pressure over a couple of bar, so without scaling vibration would decide every
neighbour by itself. Lesson 2's scaling argument arrives here with immediate
consequences rather than as hygiene.

**What $k$ is.** Section 2.2.

### 2.2 k is the bias-variance dial, made visible

Lesson 5 decomposed error into bias and variance. In k-NN you can watch the
trade-off directly by turning one integer.

**Small $k$** — the boundary follows every point, including mislabelled ones.
Low bias, high variance. At $k = 1$ the training accuracy is **exactly 1.000**,
always, on any dataset: every point is its own nearest neighbour. That is the
purest illustration of lesson 5's point that a training score measures nothing.

**Large $k$** — the vote is taken over a wide neighbourhood, so the boundary
smooths and eventually stops following real structure. High bias, low variance.

Measured on the pumps:

| $k$ | Training | Cross-validated |
|---|---|---|
| 1 | **1.000** | 0.912 |
| 3 | 0.957 | 0.941 |
| 5 | 0.953 | **0.944** |
| 15 | 0.948 | 0.938 |
| 51 | 0.940 | 0.933 |
| 201 | 0.899 | 0.881 |
| 401 | 0.828 | 0.708 |

![](knn_choosing_k.png)

*The two curves and the two lines that bound them: the noise ceiling at 0.96,
which nothing can exceed, and the majority baseline at 0.613, which everything
should. Note where the training curve starts.*

The best value, 5, sits where the neighbourhood is wide enough to average out
the flipped labels and narrow enough to still follow the boundary.

![](knn_boundaries.png)

*The same data at three values of k. At k = 1 the boundary is ragged, with
islands around individual mislabelled points. At k = 15 it is a clean disc,
close to the envelope that generated the data. At k = 401 it has **inflated**
past the true envelope and swallows faulty pumps — it has not collapsed to one
class, it has stopped following the boundary and started averaging over it.*

### 2.3 What it costs

No training time at all, and you pay at every prediction instead. A naive
implementation compares the query against **every** training point:

$$O(mn) \text{ per prediction, for } m \text{ training rows and } n \text{ features}$$

For 1,200 pumps that is nothing. For ten million rows answering a thousand
queries a second it is the entire engineering problem, and it is why approximate
nearest-neighbour indexes are an industry.

There is a second cost that is easy to miss: **the model is the dataset**. You
cannot ship the model without shipping the training data, which is a legal
question as much as a practical one — see this lesson's companion reading and
lesson 2's.

---

## 3. The curse of dimensionality

### 3.1 The demonstration

k-NN just solved a problem that defeated logistic regression completely. Here is
the price.

Add columns of **pure noise** — drawn from a normal distribution, unrelated to
anything. The two real readings are untouched, so the problem is exactly as
solvable as before. Only the number of columns changes.

| Noise columns | Total columns | Accuracy | Above baseline |
|---|---|---|---|
| 0 | 2 | 0.938 | +0.325 |
| 5 | 7 | 0.854 | +0.242 |
| 10 | 12 | 0.762 | +0.149 |
| 25 | 27 | 0.662 | +0.049 |
| 50 | 52 | 0.602 | **−0.011** |
| 100 | 102 | 0.578 | −0.035 |

**The signal never left.** Those two columns are still there and still
sufficient. But by fifty noise columns, k-NN is below the majority baseline: a
model that ignored the data entirely would now do better.

### 3.2 Why: distances stop varying

The mechanism is geometry, not statistics, and it has nothing to do with k-NN
specifically.

**The picture first.** Scatter points uniformly in a cube, pick one, and measure
the distance to its nearest neighbour and to its farthest. In two dimensions
those are very different numbers, and that difference is exactly what makes
"nearest" a meaningful word.

Now add dimensions, and follow a single distance while it happens. Squared
distance is a sum with one term per dimension, and for points scattered
independently those terms are independent draws from the same distribution:

$$d(x, x')^2 = \sum_{j=1}^{n} (x_j - x'_j)^2$$

Call the mean of one term $\mu$ and its variance $v$. Both are fixed by how a
single coordinate is distributed — for coordinates uniform on $[0, 1]$ they are
$\mu = 1/6$ and $v = 7/180$ — and, crucially, **neither depends on $n$**.

**Where the two numbers come from.** Both belong to a single pair of
coordinates, so one dimension is enough, and the only thing to know is how the
gap between two random points on a line is distributed. Write $D = x_j - x'_j$.
The mean needs nothing but variances: $D$ has mean zero, so
$\mathbb{E}[D^2] = \operatorname{Var}(D)$, and for independent points the
variances add. A uniform coordinate on $[0, 1]$ has variance
$\mathbb{E}[x^2] - \mathbb{E}[x]^2 = 1/3 - 1/4 = 1/12$, so
$\mu = 1/12 + 1/12 = 1/6$. The variance needs the fourth power,
$v = \mathbb{E}[D^4] - \mu^2$, and for that the shape of $D$. A small gap can
arise from many positions of the pair, a gap near 1 only from one point at each
end, so $D$ has the triangular density $1 - |t|$ on $[-1, 1]$. It is symmetric,
so integrate over $[0, 1]$ and double:

$$\mathbb{E}\left[D^4\right] = 2\int_0^1 t^4 (1 - t)\,dt
  = 2\left(\tfrac{1}{5} - \tfrac{1}{6}\right) = \tfrac{1}{15},
  \qquad v = \tfrac{1}{15} - \tfrac{1}{36} = \tfrac{7}{180} \approx 0.039$$

The same density gives the mean by a second route,
$2\int_0^1 t^2 (1 - t)\,dt = 2(1/3 - 1/4) = 1/6$.

Adding $n$ such terms adds their means, and because they are independent it
adds their variances too:

$$\mathbb{E}\left[d^2\right] = n\mu,
  \qquad \operatorname{Var}\left(d^2\right) = nv,
  \qquad \text{so} \quad \operatorname{sd}\left(d^2\right) = \sqrt{nv}$$

**That is the entire mechanism: the centre grows like $n$ and the spread only
like $\sqrt{n}$.** What the word "nearest" needs is not a large spread but a
large spread *measured against the distances themselves*, and dividing one by
the other shows what becomes of it:

$$\frac{\operatorname{sd}(d^2)}{\mathbb{E}\left[d^2\right]}
  = \frac{\sqrt{nv}}{n\mu}
  = \frac{1}{\sqrt{n}} \cdot \frac{\sqrt{v}}{\mu}$$

For the uniform cube $\sqrt{v}/\mu = 1.18$, so the relative spread is
$1.18/\sqrt{n}$: **0.84 in two dimensions, 0.12 in one hundred.** Going back
from $d^2$ to $d$ does not change the rate — at $n = 100$ the relative spread of
the distance itself is 0.059, half that of its square.

Every pairwise distance is a draw from that distribution, the nearest and the
farthest among them included. Squeeze the distribution against its own centre
and those two are squeezed together with it, so the ratio of the smaller to the
larger is pushed towards 1:

| Dimensions | nearest ÷ farthest |
|---|---|
| 2 | 0.016 |
| 10 | 0.263 |
| 50 | 0.592 |
| 100 | **0.701** |
| 500 | 0.855 |

![](distance_concentration.png)

*As dimensions grow, the nearest point stops being near. The ratio climbs
towards 1, where every point is the same distance from every other.*

**In two dimensions the nearest point is about 2% as far away as the farthest.**
"Nearest" is a strong claim.

**In one hundred dimensions it is 70% as far away as the farthest.** The nearest
point is barely nearer than a random one, and a vote among "the five nearest" is
close to a vote among five taken at random.

### 3.3 What the curse is, and is not

It is **not** that high-dimensional problems are inherently unlearnable. Lesson
9's networks work in thousands of dimensions.

It is that **methods built on distance lose their footing**, because the
quantity they depend on stops varying. That includes k-NN, k-means (lesson 8),
and RBF kernels — which is why the RBF SVM also degraded in the table above,
though more slowly.

**The predictable mistake, and why the instinct is sound.** Adding features
because they might help is good practice with a linear model, where an
irrelevant feature costs you a coefficient near zero and very little else. With
k-NN it costs you a dimension in the distance, and dimensions are what the
method is made of. The same habit, transferred one lesson later, does real
damage.

---

## 4. Naive Bayes

### 4.1 Bayes' rule, and where the difficulty is

We want $P(y = c \mid x)$. Bayes' rule turns it into quantities we can estimate:

$$P(y = c \mid x) = \frac{P(x \mid y = c)\,P(y = c)}{P(x)}$$

$P(y = c)$ is the prior — a count. $P(x)$ is identical across classes, so it
cannot change which class wins and can be dropped.

**Dropped for choosing, not lost.** $P(x)$ is the probability of seeing this
combination of readings whatever the class — by the law of total probability,
the sum of the numerators, $\sum_c P(x \mid y = c)\,P(y = c)$. That is why
dropping it costs nothing even when a probability is wanted: divide each
class's score by the sum of the scores and the denominator is back, exactly.
The tempting conclusion is that a model which drops $P(x)$ can no longer report
a calibrated probability. It is reasonable, because a term has been thrown
away, and wrong, because that term is recoverable from what is kept. The pumps
are 465 healthy and 735 faulty out of 1,200, priors 0.3875 and 0.6125. Take
illustrative likelihoods of 0.12 and 0.03 for one pump's readings: the scores
are 0.0465 and 0.0184, their sum, 0.0649, is $P(x)$, and the probabilities are
0.717 and 0.283.
scikit-learn's `predict_proba` does exactly this, in logarithms. Where Naive
Bayes does lose calibration is in the numerator, Section 4.5.

Everything hard is in
$P(x \mid y = c)$: **the probability of this exact combination of readings among
examples of that class.**

With two features that is a two-dimensional density and we could estimate it.
With twenty it is a twenty-dimensional one, and no quantity of data populates a
twenty-dimensional space — the curse of Section 3, arriving from an entirely
different direction.

### 4.2 The assumption

> **Given the class, the features are independent of one another.**

If that holds, the joint density factorises:

$$P(x \mid y = c) = \prod_{j=1}^{n} P(x_j \mid y = c)$$

and each factor is a one-dimensional density estimated from the rows of that
class. The classifier is then

$$\hat{y} = \arg\max_c \; P(y = c) \prod_{j=1}^{n} P(x_j \mid y = c)$$

In practice that product is never computed: hundreds of small probabilities
multiplied together underflow to zero in floating point, exactly as lesson 4's
log-likelihood did. The repair is available because the logarithm is
**increasing** — whichever class makes the product largest makes its logarithm
largest too, so swapping one for the other cannot change which class wins — and
because the logarithm of a product is a sum of logarithms:

$$\hat{y} = \arg\max_c \left[ \log P(y = c) + \sum_{j=1}^{n} \log P(x_j \mid y = c) \right]$$

The two expressions pick the same class. The second adds $n + 1$ numbers of
ordinary size instead of multiplying $n + 1$ tiny ones.

**Why it is such a good bargain.** One $n$-dimensional estimation problem
becomes $n$ one-dimensional ones. Training is a single pass computing means and
variances. It needs very little data per feature, and adding features costs
almost nothing.

**Why it is almost never true.** Vibration and pressure are both driven by the
operating point. "New" is not independent of "York" given the topic. Symptoms
co-occur.

The interesting question is therefore not whether the assumption holds, but
**when being wrong about it costs you nothing**.

### 4.3 On the pumps, the assumption happens to hold

| Model | Accuracy |
|---|---|
| Majority baseline | 0.613 |
| Logistic regression | 0.613 |
| **Gaussian Naive Bayes** | **0.933** |
| k-NN, $k = 5$ | 0.944 |

A point behind k-NN, and an enormous distance ahead of the linear model. So test
the assumption directly — remembering that it concerns independence *given the
class*, which is not the same as independence overall:

| Correlation between the two readings | |
|---|---|
| overall | −0.046 |
| within healthy pumps | **−0.006** |
| within faulty pumps | **−0.049** |

**Within each class the readings are essentially uncorrelated.** Naive Bayes is
doing well here because its assumption is true here — which is far more useful
to know than the score, because it tells you when to expect the score to hold.

### 4.4 One step away, worse than guessing

Now a second pair of sensors on the same fleet. The pump is faulty when
**exactly one** of the two readings is high: both high is the designed
high-load mode, both low is idle, and one without the other is a mismatch
between demand and delivery.

![](interaction_marginals.png)

*Left: together, the two sensors show four clear groups and a perfectly
learnable rule. Middle and right: what Naive Bayes gets to see — each sensor
alone, where the two classes sit almost exactly on top of one another.*

| Model | Accuracy |
|---|---|
| Majority baseline | 0.523 |
| **Gaussian Naive Bayes** | **0.404** |
| Logistic regression | 0.393 |
| SVM, linear kernel | 0.606 |
| k-NN, $k = 5$ | 0.967 |
| SVM, RBF kernel | 0.972 |

**0.404 — below chance, and well below the majority baseline.**

Being below chance looks impossible, and the explanation is worth having. The
class means on each sensor differ by about 0.17 against a spread near 1, purely
as an artefact of a finite sample. That accident is the *only* per-feature
evidence available, Naive Bayes has nothing else to multiply, and in this sample
it points the wrong way.

**A model with no signal does not sit politely at 50%.** It follows whatever
spurious structure it can find.

### 4.5 Its probabilities are not probabilities

Measured on the interacting data: mean confidence when correct **0.567**, mean
confidence when wrong **0.555**. It cannot tell the difference.

The more common complaint runs the other way, and it is not measured on this
dataset — it needs correlated features, which the pump data does not have. When
features *are* correlated, multiplying their probabilities counts the same
evidence repeatedly, and Naive Bayes becomes wildly overconfident. Ten near-copies
of one informative column is enough: accuracy around 0.75, and two thirds of the
predictions asserted above 0.999. Try it — it takes four lines.

Either way, lesson 5's distinction applies: **the ranking may be useful while
the probabilities are not.** Naive Bayes is the classic example of that gap, and
the reason it should not be used where a calibrated probability is needed — such
as the cost calculation of lesson 4, Section 7.2.

### 4.6 When to use it anyway

**Very high dimensions with little data.** Text classification is canonical:
tens of thousands of word-count features, an assumption that is transparently
false, and a method that works anyway — because being roughly right in ten
thousand dimensions beats being unable to estimate anything at all.

**As a baseline.** It trains in one pass. If your tuned model does not beat
Naive Bayes, you have learned something quickly and cheaply.

**Not** when the signal is an interaction. No quantity of data repairs Section
4.4, because the model cannot represent what you are asking of it.

---

## 5. Support vector machines

### 5.1 The margin: a different criterion

On separable data there are infinitely many separating lines, and **every one of
them has zero training error**. Minimising the error therefore cannot choose
between them; logistic regression breaks the tie with log loss, which is one
answer among several.

The support vector machine uses a different one: choose the boundary with the
**widest margin**. Push a slab out from the boundary until it touches the
nearest point of each class, and pick the boundary that makes the slab thickest.

**The intuition.** A boundary passing close to a training point is one small
perturbation away from getting it wrong. Maximising the distance to the closest
points chooses the boundary that tolerates the most movement in the data before
it changes its mind. That is a statement about generalisation, not about fit.

![](svm_margin.png)

*The solid line is the boundary, the dashed lines the edges of the slab, and the
circled points the support vectors touching it. Of eighty points, three
determine the answer; move any of the others and nothing changes.*

### 5.2 The optimisation, and the soft margin

Write the boundary as the set of points where $w^\top x + b = 0$, and the labels
as $y_i \in \{-1, +1\}$ rather than $\{0, 1\}$ — a relabelling with no content,
chosen because it lets "correctly classified" be written as a single product
being positive: $y_i(w^\top x_i + b) > 0$ says the sign of the prediction and the
sign of the label agree.

Turning "widest slab" into something a computer can minimise takes three steps.

**Step 1 — how far a point is from the boundary.** First, $w$ is perpendicular to
the boundary: if $x$ and $x'$ both lie on it then subtracting the two equations
gives $w^\top(x - x') = 0$, so $w$ is orthogonal to every direction that runs
along the boundary. To measure how far a point sits from it, then, project onto
the unit vector $w / \lVert w \rVert$ pointing across it:

$$\text{distance}(x) = \frac{\lvert w^\top x + b \rvert}{\lVert w \rVert}$$

The numerator by itself is not a distance. Multiply $w$ and $b$ both by ten and
it grows tenfold while the boundary — the place where the expression is zero —
has not moved a millimetre. Dividing by $\lVert w \rVert$ is what removes that
freedom.

**Step 2 — spend the freedom on purpose.** That same freedom is a nuisance in an
optimisation: infinitely many pairs $(w, b)$ describe one boundary, so the
minimum would not be unique. Fix it by a choice — rescale $(w, b)$ so that the
training points closest to the boundary satisfy
$\lvert w^\top x_i + b \rvert = 1$ exactly. Any separating hyperplane can be
written this way, so nothing is given up, and two things are bought:

- **The constraints become clean.** Every point correctly outside the slab now
  reads $y_i(w^\top x_i + b) \geq 1$, where the 1 means "at least as far out as
  the closest points are".
- **The margin becomes a formula.** Substituting $\lvert w^\top x_i + b \rvert = 1$
  into Step 1, the closest points sit at distance $1 / \lVert w \rVert$, and the
  slab reaches that far on both sides:

$$\text{margin} = \frac{2}{\lVert w \rVert}$$

**That is where the $2 / \lVert w \rVert$ comes from**, and it is worth noticing
what has happened: with the scale pinned down, the width of the slab is decided
by $\lVert w \rVert$ and by nothing else. A short $w$ is a wide margin.

**Step 3 — maximise by minimising.** Maximising $2 / \lVert w \rVert$ is
minimising $\lVert w \rVert$, and minimising $\lVert w \rVert$ is minimising
$\tfrac{1}{2}\lVert w \rVert^2$ — the same minimiser, because squaring is
increasing on non-negative numbers, with the square and the $\tfrac{1}{2}$ there
only so that the derivative comes out as $w$. So:

$$\min_{w, b} \ \tfrac{1}{2}\lVert w \rVert^2
  \quad \text{subject to} \quad y_i\left(w^\top x_i + b\right) \geq 1 \ \ \forall i$$

A convex objective with linear constraints, which is the easiest kind of
constrained problem there is: one minimum, and no local ones to get stuck in.

Real data is not separable — ours has 4% of its labels flipped — and this
problem then has no solution at all. The fix is to allow violations $\xi_i$ and
charge for them:

$$\min_{w, b, \xi} \ \tfrac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{m} \xi_i
  \quad \text{subject to} \quad y_i\left(w^\top x_i + b\right) \geq 1 - \xi_i, \ \ \xi_i \geq 0$$

**$C$ is the price of a training error.** Large $C$ makes violations expensive,
so the model contorts to classify everything: narrow margin, low bias, high
variance. Small $C$ buys a wider, calmer boundary at the cost of some errors.

It is the same dial as $k$ in Section 2 and $\lambda$ in lesson 3, in a third
costume — and note the direction, which catches people out: **large $C$ means
less regularisation.**

### 5.3 On the pumps, the margin alone does not help

| Model | Accuracy | Support vectors |
|---|---|---|
| SVM, linear kernel | **0.613 ± 0.000** | 947 of 1,200 (**79%**) |
| SVM, RBF kernel | 0.947 ± 0.005 | 278 of 1,200 (23%) |

The linear kernel scores the base rate, exactly as logistic regression did.
Choosing the best straight line does not help when no straight line works.

The support-vector counts say the same thing in another language. The linear
model needs 79% of the training set, because almost every point sits on or
inside the margin — there is no slab that separates anything. **A high
support-vector fraction is a free warning** that the model is struggling to find
room.

### 5.4 The kernel trick

**The picture first.** Our healthy pumps sit in a disc, the faulty ones around
it, and no line separates them *in the plane*. Add a third coordinate — the
distance from the centre — and lift each point to that height. Healthy pumps
rise a little, faulty ones a lot, and a flat horizontal plane separates them
perfectly.

The classes were always separable. They needed different coordinates.

![](kernel_lift.png)

*The same 1,200 pumps twice: in the plane where they were measured, and lifted
by their distance from the design point. The gold plane does what no line
could.*

That is the idea in general: map into a space $\phi(x)$ where a linear boundary
works, and run the linear method there. The obstacle is that useful spaces are
enormous, sometimes infinite-dimensional, and computing $\phi(x)$ would be
impossible.

**The trick is that the SVM never needs $\phi(x)$** — and seeing why requires
knowing the shape of its solution, so here it is. A constrained minimisation is
attacked by attaching a multiplier to each constraint, one price per training
point, charged when that point pushes against its own constraint. Writing $a_i$
for the multiplier on point $i$ — most texts write $\alpha_i$, but this course
reserves $\alpha$ for the learning rate — Section 5.2's problem becomes

$$\mathcal{L}(w, b, a) = \tfrac{1}{2}\lVert w \rVert^2
  - \sum_{i=1}^{m} a_i\left[y_i\left(w^\top x_i + b\right) - 1\right]$$

to be minimised over $w$ and $b$ and maximised over the $a_i \geq 0$. Setting the
two derivatives to zero:

$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i} a_i y_i x_i = 0
  \quad \Longrightarrow \quad w = \sum_{i} a_i y_i x_i,
  \qquad \frac{\partial \mathcal{L}}{\partial b} = -\sum_{i} a_i y_i = 0$$

**The first of those is the whole story: the best $w$ is a weighted sum of the
training points themselves.** Substitute it back into $\mathcal{L}$, using
$\sum_i a_i y_i = 0$ to delete the $b$ term, and what is left contains the data
in exactly one form:

$$\max_{a} \ \sum_{i} a_i
  - \tfrac{1}{2}\sum_{i}\sum_{j} a_i a_j y_i y_j \langle x_i, x_j \rangle
  \quad \text{subject to} \quad a_i \geq 0, \ \ \sum_i a_i y_i = 0$$

The soft margin changes one thing: violations bounded by $C$ put a ceiling
$a_i \leq C$ on each multiplier, which is the sense in which $C$ really is the
price of a point.

Two facts drop out, and both are things this lesson has already been using.

**Support vectors are the model.** A multiplier is a price paid only where the
constraint binds. Points sitting strictly outside the slab have $a_i = 0$ and
vanish from every sum above — which is why, in Section 5.1's figure, three points
of eighty decide the answer and the other seventy-seven can be moved freely.

**Prediction needs inner products too, and nothing else.** Substituting $w$ into
$w^\top x + b$:

$$w^\top x + b = \sum_{i} a_i y_i \langle x_i, x \rangle + b$$

So neither fitting nor predicting ever touches a coordinate except inside
$\langle \cdot, \cdot \rangle$. Replace every one of those with a function that
returns the inner product *in the new space* while computing only with the
original coordinates, and the machine is running in $\phi$'s space without ever
visiting it:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

The standard choice, and scikit-learn's default, is the **radial basis
function**:

$$K(x, x') = \exp\left(-\gamma \lVert x - x' \rVert^2\right)$$

**Why that one is infinite-dimensional**, rather than merely large. Expand the
squared norm as $\lVert x \rVert^2 - 2\langle x, x' \rangle + \lVert x' \rVert^2$
and the kernel splits into a factor for each point and one joint factor:

$$K(x, x') = e^{-\gamma \lVert x \rVert^2}\, e^{-\gamma \lVert x' \rVert^2}\,
  e^{2\gamma \langle x, x' \rangle}$$

and the joint factor is an exponential, whose power series never terminates:

$$e^{2\gamma \langle x, x' \rangle}
  = \sum_{k=0}^{\infty} \frac{(2\gamma)^k}{k!} \langle x, x' \rangle^k$$

Each term is a polynomial kernel: expanded, $\langle x, x' \rangle^k$ is an inner
product between the vectors of all degree-$k$ monomials in the coordinates, each
carrying a fixed weight. The sum therefore has monomials of *every* degree, so
$\phi(x)$ has infinitely many entries — and one exponential per pair buys all of
them.

We chose the lift in the picture above by knowing the answer; the RBF kernel does
something equivalent without being told, which is why it works where nobody could
guess the right coordinates.

### 5.5 Gamma, C, and overfitting you can see

$\gamma$ sets how far a single training point's influence reaches. Small
$\gamma$: wide reach, smooth boundary. Large $\gamma$: each point influences
only its immediate neighbourhood, and the boundary can dissolve into islands.

![](svm_gamma_c.png)

*Three settings, with the training and cross-validated scores in each title.
Read them together, as lesson 5 taught.*

| Setting | Training | Cross-validated |
|---|---|---|
| $\gamma = 0.1$, $C = 1$ | 0.936 | 0.929 |
| $\gamma = 1$, $C = 1$ | 0.950 | **0.944** |
| $\gamma = 50$, $C = 1000$ | **0.995** | 0.902 |

The last row has the **highest training score and the lowest honest one** — the
signature of overfitting, and here it is visible as well as measurable: the
boundary has broken into bubbles around individual points, including the
mislabelled ones.

Choosing between these three on the training score would select the worst model
with complete confidence.

---

## 6. The three compared

Everything in this lesson on the same 1,200 pumps, cross-validated:

| Model | Accuracy |
|---|---|
| Majority baseline | 0.613 |
| Logistic regression (lesson 4) | 0.613 |
| SVM, linear kernel | 0.613 |
| Gaussian Naive Bayes | 0.933 |
| k-NN, $k = 5$ | 0.944 |
| SVM, RBF kernel | **0.947** |
| *noise ceiling* | *≈ 0.96* |

Three methods reach the ceiling by three unrelated routes — remembering the
neighbourhood, assuming independence, bending the space — and two do not, both
of them linear.

**The gap between best and worst is 0.334**, larger than any difference this
course has shown between a good model and a tuned one. That is the lesson of the
table: **choosing the right family matters far more than tuning the wrong one**,
and the way to tell which family you need is to look at the data first, as
lesson 2 insisted.

### How to choose, in practice

| Situation | Reach for |
|---|---|
| Few features, plenty of data, odd-shaped boundary | k-NN, or an RBF SVM |
| Very many features, little data | Naive Bayes |
| Need a calibrated probability | Not Naive Bayes |
| Need the model to be small, or fast at prediction | SVM, not k-NN |
| Signal is an interaction between features | k-NN or a kernel; not Naive Bayes |
| Need to explain the decision to the person affected | Neither — lesson 3's Resources |

---

## 7. Summary

- **No straight line separates the pumps**, and both linear models score exactly
  the base rate, 0.613, with zero variance across folds.
- **k-NN learns nothing and scores 0.944.** Scale first; distance is all it has.
- **$k$ is the bias-variance dial**: training accuracy is exactly 1.000 at
  $k = 1$ and means nothing.
- **The curse of dimensionality is about distance, not difficulty.** Fifty noise
  columns took k-NN below the baseline. In 100 dimensions the nearest point is
  **70%** as far away as the farthest.
- **Naive Bayes assumes independence given the class.** On the pumps that holds
  (−0.006 within class) and it scores 0.933; where the signal is an interaction
  it scores **0.404**, below chance.
- **Its probabilities are not calibrated**, in either direction.
- **The margin is a criterion distinct from the error**, and support vectors are
  the model. A high support-vector fraction is a free warning.
- **The kernel trick separates by changing coordinates**, through inner products
  in a space never computed.
- **Choosing the family beats tuning the wrong one**: 0.613 to 0.947 on
  identical data.

### Homework

`Exercises/06_knn_naive_bayes_svm.md`, discussed at the start of Lesson 7, **Friday 13 November 2026**.

### Notation used in this lesson

| Symbol | Meaning |
|---|---|
| $x$, $x'$ | two feature vectors |
| $d(x, x')$ | Euclidean distance between them |
| $k$ | number of neighbours voting |
| $m$, $n$ | number of examples, number of features |
| $\mu$, $v$ | mean and variance of one squared coordinate difference |
| $w$, $b$ | the hyperplane's coefficients and intercept |
| $\xi_i$ | how far example $i$ violates the margin |
| $C$ | the price of a margin violation |
| $\gamma$ | how far one point's influence reaches, in an RBF kernel |
| $a_i$ | the multiplier on example $i$'s margin constraint (usually written $\alpha_i$) |
| $\phi$ | the map into the higher-dimensional space |
| $K$ | the kernel, an inner product in that space |

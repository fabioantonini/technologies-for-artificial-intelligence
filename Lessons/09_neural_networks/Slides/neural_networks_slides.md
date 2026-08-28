---
title: "Lesson 9: Neural Networks"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "20 November 2026"
---

# Agenda

- A boundary whose **shape** is learned, not chosen in advance
- One neuron is one line, and exactly where that runs out
- Forward propagation, and backpropagation derived right to left
- Softmax, activations, and the vanishing gradient
- Initialisation, optimisers, regularisation: the failure modes

::: notes
Frame the whole three hours before any content. Every model this course has
built so far drew a boundary whose shape was fixed before the data arrived:
logistic regression a straight line, a decision tree axis-parallel boxes, a
kernel machine a curve out of a family somebody chose when they picked the
kernel. Today the shape itself is learned, and that is the single idea of the
lesson.

Say what the lesson deliberately is not: it is not a survey of architectures,
and there is no result here that needs a graphics card. Three hours is enough
to derive backpropagation properly, implement it from scratch, and meet the
handful of failure modes that account for most of the time anyone ever spends
debugging a network. That is a better use of the time than a tour.

Two of today's three datasets come from a fictional sensor maker, Meridian
Instruments; the third is real handwritten digits. Handout section 1.
:::

# Exercise 8 returned

- Marks were for **what the metric could see**, not for the score itself
- Recurring gap: a silhouette reported as endorsement of a clustering, when
  it can only endorse roundness
- Today the same obligation returns in a new form: a network will reach a
  high training accuracy very quickly, and that number on its own says
  almost nothing

::: notes
Hand back exercise 8 briefly. One sentence on what went well, one on the
recurring gap - several reports quoted a silhouette score as though it
validated the clustering, which is precisely the blind spot section 4 of that
lesson was about.

Link forward deliberately. Today's methods are the easiest in the whole course
to make *look* like they work: a network with enough units will drive its
training loss to essentially zero on any dataset you hand it, including one
with random labels. Everything in this lesson that reads as caution - the
seed-to-seed spreads in every table, the ceiling of 0.97, the selection-bias
number in section 2 - is there because of that.
:::

# What a network adds

- Boundaries so far had a shape **fixed in advance**
- A line; axis-parallel boxes; a curve from a chosen kernel
- A network learns the **shape of the boundary itself**
- Everything else today makes that possible, or stops it going wrong
- No architecture tour, no hardware, no network call

::: notes
This is the one-sentence thesis of the lesson and it is worth saying slowly,
because students often arrive believing that what makes a network special is
its size. It is not. It is that the family of boundaries is not fixed before
training starts.

Put the question to the room: given lesson 6's kernel machines, which also
draw curved boundaries, what is actually different here? The answer is who
chose the family - with a kernel you did, in advance, and if you chose badly
no amount of data repairs it; with a hidden layer the data chooses.

Handout section 1.
:::

# Three problems, and a ceiling of 0.97

- **Acceptance test**: two axes, one pass/fail verdict
- **Two-channel drift**: the exclusive-or (XOR) function
- **Handwritten digits**: 1,797 images, ten classes, real
- The rig is **wrong on 3% of units**, so every score today is read against a
  ceiling of **0.97**
- A line scores **0.55** on acceptance and **0.93** on digits

::: notes
Introduce all three problems at once, because the contrast between them is
what stops this lesson from being propaganda for neural networks. On the
acceptance data a linear model scores 0.55 and a network 0.94 - a 39-point
gap. On the digits a linear model scores 0.93 and the best network here 0.97  - 
three and a half points. A lesson that only showed problems where networks win
would teach the opposite of what is true.

The 3% rig error is the number to keep visible all afternoon. The two
synthetic datasets publish their generating rules as `TRUE_*` constants in
`Notebooks/instrument_data.py`, which is why this lesson can quote both what a
model scores and what it actually learned - a comparison Meridian's own
engineers could never make.

Ask the room: if the measuring instrument is wrong 3% of the time, what is the
best accuracy any classifier can be *scored* at? Handout section 1.1.
:::

# The perceptron, and a unit you already have

- Rosenblatt, 1958: weights, a bias, "output 1 if the sum is positive"
- Swap the threshold for a sigmoid: lesson 4's logistic unit, unchanged
- What is new is one word: **component**
- Undecided at 0.5 means the weighted sum is zero
- A line in two dimensions, a hyperplane in general

::: notes
Nothing on this slide is new mathematics, and saying so out loud is the point:
students should feel that they already own the building block, so that the
only genuinely new idea today is the stacking.

Land the last two bullets carefully, because everything in the next twenty
minutes depends on them. Everything a single neuron can express is "which side
of this line are you on, and how far". Not "is this point unusual", not "is
this point inside a region" - one side of one line.

Handout section 2.1.
:::

# A neuron is a line

$$\hat{y} = \sigma(w^{\top}x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

::: notes
Read it out in words rather than in symbols: take a weighted sum of the
inputs, add a bias, squash it into the interval from zero to one. The squashing
does not change which side of the line a point is on - the sigmoid is monotone
 -  so the boundary is still exactly the straight line where the weighted sum
vanishes.

The whole of the next segment is an attempt to solve, with this object, a
problem that a line cannot solve. Handout section 2.1.
:::

# Meridian's acceptance test

- Two calibration axes: a gain offset in decibels, a phase offset in degrees
- It passes when both offsets are **jointly** small enough for one global
  firmware correction
- 2,250 training sensors, 3,000 generated in all
- So the accept region is the inside of a **closed curve**
- And no straight line encloses anything

::: notes
Describe the physical situation before any picture: this is a real shape of
industrial problem, not a toy built to embarrass linear models. The firmware
applies one correction, so a unit is usable when the pair of offsets is small
together, not when either one is small on its own.

Ask the room to predict, before the next slide, what the accept region will
look like once each axis is divided by its own production spread. Most will
say "a box", because two tolerances sounds like two independent thresholds.
The answer is a circle, and the reason is on the slide after next.

Handout section 1.1 and section 2.2.
:::

# 2,250 sensors, raw and standardised

![](acceptance_data.png)

::: notes
Left panel: the raw measurements in decibels and degrees, which is all
Meridian's engineers ever see. Right panel: the same points after each axis is
divided by its own production spread, with the rule that generated the labels
drawn on top.

Point directly at the wrong-coloured points scattered along the circle. Those
are not model errors - nothing has been fitted yet. They are the 3% of
verdicts the test rig recorded incorrectly, and they are the reason no score
today can exceed 0.97.

Let the room look at the right panel and ask the obvious question out loud:
where would you put a straight line? Handout section 2.2.
:::

# A circle of radius 1.25

- Gain tolerance **0.50**, production spread **0.40**
- Phase tolerance **3.75**, production spread **3.00**
- Both are exactly **1.25 spreads**, so standardised, the accept region is a
  circle of radius 1.25
- Two units, two tolerances, one shared shape
- **0.5497** of the 3,000 generated units fall inside it

::: notes
This is the payoff for the "try this" in the handout: the circle is not an
arbitrary teaching choice, it falls out of the two tolerances being the same
multiple of their own axis's spread. Standardising is what makes that
visible - in decibels and degrees the region is an ellipse and the coincidence
is invisible.

The measured 0.5497 is worth stating before the theoretical number on the next
slide, so that the comparison runs from data to formula rather than the other
way round.

Handout section 2.2.
:::

# What fraction of units the circle holds

$$P(\|z\| < 1.25) = 1 - e^{-1.25^{2}/2} = 1 - e^{-0.78125} = 0.5422$$

::: notes
For a standard two-dimensional normal the probability of landing inside a
circle of radius s is one minus e to the minus s squared over two - a fact
worth a moment, since it is the same integral that reappears in section 11
when the lesson computes how well a polygon of a given number of sides could
fence this region.

0.5422 predicted, 0.5497 measured on 3,000 units. The gap is sampling noise on
three thousand draws and nothing else.

`Docs/worked_examples.py` recomputes this from the raw tolerances rather than
from the standardised radius. Handout section 2.2.
:::

# What one line is worth

| | test accuracy |
|---|---|
| always predict "accepted" | 0.5480 |
| fitted logistic regression | **0.5507** |
| the best straight line for this population | 0.6491 |
| the rig's ceiling | 0.9700 |

::: notes
Read the table from the top. A model that ignores its inputs entirely scores
0.5480. Logistic regression, fitted properly on 2,250 examples, scores
0.5507 - 27 ten-thousandths better than knowing nothing.

Then read the gap at the bottom, which is the real content: even the best line
that exists for this population reaches 0.6491, against a reachable 0.97.
Three points of the first gap are the fit; thirty-two points of the second are
the shape of the model family.

Ask the room whether they would suspect a bug if they saw 0.5507 come out of
their own experiment. Most would, and they would be wrong. Handout section 2.2.
:::

# No line encloses it

![](linear_boundary_fails.png)

::: notes
Solid line: the fitted logistic boundary. Dashed: the best line found by
brute-force search over 72,762 candidates.

The thing to point at is what the *best* line does, because it is not what
students expect. It does not attempt to enclose the accept region - it cannot
 -  so it slices off one far tail of the distribution where almost every unit is
a reject, and banks that. That is the honest ceiling for the family.

Handout section 2.2.
:::

# The fit did not fail

- Coefficients **(0.0096, 0.0701)**, intercept **0.1892**
- Almost exactly the constant model, and the correct answer to the question a
  line can ask
- Not an optimiser that gave up. The reason is symmetry
- The accept region is a disc on the origin; every accepted unit has a partner
  opposite
- Any tilt gaining accuracy on one side gives it back on the other

::: notes
This slide exists to head off the single most likely wrong diagnosis. A
student who sees 0.5507 will reach for the optimiser: more iterations, a
different solver, a smaller learning rate, scaling. None of it will help, and
the reasoning behind reaching for it is perfectly sound - a score at chance
usually does mean something failed to converge.

Walk the symmetry argument out loud, because it is a proof and it takes ten
seconds: rotate the picture by 180 degrees and the labels are statistically
unchanged, while any line becomes a different line. A quantity that must be
invariant under a symmetry the model cannot represent has nowhere to go.

Handout section 2.2.
:::

# 72,762 candidate lines, and 4 points of self-deception

- The search tries 72,762 boundaries and scores the winner **on the data that
  chose it**: 0.6880
- The honest figure for the population, with no test set: **0.6491**
- **Almost 4 points of pure selection bias**
- Nothing was fitted. No parameter estimated. Only a choice made
- Lesson 5's argument, inside a lesson-9 experiment

::: notes
This is a deliberate ambush and it is worth pausing on. The brute-force search
has none of the features students associate with overfitting: no parameters,
no training, no capacity. It picks the best of a list. And picking the best of
a long list, then reporting that best, is enough on its own to inflate a score
by four points.

Ask the room how many candidates they would have to try before the effect
becomes serious, and whether they have ever reported the best of several
configurations. Almost everyone has.

The 0.6491 is computed analytically in handout section 11.1, from the geometry
rather than from any sample, which is precisely why it is trustworthy here.
Handout section 2.2.
:::

# The smallest problem that needs a hidden layer

- Same-direction drift: removable. Opposite drift: not
- Correctable when the two drifts **share a sign**
- That is the exclusive-or (XOR) function
- Four clouds of 200 units; a line scores exactly **0.5000**
- Coding the channels as $\pm 1$, the rule is about $|a + b|$
- "Large in absolute value" is the outside of a strip
- **Two boundaries, not one**

::: notes
XOR is the traditional example and it is usually presented as an abstract
logic puzzle. It is worth insisting that this version is a physical situation:
common-mode drift is removable, differential drift is not, and the rule
follows from the electronics rather than from a truth table.

The last two bullets are the whole diagnosis. The problem is genuinely
one-dimensional - everything depends on a single number, the sum of the two
drifts - and it is still not linearly separable, because the question asked of
that number needs two thresholds rather than one. Say that plainly: it is not
about dimension, it is about how many boundaries the answer needs.

Handout section 3.1.
:::

# Two units, written down rather than trained

- Two rectified linear units (ReLU), each max(0, ·) of the same sum
- One fires when a + b > 1, the other when a + b < -1
- Between them they compute $|a + b|$, clipped
- Output weights 10 and 10, bias -1: the rule is $|a + b| > 1.1$
- Over all 800 units, **nothing trained**, it scores **0.9938**
- Chosen by reasoning about the problem, and they work

::: notes
This is the moment the lesson turns. Write the four hidden weights on the
board if there is time - the first weight matrix is a column of ones beside a
column of minus ones, both biases minus one, and the output weights are ten
and ten. The handout works all four cloud centres through the network by hand
in a table; do one row live, the plus-one plus-one corner, and let them check
the rest afterwards.

The rectified linear unit is introduced here as a piece of machinery rather
than as a topic: max of zero and the input, nothing more. Its properties get
their own segment after the break.

The sentence to leave in the room: whatever training does, *this* is the kind
of thing it is searching for. Handout section 3.2.
:::

# What the hidden layer actually did

![](xor_hidden_space.png)

::: notes
Left: the four clouds in the input space, where no line separates the colours.
Right: the same 800 units plotted by what the two hidden units output.

Trace one cloud across the two panels with a finger. The two correctable
clouds, which sat in opposite corners of the input space, have been moved to
(1, 0) and (0, 1); both uncorrectable clouds have been stacked on top of each
other at the origin. In the right-hand panel a single line separates them
trivially.

Ask the room what the hidden layer classified. The answer is nothing - and
that is the next slide. Handout section 3.3.
:::

# The hidden layer classified nothing

- It **moved the data** until the output line was enough
- Correctable clouds to (1,0) and (0,1); the rest to the origin
- The name for this is **representation learning**
- What is learned: coordinates in which the boundary is simple
- Lesson 10's convolutional networks: the same trick, constrained

::: notes
This is the idea to carry out of the first half of the lesson, and it is worth
more than the algebra that follows. Students tend to picture a deep network as
a stack of classifiers, each refining the last. It is not. Every layer but the
final one is changing coordinates; only the last one classifies.

That reading is also what makes the rest of the field intelligible: a
convolutional layer, a recurrent layer, an embedding are all answers to the
question "what coordinates would make this problem easy", constrained in
different ways.

Forward-reference lesson 10 explicitly here, so next week opens on a question
already asked. Handout section 3.3.
:::

# Four lines enclose a region; one never does

![](four_lines_fence.png)

::: notes
Back to the acceptance data, with a trained four-unit hidden layer. Left: the
four lines those units draw, plotted straight from the columns of the first
weight matrix - not inferred from predictions, read off the parameters. Right:
the decision region the four produce together, with the true circle dashed.

This is the picture that makes "a hidden layer of H units is H lines" concrete.
Ask the room how many lines they think are needed before the score stops
improving; the answer, from the sweep in handout section 11.2, is about three,
which surprises most people.

Handout sections 4.1 and 11.2.
:::

# Capacity is not the same as findability

| hidden units | median accuracy | best of 20 | runs above 0.95 |
|---|---|---|---|
| 2 | 0.7500 | 0.9975 | **4 of 20** |
| 3 | 1.0000 | 1.0000 | 15 of 20 |
| 4 | 1.0000 | 1.0000 | 19 of 20 |
| 8 | 1.0000 | 1.0000 | 20 of 20 |

::: notes
Twenty independent restarts at each width, on the drift problem. Two units are
provably enough - the lesson wrote a two-unit solution down by hand three
slides ago and it scored 0.9938 - and plain gradient descent finds something
that good in four runs out of twenty.

The median tells you what goes wrong. 0.7500 is exactly three of the four
clouds: the solution you get by spending both lines carving off a single
quadrant. It is a local minimum, and from most starting points it is downhill.

The consequence is the honest reason production networks are wider than their
task requires. The extra units are not extra capacity - they are extra
starting points, so that some unit begins near a useful line.

Handout section 3.4.
:::

# What each extra line is actually worth

![](width_sweep.png)

::: notes
This is the capacity argument in one picture, and it is worth two minutes even
though the segment is tight. The purple line is the best fence you could build
from that many straight lines - a regular polygon, computed exactly rather
than searched for. The blue line is what training actually reaches.

Three things to point at. Almost the whole gain arrives by three units: two to
three is worth 20 points, three to thirty-two is worth half a point. At three
and four units the trained network is *above* the polygon, because H lines cut
the plane into far more than one polygon and the output unit votes over the
pieces. And past six units the trained curve flattens while the polygon curve
keeps climbing towards the ceiling - the lines are available and gradient
descent does not use them.

Ask the room which of the two curves is a statement about the architecture and
which is a statement about the optimiser. The gap between them, after six
units, is the whole of section 9's subject.

Handout section 11.2.
:::

# Universal approximation, and what it does not promise

- Cybenko (1989), Hornik (1991): one hidden layer, enough units, any
  continuous function
- Read the quantifiers. It says such a network **exists**
- Nothing about how many units "enough" is
- Nothing about whether training will find it
- Nothing about behaviour outside the region
- Two units sufficed on the drift data: descent missed it 16 times in 20

::: notes
This theorem is cited far more often than it is read, and the citation almost
always carries an implication the theorem does not license - that a network is
guaranteed to work if you make it big enough.

Put the question to the room directly: does the theorem say anything at all
about training? It does not mention training. It is a statement about the
existence of a set of weights, proved without reference to how anybody would
obtain them.

Its real content is a licence to stop worrying about expressiveness and start
worrying about optimisation and data, which is exactly what the remaining two
hours do. Handout section 3.5.
:::

# Forward propagation: examples in rows

- One hidden layer of $H$ units, one output unit, $m$ examples, $n$ inputs
- Two matrix multiplies, each followed by an elementwise function
- This course puts examples in **rows**: as scikit-learn and Keras do
- Textbooks often use columns, and every transpose flips if you do

::: notes
The convention warning is not pedantry. It is the single most common source of
wasted hours when a student implements backpropagation from a textbook while
debugging against scikit-learn shapes, and it is worth thirty seconds now to
save an evening later.

State the architecture once so the two equation slides that follow can be read
without narration: inputs, one hidden layer, one output unit, binary
classification. Multiclass comes after the break.

Handout section 4.1.
:::

# Layer 1: a line per unit, then a non-linearity

$$Z^{[1]} = X W^{[1]} + b^{[1]}, \qquad A^{[1]} = g(Z^{[1]})$$

::: notes
Read the bracketed superscript as the layer index, not a power - say it once
and it will not confuse anybody again. Z is the pre-activation, the raw
weighted sum; A is what comes out after the activation function g.

The bias is added by broadcasting across rows: one bias per hidden unit, the
same value for every example. Handout section 4.1.
:::

# Layer 2: one more line, then a probability

$$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}, \qquad \hat{y} = \sigma(Z^{[2]})$$

::: notes
The second layer is exactly the single neuron from the start of the lesson,
with one change: its inputs are no longer the measurements, they are the
hidden layer's outputs. That is the whole architecture.

Say the consequence out loud, because it ties back to the drift example: the
output unit is still just a line. It has not become cleverer. It is being
handed better coordinates. Handout section 4.1.
:::

# The shapes

| symbol | shape | what it is |
|---|---|---|
| $X$ | $m \times n$ | the design matrix, one example per row |
| $W^{[1]}$ | $n \times H$ | one column per hidden unit |
| $b^{[1]}$ | $H$ | broadcast across rows |
| $Z^{[1]}, A^{[1]}$ | $m \times H$ | pre-activations and activations |
| $W^{[2]}$ | $H \times 1$ | the output unit's weights |
| $\hat{y}$ | $m \times 1$ | one prediction per example |

::: notes
Shapes are the cheapest debugging tool in the whole subject, and the habit
worth installing is checking them before running anything. A backward pass
that produces a gradient of the wrong shape has a bug you can see; one of the
right shape may still have a bug, which is why section 5.5 exists.

Ask the room what shape the gradient of the cost with respect to the first
weight matrix must have. The answer - the same shape as the matrix itself - is
the check that catches most transpose errors before they cost anything.

Handout section 4.1.
:::

# Two facts the shapes hand you

- **Examples never mix**: the row index passes through untouched
- Nothing lets example 3 influence example 7, which is what makes
  mini-batching valid
- **Each column of the first weight matrix is one unit**
- Its weight vector, and the line that unit draws
- So $H$ hidden units are $H$ lines, and the output unit votes

::: notes
Both facts look like bookkeeping and both are structural. The first is the
licence for every batching, sharding and parallelism decision anybody makes
later; if examples could influence one another, computing a gradient on 32 of
them would not approximate the gradient on all of them.

The second is what makes a hidden layer interpretable at all in two
dimensions, and it is the bridge to the fence picture and to the polygon
yardstick in handout section 11. Put the question to the room: if a layer of
H units is H lines, how many lines does it take to fence a circle well? Answer
after the break, in the notebook.

Handout section 4.1.
:::

# An untrained network costs 0.6721; guessing costs 0.6931

$$J = -\frac{1}{m}\sum_{i=1}^{m}\left[ y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

::: notes
Cross-entropy, unchanged from lesson 4 - say that explicitly, because students
expect a new cost to come with a new model and it does not.

The two numbers in the title are the sanity check to run before training
anything. A network at random initialisation should cost about log 2, roughly
0.693, on a balanced binary problem: it knows essentially nothing, and the
cost says so. 0.6721 is that, slightly better than pure ignorance because the
classes are not exactly balanced.

A network whose initial cost is far from this has a bug, usually in
initialisation scale or in the labels, and it is worth ten seconds to check
before spending an hour on the training curve. Handout section 4.1.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. What comes back after the break is the derivation that makes
all of this trainable - and the four-line check that tells you whether you got
it right. Worth saying so before they leave the room.
:::

# Two bad ways to get every gradient

- We need the derivative of the cost for **every** weight and bias
- **Perturb and re-run**: one forward pass per parameter, 301,066 of them,
  for one step
- **Differentiate symbolically**: the same sub-expressions, thousands of times
- Both correct. Both unusable

::: notes
Set the problem up as a cost question rather than a calculus question, because
that is what it is. The chain rule was never in doubt; what was in doubt for
two decades was whether the gradient could be obtained cheaply enough to take
millions of steps with it.

Say the 301,066 figure slowly - it is a real network from later in this lesson,
two hidden layers of 512 units on eight-by-eight digit images - and let the
absurdity land: three hundred thousand forward passes to take one step, and
you would need thousands of steps.

Handout section 5.1.
:::

# Compute each repeated piece once, right to left

- Carry the derivative of the cost with respect to a layer's
  **pre-activations**
- Given it for one layer: that layer's weight gradients, **and** the layer
  below
- One backward sweep produces every gradient in the network
- **A weight's gradient is how wrong the layer above was, times what this
  weight contributed**

::: notes
The fourth bullet is the sentence students should be able to repeat to a
friend without writing anything down, and it is worth asking someone to do
exactly that before moving on. Both halves matter: how wrong the layer above
was, and how much this particular weight contributed. A weight attached to a
feature that was zero for every example gets no gradient no matter how wrong
the output was - which is precisely the dead-unit failure later in the lesson.

Say why the right-to-left order is forced rather than chosen: the error is
only known at the output, so information has to travel from there back towards
the inputs, and the repeated sub-expressions are exactly what accumulates
along the way.

Handout section 5.1.
:::

# The output layer collapses to prediction minus truth

$$\delta^{[2]} = \frac{1}{m}\left(\hat{y} - y\right)$$

::: notes
Say what this is before saying why it is remarkable: it is the derivative of
the cost with respect to the output layer's pre-activation, one number per
example, and it is the seed of the entire backward pass.

Then say why it is remarkable. Two messy derivatives went into it - the
derivative of cross-entropy, which has a fraction with the prediction in the
denominator, and the derivative of the sigmoid, which is a product - and what
came out is a subtraction. Prediction minus truth, scaled by the number of
examples. That is not luck, and the next slide says why.

The full derivation, both factors written out, is handout section 5.2.
:::

# Why sigmoid and cross-entropy belong together
- The loss derivative carries a factor 1 over ŷ(1 − ŷ)
- The sigmoid's derivative **is** ŷ(1 − ŷ)
- They cancel exactly, leaving **ŷ − y**
- Squared error keeps that factor, and it is near zero exactly when the
  network is **confidently wrong**

::: notes
This is the slide that turns a pairing students have seen presented as a
convention into a piece of engineering. Ask the room what happens, under
squared error, to an example the network predicts at 0.999 when the true label
is 0. The prediction is as wrong as it can be; the sigmoid derivative there is
about 0.001; so the gradient is about a thousand times smaller than it should
be, and the network barely learns from its worst mistake.

Cross-entropy removes the factor, and the confidently wrong example gets a
gradient proportional to how wrong it is. Same idea, one dimension up, is what
makes softmax work after the break.

Handout section 5.2.
:::

# One step down, through the activation
- Weight gradient = incoming activations, transposed, times the signal above
- To step down: multiply by the weight matrix above, then by the activation's
  derivative
- For the ReLU that derivative is an indicator: 1 where the pre-activation
  was positive
- A unit that contributed nothing forward passes nothing backward

::: notes
The two multiplications on this slide are the entire algorithm, repeated once
per layer. Emphasise that only one of them involves the data at all - the
other is the same weight matrix used in the forward pass, transposed. That
symmetry is why the backward pass costs about what the forward pass costs.

The ReLU indicator is worth a beat because it comes back twice: it is why the
rectified linear unit does not shrink gradients the way a sigmoid does, and it
is why a unit can die permanently. Both are in the segment after notebook 02.

Handout section 5.3.
:::

# Backward through one layer

$$\delta^{[1]} = \left(\delta^{[2]} (W^{[2]})^{\top}\right) \odot g'(Z^{[1]})$$

::: notes
Read the two operations in order. First the matrix product with the transposed
weights above, which redistributes the error at the layer above across the
units that fed it. Then the circled dot - the elementwise product - with the
activation's derivative, which asks each unit how much it was actually
responding when the example came through.

This one line is the recursion. Written for a general layer index it is
identical, which is why depth costs nothing conceptually and everything
numerically. That is the subject of the vanishing-gradient segment.

Handout section 5.3.
:::

# What backpropagation costs
- The backward pass does about the same arithmetic as the forward pass
- **A gradient costs roughly what a prediction costs**: this is what makes
  training feasible at all
- But every layer's activations must be **stored**
- Memory grows with depth times batch size, and memory, not arithmetic, is
  what usually limits batch size

::: notes
Both halves of this slide get quoted at students later in their careers and
neither is obvious, so state them plainly.

The first is why the field exists: if a gradient cost a hundred times a
prediction, nothing would have been trainable in 1986 and very little would be
trainable now.

The second is why anybody has ever had to think about batch size at all. Ask
the room what they would expect to run out of first when training a deep
network, time or memory. Most say time. The answer in practice is memory, and
the reason is on this slide - every activation from the forward pass has to be
kept alive until the backward pass consumes it.

Handout section 5.4.
:::

# Check the gradient before you trust it
- A wrong gradient **raises no exception**: it trains badly and looks like a
  modelling problem
- Compare each partial against a central difference: error of order h²
- Notebook 01: worst disagreement 1.97 × 10⁻⁸, median 7.83 × 10⁻¹¹
- **Below 10⁻⁶ believe it; above 10⁻⁴ there is a bug**
- Four lines of code, and not optional

::: notes
This is the highest-value slide in the lesson measured in hours saved, and it
is worth saying why in the strongest terms available: a wrong gradient is the
one bug in machine learning that produces no error message, no warning, and a
plausible-looking training curve.

Ask the room how they would notice. The honest answer is that without this
check they would not - they would conclude the architecture was wrong, or the
learning rate, or the data, and they would be looking in the wrong place for
as long as they had patience.

Run it once, on a small network with a handful of examples, whenever a
backward pass is written by hand. Handout section 5.5.
:::

# 21 partial derivatives, on the diagonal

![](gradient_check.png)

::: notes
Every one of the 21 partial derivatives of a small network: the analytic
gradient on one axis, the finite-difference estimate on the other. The points
lie on the diagonal to eight decimal places.

Say what a failure would look like on this plot, because that is what students
will actually meet: a bug rarely displaces every point. It displaces one
parameter block - all the biases, say, or one weight matrix - so the picture
shows a tight diagonal with a handful of points visibly off it, and the ones
off it name the layer with the mistake.

Handout section 5.5.
:::

# Notebook 1, live

- Backpropagation written from scratch, and gradient-checked before it is
  trusted with anything
- The two-unit hand-built network, then the same problem trained from 20
  random starts at each width
- The polygon yardstick: how well could $H$ lines possibly fence a circle?
- The one initialisation that cannot work, and the exact cost it converges to

::: notes
Run `Notebooks/01_backpropagation_from_scratch.ipynb`. Twenty-two minutes.

The cell worth protecting if time runs short is the gradient check - watching
the worst relative disagreement print as ten to the minus eight lands the
previous slide far harder than reading it does.

Second priority is the 20-restart sweep at width 2, because seeing sixteen of
twenty runs stall at 0.75 is what makes "representable is not findable" a fact
rather than a slogan.

Have them predict the sign of the disagreement before it prints, and have them
run the sweep with their own seed rather than the notebook's.
:::

# Ten classes need ten outputs
- **Softmax** turns K scores into a probability distribution
- For K = 2 it reduces to the sigmoid: one construction, two sizes
- The loss is categorical cross-entropy: minus the log probability given to
  the true class
- The gradient at the output is again **ŷ − y**, so everything derived before
  the break applies unchanged

::: notes
The last bullet is the point of the slide, and it is worth deriving on the
board if there is appetite - it is four lines, using only that the one-hot
target sums to one. The full version is handout section 6.2.

Say why the coincidence is not a coincidence: sigmoid-with-cross-entropy and
softmax-with-cross-entropy are the same construction for two classes and for
K classes, and in both the loss was chosen as the one whose derivative cancels
the output non-linearity's. If you ever meet a third output non-linearity, the
matching loss is the one that does the same job.

The max-subtraction trick is a one-line implementation detail that students
will hit for real the first time they write softmax themselves.

Handout section 6.1.
:::

# Softmax

$$\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

::: notes
Read it in words: exponentiate every score, then divide by the total. The
exponential makes everything positive; the division makes it sum to one.
Monotone in each score, so the largest score still wins - softmax changes the
confidences, never the ranking.

Handout section 6.1.
:::

# A hidden layer is not always the answer

| architecture | parameters | validation accuracy | sd |
|---|---|---|---|
| softmax alone, no hidden layer | 650 | 0.9324 | 0.0035 |
| one hidden layer of 32 | 2,410 | 0.9602 | 0.0052 |
| one hidden layer of 64 | 4,810 | 0.9611 | 0.0045 |
| two hidden layers of 64 | 8,970 | 0.9676 | 0.0057 |

::: notes
Handwritten digits, eight by eight, ten classes, five seeds per row. The `sd`
column is the seed-to-seed standard deviation, and it is on the slide because
without it three of these four rows are indistinguishable.

Read the first row first: multiclass logistic regression, no hidden layer at
all, 650 parameters, 0.9324. That is the baseline every architecture on this
table has to beat, and most of them barely do.

Ask the room which row they would ship. The next slide is the reading.

Handout section 6.3.
:::

# Reading that table
- No hidden layer at all is within **3.5 points** of the best network here
- The first hidden layer is worth **2.8 points**; 32 units to 64, **0.09**
- On the acceptance data the same step was worth **39 points**
- Digits vote pixel by pixel: the acceptance rule needed two measurements
  **at once**

::: notes
This is the slide that keeps the lesson honest, and it should be delivered
without apology. Two thirds of the way through a lesson about neural networks,
the headline result on the real dataset is that a linear model was almost as
good.

The last two bullets are the generalisable part and they belong in the summary
table of handout section 12: reach for a hidden layer when the answer depends
on inputs jointly, and try the linear model first when the inputs vote roughly
independently.

Put it to the room as a diagnostic question they can ask before training
anything: can I imagine a single weighted sum of these features answering this
question? If yes, start there.

Handout section 6.3.
:::

# A sigmoid layer divides the gradient by about four
- Any number of linear layers composes to one: the activation is the entire
  reason depth buys anything
- The sigmoid's derivative σ(z)(1 − σ(z)) peaks at **¼**, at zero and nowhere
  else
- Backpropagation multiplies by it **once per layer**
- Initialisation makes the weight factor about 1, leaving the derivative in
  charge, so **each sigmoid layer divides the gradient by about four**

::: notes
This is the number the lesson exists to leave behind, so say it, write it, and
come back to it at the close: a sigmoid layer divides the gradient by about
four.

Take the first two bullets quickly but do not skip them - the collapse of
stacked linear layers is a one-line calculation and it is the reason the whole
activation question exists. It is also worth naming the consequence: a network
whose units have all saturated or all died has effectively performed that
collapse on itself.

Then the ¼. It is not an empirical observation, it is a maximum: the product
of a number between zero and one with one minus itself cannot exceed a quarter.
Ask the room where that maximum is attained, and why that is the worst possible
place for it to be - the answer is at zero, which is exactly where a freshly
initialised network's pre-activations live.

Handout sections 7.1 and 7.2.
:::

# Three activations, and their derivatives

![](activation_functions.png)

::: notes
Left panel: three activation functions. Right panel: their derivatives, which
is the quantity backpropagation actually multiplies by once per layer - point
at the right panel, not the left, because that is where the content is.

The horizontal line at a quarter is the subject of the previous slide. Note
that tanh is every bit as much a squashing function as the sigmoid - same S
shape, same saturation at both ends - and its derivative peaks at 1 rather
than a quarter. That difference is the whole story, and it is why the slide
after next reports tanh beating the rectified linear unit.

Handout section 7.2.
:::

# Six layers of it, measured

![](vanishing_gradients.png)

::: notes
Gradient norms at every weight matrix of an untrained six-hidden-layer
network, on a logarithmic scale, with a band spanning eight random
initialisations.

Read the slope, not the endpoints. The sigmoid line falls by three and a half
orders of magnitude from output to input; tanh and the rectified linear unit
are flat. The per-layer factors measured across those eight seeds are 3.90,
3.97, 4.14, 4.05, 3.92, 4.31, 4.06 and 3.86 - all eight between 3.9 and 4.3,
scattered around 4 exactly as the bound predicts.

Say what this means for the first layer: it is receiving a gradient thousands
of times smaller than the last layer's, so in any fixed number of epochs it
effectively does not move. The layers nearest the data, which is where
representation learning has to happen, are the ones that learn least.

Handout section 7.3.
:::

# One draw is not a law
- End-to-end ratio, last weight matrix to first: median **3,547**
- Across the same eight seeds it ranges **2,734 to 4,607**
- Quoting the largest would report one draw as though it were a law
- **The per-layer factor of about 4 is the property**
- By depth: 9.5 at two layers, 170 at four, 3,553 at six, 46,250 at eight

::: notes
This slide is a methodological point wearing a numerical costume, and it is
the habit worth more than the number. A single measurement of a compounding
quantity is a draw from a distribution, and reporting its extreme as the
headline is how plausible-looking folklore gets manufactured.

Ask the room which number they would put in a report, given the eight
measurements. The defensible answers are the per-layer factor, because it is
the thing that is actually stable, or the median with its range attached.
"4,607-fold" is indefensible and is exactly the kind of number that ends up
quoted in a slide deck five years later with no error bar.

The last bullet is the sanity check that the mechanism, not just the number, is
right: the ratios track four-to-the-depth across four orders of magnitude, and
sit slightly below it because the weight matrices give back a little of what
the derivative takes.

Handout section 7.3.
:::

# Eighty epochs: only one of them never learns
- Same six layers, same data, differing **only** in the activation
- Sigmoid: **0.1000**, exact chance on ten classes
- tanh: **0.9750**, past 0.85 within 3 epochs
- Rectified linear unit: **0.9611**, 9 epochs to get going
- What kills the sigmoid is not squashing: it is *where its derivative is
  bounded*

::: notes
The last two bullets are why this slide is not a slogan. tanh saturates just as
hard as the sigmoid does; it is fine because its derivative peaks at 1 rather
than a quarter. If the explanation were "squashing functions are bad", tanh
would have failed too, and it did not.

"Use the rectified linear unit" remains reasonable default advice - it is
cheaper to compute and does not saturate for large positive input - but on six
layers of 32 units those advantages do not show up, and pretending they did
would be teaching a slogan instead of a mechanism. Say that out loud; students
will meet the slogan everywhere else.

Ask the room what they would have concluded from a single run comparing only
the sigmoid and the rectified linear unit. The answer is the right conclusion
for the wrong reason, which is the least useful kind.

Handout section 7.3.
:::

# The sigmoid never leaves chance

![](deep_training_curves.png)

::: notes
Put the previous slide's numbers on the picture. The flat line at 0.1 is the
six-layer sigmoid network: ten classes, exact chance, eighty epochs, and a
gradient reaching its first layer some three thousand times weaker than its
last. It is not learning slowly. It is not learning.

The question for the room: both of the other two curves belong to squashing
functions or to a function with a hard corner - so what exactly is the sigmoid
guilty of? The answer is the quarter. tanh squashes just as hard and its
derivative peaks at one, which is why it is up there with the rectified linear
unit rather than down with the sigmoid.

Handout section 7.3.
:::

# Notebook 2, live

- The same network again, this time in six lines of Keras, and the same
  answer
- Softmax on ten classes, and the depth comparison behind the table
- The per-layer gradient shrinkage, measured across eight seeds at five depths
- How big the random weights should be, and what happens at both extremes

::: notes
Run `Notebooks/02_keras_softmax_and_depth.ipynb`. Twenty minutes.

Open by rebuilding notebook 01's network in Keras and checking it lands in the
same place - the point being that the library is a convenience, not a
different algorithm, and that they now know what every one of those six lines
does.

Protect the shrinkage measurement if time is short. Printing the eight
per-layer factors themselves, and watching every one of them land between 3.9
and 4.3, is what makes the quarter-bound a measurement rather than an
assertion.

Have them change the depth and predict the ratio before running it.
:::

# Zero cannot work
- Lesson 3 started from zero, and was right to: those costs are convex
- All-zero weights **freeze** a network: hidden and output weights both
  receive exactly zero gradient
- **Every parameter is frozen except the output bias**
- That bias converges to the base rate, so the cost converges to the label
  entropy: **0.6887 predicted, 0.6887 measured**
- Distinct hidden columns afterwards: **1 of 32**

::: notes
The usual explanation for random initialisation is symmetry - identical units
receive identical gradients and stay identical for ever - and that argument is
correct and applies to any initialisation that repeats a weight vector.

For all-zero weights something stronger happens, and it is worth deriving in
two lines because it is exact: the backpropagated signal into the hidden layer
is the output weights transposed, which are zero, so it vanishes; and the
output weight gradient is the hidden activations transposed, which are also
zero. Nothing moves but one bias.

The prediction is not approximate. The cost converges to the entropy of the
label distribution, 0.6887 at a base rate of 0.5471, and the notebook measures
0.6887. Ask the room why that agreement is exact rather than close - because
the mechanism is exact, not statistical.

Handout section 8.1.
:::

# Only the starting point differs

![](zero_init_symmetry.png)

::: notes
Thirty-two hidden units, identical in data, architecture and learning rate,
differing only in where the weights started. The zero-initialised network
flattens immediately at the entropy of the label distribution; the randomly
initialised one trains.

The framing to leave them with: random initialisation is not a heuristic that
happens to help. It is what makes the units *different problems to solve*.
Without it there is one unit in the layer, repeated thirty-two times.

Handout section 8.1.
:::

# How large? Propagate the variance
- A unit sums n inputs, so variance is multiplied by n × Var(w) each layer
- Set Var(w) = 1/n, which is **Glorot**. The ReLU halves it again, so it wants 2/n,
  **He**
- **Too small collapses**: indistinguishable from zero by layer 4
- **Too large saturates** rather than exploding: 73% of the last layer past
  0.99 in absolute value
- Glorot loses about half its spread over eight layers

::: notes
Give the variance argument in words before any of the numbers: a unit adds up
n things, so its output variance is n times the variance of one term, and if
that factor is not about 1 then repeating it once per layer gives you a
geometric sequence in the depth. That is the entire derivation; the constants
1/n and 2/n follow immediately.

The two failure directions are worth contrasting explicitly because students
expect "too large" to mean "explodes". With tanh it cannot explode - the output
is bounded by 1 - so it saturates instead, and a saturated layer passes signal
forward and nothing backward. That is the vanishing gradient arriving from the
other direction, which is a nice thing to notice out loud.

Handout section 8.2.
:::

# The learning rate, swept

| $\alpha$ | final training loss | validation accuracy | sd |
|---|---|---|---|
| 0.001 | 1.9550 | 0.5574 | 0.0794 |
| 0.01 | 0.1965 | 0.9398 | 0.0035 |
| 0.1 | 0.0115 | 0.9556 | 0.0039 |
| 0.5 | 0.0007 | **0.9685** | 0.0026 |
| 2.0 | 2.3222 | 0.1009 | 0.0013 |

::: notes
The same network and the same data at five learning rates, three orders of
magnitude apart, five seeds each. The learning rate is the first thing to
sweep and the last thing to guess.

Read the two ends against each other. At 0.001 the loss is still at 1.9550
when the epochs run out, and the seed-to-seed spread of 0.0794 is twenty times
any other row's - a network that has not settled anywhere. At 2.0 the accuracy
is 0.1009, which is exact chance on ten classes: the steps overshoot every
minimum they approach.

Between "still falling" and "never falls at all" there is about one and a half
orders of magnitude of useful range. Handout section 9.1.
:::

# Three regimes, and a narrow band between them

![](learning_rate_sweep.png)

::: notes
Logarithmic vertical axis, so each gridline is a factor of ten. Walk the three
regimes: the top curve is still falling when the epochs run out, the middle
band works, and the flat curve at the top is the rate that overshoots every
minimum it approaches and never descends at all.

The useful range here spans about one and a half orders of magnitude, and it
is found by sweeping, not by reasoning. Ask the room what the top curve would
look like to someone who did not run this sweep - the answer is "a network
that is too small", which is the next slide.

Handout section 9.1.
:::

# The predictable mistake
- Too small, and the loss is still falling when the epochs run out
- **It looks exactly like a network that is too small**: both accuracies low
- The instinct that follows is sound: low training accuracy really is the
  signature of underfitting
- **Vary α over orders of magnitude before touching the architecture**

::: notes
This is the named predictable mistake of the segment, and the second bullet is
the whole of it. The two situations produce the same symptoms, and the instinct
to reach for capacity is not stupid - it is the textbook response to the
textbook signature of underfitting. It is simply that the cause is in the
optimiser rather than in the architecture, and adding capacity moves you
further from the fix.

The last bullet is a second instance of the same family. Five runs in eight
finished above their own minimum, because a rate that is safe on a flat part
of the cost surface is not safe on a sharper part reached later. Ask the room
how they would detect that from a training curve - the answer is that the
curve turning upward late is the signature, and it is invisible if you only
look at the final number.

Handout section 9.1.
:::

# The rectified linear unit has its own failure
- Its derivative is exactly 1 on the active side: so, no vanishing gradient
- But a unit negative for **every** example gets zero gradient, for ever
- **Dead**: the gradient that would revive it is the one it cannot receive
- At α = 1.0, **33 of 64 are dead**: accuracy 0.9583 against a best of 0.9667

::: notes
The second predictable mistake of the lesson, and it follows directly from the
first segment doing its job too well. Having watched the sigmoid fail, the
natural conclusion is that the rectified linear unit is safe, and the argument
for that conclusion is correct as far as it goes.

Land the last bullet rather than the drama. With half the layer dead the
survivors absorb the work and accuracy drops by less than one point - nothing
in the training curve announces the failure. That is what makes it dangerous:
if that layer were your bottleneck, you would be tuning everything except the
thing that is actually wrong.

Handout section 7.4 carries this measurement, and notebook 03 reproduces the
sweep. The picture of it is the next slide, so do not point at anything yet.
Ask the room how they would check. Counting units whose activation is zero on
every training example takes one line.
:::

# Half the layer dead, and almost no visible cost

![](dead_relu.png)

::: notes
Two axes, and the point is the disagreement between them. The bars are dead
units and they explode at the largest learning rate; the line is validation
accuracy and it barely moves.

That is what makes this failure worth naming. Nothing in the training curve
announces it - accuracy is fine, the loss is fine, and a third of the layer
has been doing nothing for the whole run. Ask the room how they would ever
detect it. The answer is the one measurement on this slide: push the training
data through and count the units whose maximum activation over the entire set
is exactly zero.

Handout section 7.4.
:::

# Mini-batches, momentum, and Adam
- Mini-batch **stochastic gradient descent (SGD)**: noisy, unbiased, vectorises
- **Momentum** averages past gradients: oscillations cancel, and lesson 3's
  stretched valley is repaired
- **Adam (adaptive moment estimation)**: a per-parameter step size
- Which is what makes it forgiving of a badly chosen global α

::: notes
Three ideas at speed, because the measured comparison on the next slide is
where the content is.

The one to spend a sentence on is why batch size and learning rate cannot be
tuned independently: a larger batch gives a less noisy gradient, which
tolerates a larger step. Students who tune them one at a time will find each
sweep contradicting the other.

For Adam, the phrase worth remembering is per-parameter step size. A single
global learning rate has to serve every coordinate, and coordinates differ
enormously in scale - which is exactly the condition-number picture from
lesson 3, now with one step size per direction.

Handout sections 9.2 and 9.3.
:::

# Adam, measured: read the spread before the mean

| optimiser | test accuracy | sd | worst of 5 | vs the true rule |
|---|---|---|---|---|
| plain gradient descent | 0.9389 | 0.0184 | 0.9027 | 0.9685 |
| with momentum 0.9 | 0.9349 | 0.0094 | 0.9173 | 0.9645 |
| Adam | **0.9485** | 0.0072 | 0.9360 | 0.9792 |

::: notes
All three on the acceptance problem, at the width where notebook 01's plain
gradient descent had stalled, five seeds each.

Read the `worst of 5` column before the mean. Adam's mean is about a point
above plain descent, which is unremarkable; its worst run is more than three
points above, and its spread is less than half. What Adam bought is the
disappearance of the bad case - which is exactly what notebook 01 spent five
restarts buying by hand. Adam in one run matches what plain descent needed five
to reach.

Now the row nobody would put in a marketing table: **momentum did nothing at
all here**, a fraction of a point below plain descent and well inside either
method's spread. Say it rather than dropping it. Momentum is a good default,
not a guarantee, and a comparison in which every row improves on the last is
usually a comparison that has been curated.

The last column is the same networks scored against the true acceptance rule
instead of the rig's recorded verdict - about three points higher throughout,
which is the measuring instrument, not the model. Handout sections 9.3 and 11.3.
:::

# Three regularisers, and what they bought

| method | test accuracy | sd | train − test gap | epochs run |
|---|---|---|---|---|
| early stopping only | 0.9411 | 0.0069 | 0.0589 | 38.8 |
| + dropout 0.4 | 0.9483 | 0.0038 | 0.0503 | 48.8 |
| + L2 $10^{-3}$ | 0.9478 | 0.0011 | 0.0522 | 198.4 |
| + dropout and L2 | **0.9511** | 0.0045 | 0.0489 | 161.6 |

::: notes
Trained with early stopping on validation, best weights restored, scored on a
test set that nothing was chosen against, five seeds each. Name the three
methods first: early stopping keeps the weights from the best epoch and is the
cheapest regulariser there is; L2 adds a penalty on the squared weights,
identical in form to Ridge in lesson 3; dropout deletes each unit
independently on every training batch and rescales the survivors, so no unit
can rely on any other being present.

Now read the table the way lesson 5 asks. All three land between 0.7 and 1.0
points above early stopping alone, against a seed-to-seed spread of 0.1 to 0.7
points - one to two standard deviations. **Enough to say they helped, nowhere
near enough to rank them.** On a single run with one seed you could comfortably
have measured the three in any order.

The `epochs run` column says something the accuracy column cannot: L2 kept
validation loss improving for roughly four times as long before early stopping
fired. A real qualitative difference, in a comparison whose headline numbers
are not separable.

Handout section 10.3.
:::

# It memorises the training set, and generalises anyway

![](overfitting_curves.png)

::: notes
301,066 parameters on 300 training examples - a thousand parameters per
example. On the left, training accuracy reaches 1.000 and stays there: the
network has enough freedom to store the answers outright, and it does.
Validation accuracy sits near 0.95 and stays there too, which classical
bias-variance reasoning from lesson 5 does not lead you to expect.

The right panel is the one to dwell on. Validation *loss* bottoms out at epoch
13 - the dotted line - and then climbs for the rest of training, while
validation *accuracy* holds flat and in fact peaks later, at epoch 17. The network is not getting more answers wrong - it is getting
steadily more confident about the ones it already has wrong. Accuracy cannot
see that; cross-entropy can. Ask the room which of the two they would early-stop
on, and why.

Handout section 10.1.
:::

# Notebook 3, live
- The learning-rate sweep and the dead-unit count, from scratch
- Plain descent, momentum and Adam from an identical start
- **301,066 parameters on 300 examples**: memorises, generalises anyway
- Validation *loss* climbs while *accuracy* stays flat
- More data beats more tuning, three to one

::: notes
Run `Notebooks/03_training_in_practice.ipynb`. Eighteen minutes.

Two cells to protect. First, the overfitting run: a thousand parameters per
training example, training accuracy at exactly 1.0000, and validation accuracy
holding near 0.95 anyway. Classical bias–variance reasoning from lesson 5 does
not lead you to expect that, and taking it seriously is an open research
question - but taking it as licence to stop validating would be a serious
mistake.

Second, the loss-versus-accuracy divergence, because it is the practical
consequence: the network is not getting more answers wrong, it is getting
steadily more confident about the ones it already has wrong. Accuracy cannot
see that; cross-entropy can.

The last bullet is the one to say slowly. More data beat the best regulariser
by a factor of three, and unlike every method in the previous segment it is not
something you can tune your way to. Handout sections 10.1, 10.3 and 10.4.
:::

# What to take away
- **A neuron is a line**: three enclose a region, one never does
- A hidden layer does not classify: it **moves the data**
- **Backpropagation is the chain rule right to left**: so gradient-check it
- **A sigmoid layer divides the gradient by about four**
- Report a difference with the spread it was measured against

::: notes
The number to carry out of the room is the fourth bullet, and it is worth
asking them to write it down: a sigmoid layer divides the gradient by about
four, because its derivative cannot exceed a quarter. Not 4,607-fold - that was
one draw of a compounding quantity, and the honest version is the per-layer
factor.

That is the fifth of these carry-home numbers the course has produced: 77%
accuracy on coin-flip labels, 98 of 128 imputed rows that borrowed from the
test set, a coefficient of 365 where an unpenalised fit wanted billions,
37 of 40 disguised accounts
caught by reconstruction error, and now a factor of four per sigmoid layer.

If one habit survives today, make it the last bullet. This lesson's methods
produce impressive-looking numbers faster than anything else in the course, and
every table today carries a spread beside its mean for that reason.

Lesson 10 changes exactly one assumption: everything here treats the 64 pixels
of a digit as 64 unrelated numbers, and permuting them consistently would
change nothing. That is obviously wrong for an image.

Handout section 12 and the summary.
:::

# Homework

- **Exercise 9**, due **Friday 27 November 2026, 23:59**
- `Exercises/09_neural_networks.md`

::: notes
Set it explicitly and say the deadline out loud: Friday 27 November, 23:59, at
the start of lesson 10.

Two standing requirements, both of which today's material makes easy to
forget. Every reported difference needs the spread it was measured against  - 
one seed is one draw, and this lesson has three separate tables where the
seed-to-seed spread is comparable to the effect being reported. And any
backward pass written by hand must be gradient-checked before any conclusion is
drawn from what it trained; a wrong gradient trains, and it will not tell you.

Next week is convolutional networks, which is this lesson's architecture with
one assumption changed. Tell them so, and tell them which assumption: that the
inputs have no spatial structure worth exploiting.
:::

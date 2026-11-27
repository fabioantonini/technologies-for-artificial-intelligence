---
title: "Neural Networks — Commentary"
subtitle: "Lesson 9 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Not examinable · reading time about two hours"
header-includes:
  - \usepackage{needspace}
  - \sloppy
---

> **What this document is.** A companion to the slides of lesson 9: for every slide,
> what it means, the steps it leaves implicit, and the questions it tends to raise,
> answered under the slide they belong to. It is a second explanation of the same
> material, in a different voice from the handout, which remains the reference for
> the derivations. It is **not examinable**: nothing here is needed for the exam that
> is not already in the handout and the notebooks.
>
> Slide numbers and titles are the ones in the lesson's slide deck, in `Slides/`.
> Where a section mentions "the notes", it means the speaker notes stored in the
> slide deck. The figures are the lesson's own, from `Figures/`.

<!-- examples-note:begin -->

> **Worked examples** sit under three slides (9, 34, 44):
> a handful of invented values each, small enough to work through by hand. Their
> numbers are not the lesson's data; every one is computed by `commentary_examples.py`,
> beside this file, which also writes them here.

<!-- examples-note:end -->

<!-- numbers-not-from-data
0.98: a hypothetical score, in a student's question under slide 5
1,056: 33 × 32, the images in 33 full mini-batches, computed in the text under slide 42
-->

---

## Slide 1 — Lesson 9: Neural Networks

The title slide. The lesson has three problems and one question asked of all three:
what is a hidden layer worth? Two problems come from Meridian Instruments, a
fictional maker of sensors whose data is generated with every rule published; the
third is a set of real handwritten digits. The answer the lesson arrives at is a
pair of numbers, 39 points against 2.8, and the rest of the three hours explains
why the same idea is worth so much more on one problem than on the other.

---

# Part I — One neuron is one line

## Slide 2 — Agenda

Every model so far drew its boundary from a family chosen in advance: a straight
line for logistic regression, a kernel's curves for the support vector machine, a
staircase of rectangles for a tree. A neural network learns the shape of the
boundary itself. The agenda goes in four steps: one unit, which is a line; many
units side by side; how such a network is trained; and training in practice, on
real digits.

---

## Slide 3 — Exercise 8, before we start

The lesson opens by discussing the previous exercise: nothing is collected. What
mattered there was the prediction made *before* plotting, and its reason. The
common mistake was to trust a two-dimensional plot of the principal components to
show every anomaly: an anomaly that changes how the columns move together, without
making any single column unusual, need not sit at the edge of that picture.

The last line carries the same trap into today: a network's high training accuracy,
on its own, says little. You will see a network reach 1.000 on its training images
while learning less than it seems.

---

## Slide 4 — What a network adds

A network computes new coordinates from the inputs, its hidden units, and then draws
a simple boundary in those coordinates. Lesson 6's kernels also changed the
coordinates, but the coordinates were fixed by whoever chose the kernel; a network
learns them from the data, together with the final boundary, within the
architecture you choose.

**Isn't the output layer then deciding on "simpler" features?** Not simpler: features
in which a straight line is enough. The output unit is still a logistic regression;
what changed is what it is given to look at. Slides 18 and 19 show it on a concrete
problem.

**Didn't the support vector machine already learn a curved boundary from data?** It
learned where to put the boundary, but in coordinates set by the kernel. A network
learns the coordinates themselves.

---

## Slide 5 — Three problems, and a ceiling of 0.97

The three problems: Meridian's acceptance test (3,000 sensors, pass or fail), the
channel-drift problem (800 sensors, correctable or not), and 1,797 handwritten
digits of 8 × 8 pixels. The test rig that records Meridian's verdicts gets 3% of
them wrong at random, so even a model that knew the true rule would disagree with
3% of the recorded labels: the best expected accuracy is 1 − 0.03 = 0.97.

The last bullet announces the lesson's number: a hidden layer adds 39 points of
accuracy on the sensors and 2.8 on the digits. Why so different is the real
question of the lesson; the answer comes after the digits.

**Is the 2.8 a preparation for the lesson on convolutional networks?** No, it is this
lesson's own number. A hidden layer is worth a lot when the inputs act together (the
sensors, 0.5507 to 0.9445) and little when a weighted sum of them already does the
job (the digits, 0.9324 to 0.9602). The bridge to convolutional networks is a
different finding, at the end of the lesson: shuffling the pixels changes nothing.

**Is it 2.8%?** No: 2.8 *points*. It is the difference of two accuracies,
0.9602 − 0.9324 = 0.0278. Accuracies are read as percentages, so their difference
sounds like one; but "2.8%" would mean 2.8% of the score, which is not what was
measured.

**Could a model score 0.98 if the rig is wrong 3% of the time?** On a particular
sample, yes: 0.97 is an expected value, and the rig's mistakes on 750 test sensors
need not be exactly 3%. It applies only to the acceptance test; the drift labels and
the digits carry no such errors.

---

## Slide 6 — The perceptron, and a unit you already have

Rosenblatt's perceptron of 1958 outputs 1 if the weighted sum plus a bias is
positive, and 0 otherwise. Replace that hard step with lesson 4's smooth sigmoid and
you have logistic regression. This lesson calls it a **unit**, or a **neuron**; the
only new idea is to use it as a component, repeated.

**Is a network just logistic regressions stacked?** The structure of the unit
repeats: a weighted sum plus a bias, then an activation. But the activation changes.
The hidden units of today's networks use the rectified linear unit (ReLU, slide 16),
and the sigmoid stays only at the output. Slide 61 explains why sigmoids everywhere
would be a problem.

**Is the perceptron the same as logistic regression?** They share the weighted sum and
the straight boundary, but not the training. Rosenblatt's perceptron rule nudged the
weights only after a mistake. The sigmoid has a slope everywhere, so the weights can
follow the gradient of the cross-entropy, lesson 4's gradient descent; a hard step
has slope zero almost everywhere and gives gradient descent nothing to follow.

---

## Slide 7 — A neuron is a line

The unit computes $z = w \cdot x + b$ and then $\hat{y} = \sigma(z)$. The sigmoid is
increasing, so $\hat{y} > 0.5$ exactly when $z > 0$: the points where the unit is
undecided are those with $z = 0$, and since $z$ is linear in $x$ they form a straight
line in two dimensions, a plane in three, a flat hyperplane in general.

**Why doesn't the sigmoid bend the boundary?** Because it comes *after* the weighted
sum and does not change which side of zero a point is on; it only squashes how far.
Curves appear when several units are combined, which is the rest of the lesson.

**What is the "how far"?** $z / \lVert w \rVert$ is the signed distance of a point from
the line; the sigmoid turns it into a probability.

---

## Slide 8 — One neuron, drawn

![](one_neuron.png)

Left: the unit, inputs weighted and summed with the bias into $z$, then the sigmoid.
Right: the same unit as a line on the plane, $x_1 + 2x_2 - 1 = 0$, with the colour
showing $\hat{y}$. On one side of the line $\hat{y}$ is above 0.5, on the other below,
and on the line exactly 0.5.

**What does the shading mean if the classifier returns 0 or 1?** The shading is the
continuous output $\hat{y}$; the class comes from comparing it with 0.5.

**Which side is "above"?** That depends on the direction of $w = (1, 2)$: reverse the
signs of $w$ and $b$ and the sides swap, while the line stays where it is.

---

## Slide 9 — One unit, by hand

With $w = (1, 2)$ and $b = -1$: at $(1, 1)$, $z = 1 + 2 - 1 = 2$ and
$\hat{y} = \sigma(2) = 0.8808$, class 1; at $(0, 0)$, $z = -1$ and $\hat{y} = 0.2689$,
class 0; at $(1, 0)$, $z = 0$ and $\hat{y} = 0.5$: the point is on the line,
undecided.

**Why is $(1, 0)$ undecided rather than class 0?** Its weighted sum is exactly zero, so
the sigmoid gives exactly 0.5: a tie on the boundary. Software has to pick a
convention for it; the mathematics does not.

<!-- example:begin -->

\Needspace{18\baselineskip}

### Worked example — three more points for the unit of slide 9

The same unit, $w = (1, 2)$, $b = -1$; the rule is "class 1 when $z > 0$".

| point $(x_1, x_2)$ | $z = w_1x_1 + w_2x_2 + b$ | $\hat{y} = \sigma(z)$ | class |
|---|---|---|---|
| (0.5, 0.5) | 1 × 0.5 + 2 × 0.5 − 1 = 0.5 | σ(0.5) = 0.622 | class 1 |
| (2, −0.5) | 1 × 2 + 2 × (−0.5) − 1 = 0 | σ(0) = 0.500 | on the line |
| (0, 0.5) | 1 × 0 + 2 × 0.5 − 1 = 0 | σ(0) = 0.500 | on the line |

Two of the three land exactly on the line $x_1 + 2x_2 - 1 = 0$, although they look
nothing alike: the line runs through every point whose weighted sum is zero.

<!-- example:end -->

---

## Slide 10 — Meridian's acceptance test

At the end of the line a test rig sends each sensor a reference signal and measures
two **calibration offsets**: by how many decibels its output is stronger or weaker
than specified (the gain offset $g$), and by how many degrees its output wave is
shifted from the reference (the phase offset $p$). An offset of 0 is a perfect
sensor. A sensor passes when $(g/0.50)^2 + (p/3.75)^2 < 1$: each offset is divided
by its tolerance, and the two results are squared and added, so the two offsets are
judged *together*.

The split comes first, 2,250 sensors to train and 750 to test, and the scaler is
fitted on the training part only, as lessons 1 and 2 require.

**Why is the pass region a circle of radius 1.25?** Both tolerances are exactly 1.25
times the production spread of their axis: 0.50 dB against a spread of 0.40, and
3.75 degrees against 3.00. Dividing the gain by 0.50 is the same as dividing it by
0.40 and then by 1.25. After standardising, call the two results $u$ and $v$; the
rule becomes $u^2 + v^2 < 1.25^2$, a circle of radius 1.25.

**Why "very nearly" a circle?** The scaler uses the spreads of the 2,250 training
sensors, which are close to 0.40 and 3.00 but not equal to them (atom 2 prints 0.39
and 3.057). Drawn in the scaler's units the true boundary is an ellipse with
semi-axes 1.28 and 1.23, centred within 0.02 of the origin.

**Isn't the radius 1?** The rule ends in "< 1", and standardising seems to bring
everything to unit scale; but that "1" counts tolerances, and in units of spread
each tolerance is 1.25.

**Why standardise if the rule is written in physical units?** The two axes have very
different spreads, 0.40 dB and 3.00 degrees. Standardising puts them on the same
scale, which matters to the model's weights and gradient steps.

---

## Slide 11 — 2,250 sensors, raw and standardised

![](acceptance_data.png)

Left: the raw measurements, decibels against degrees. Right: the same points after
standardising, with the rule that made the labels drawn as a circle. The
wrong-coloured points, rejected inside and accepted outside, are the rig's mistaken
verdicts: it flips about 3% of them wherever a sensor lies, so they are scattered
over the whole plane, not lined up along the boundary. No model can get them right,
and that is the ceiling of 0.97.

**Why do the two panels look alike?** Because each panel's axes are scaled to its own
data. Read the numbers on the axes: the gain runs over about ±1 decibel and the phase
over about ±10 degrees, 7.5 times wider; after standardising, both run over about
±3. Standardising changes the units, not the shape the eye sees.

---

## Slide 12 — What one line is worth

Logistic regression, trained on the 2,250, scores 0.5507 on the 750 test sensors:
413 right. Always answering "accepted" gets the 411 accepted sensors right,
411 / 750 = 0.5480. Both are scored on the same 750 sensors, which neither saw in
training, so the comparison is like for like. Always answering the most common class
is the **baseline** of lessons 4 and 5.

**Why measure on the 750 only?** Because they were not used in training: a score on
the training sensors would be optimistic. This is lesson 1's rule.

**Can a line really do no better?** A little: a line that cuts off a cap beyond the
circle, where every sensor is rejected, gets those right. Here that cap is worth two
sensors. The next slide shows where the line went.

The predictable mistake is to read 0.5507 as a bug, an optimiser that gave up.
Usually that diagnosis would be right; here it is wrong.

---

## Slide 13 — No line encloses a region

![](linear_boundary_fails.png)

The fitted coefficients are 0.0096 and 0.0701, with intercept 0.1892: almost a
constant model. Setting the weighted sum to zero gives the line
$v \approx -2.7 - 0.14u$, almost horizontal and near the bottom of the picture, far
outside the circle of radius 1.25. Above it the model says "accepted", below it
"rejected": it cuts off a thin cap of rejected sensors.

The fit did not fail. The accept region is a disc centred on the origin: for every
accepted sensor there is, on average, a matching one on the opposite side, and
whatever a line gains by tilting one way it gives back on the other.
**No straight line can enclose a region.**

**Would training longer, or another optimiser, help?** No. This is a limit of what a
line can represent, not of how it was found.

**Is this linear regression?** No: logistic regression. The boundary is a straight
line, which is why "linear" comes to mind first.

---

# Part II — A hidden layer

## Slide 14 — New data: two drifting channels

The data changes: Meridian's second problem. 800 sensors, each with two channels
whose readings drift. When both channels drift the same way, one global correction
removes the drift; when they drift opposite ways, nothing does, because a single
correction cannot move one channel up and the other down.

![](drift_clouds.png)

Four clouds: the correctable ones, where the drifts share a sign, sit in two
opposite corners.

**Have we seen this before?** Yes: lesson 6's pumps, where a fault meant exactly one of
two sensors high. It is an **interaction**: one channel's drift says nothing on its
own, because whether $+1$ is good or bad depends on the sign of the other. Naive Bayes
failed there because it assumes the features independent given the class; logistic
regression fails here because it draws one line.

---

## Slide 15 — The smallest problem that needs a hidden layer

Correctable, class 1, means the drifts share a sign. In logic, "not correctable" is
the exclusive-or (XOR) of the two signs, and "correctable" is its complement, sometimes
called XNOR; either way no single line separates the classes, and logistic
regression scores exactly 0.5000.

The diagnosis is in the fourth and fifth bullets. $a + b$ is near $\pm 2$ when the
sensor is correctable and near 0 when it is not, so the question "is $|a + b|$
large?" needs **two** thresholds, near $a + b = 1$ and $a + b = -1$. One line draws
one.

A historical note from the slide: Minsky and Papert's *Perceptrons*, 1969, proved that
a single perceptron cannot compute the exclusive-or.

---

## Slide 16 — Two rectified linear units (ReLU), drawn

![](two_unit_network.png)

The network of the next slide, drawn before it is described. Two inputs, the drifts
$a$ and $b$. Two hidden units, each a weighted sum followed by a rectified linear
unit, $\max(0, z)$, drawn as the small kink inside each node. One output unit with
the sigmoid. The first hidden unit computes $a + b - 1$ and wakes up only when
$a + b > 1$; the second computes $-a - b - 1$ and wakes up only when $a + b < -1$.

**Why multiply by 10 at the output?** So that either active unit pushes the output
firmly to "correctable"; with both off, the output's sum is just the bias, −1.

---

## Slide 17 — Two units, written down rather than trained

At the centre $(+1, +1)$: $h_1 = \max(0, 2 - 1) = 1$, $h_2 = 0$, the output's sum is
$10 - 1 = 9$ and $\sigma(9) = 0.9999$. At $(+1, -1)$ both hidden units are 0, the sum
is −1, and $\sigma(-1) = 0.2689$. On all 800 sensors, with nothing trained, the
network scores **0.9938**.

Each hidden unit is the unit of slide 6, a weighted sum plus a bias, with the ReLU in
place of the sigmoid.

**Where exactly does it decide?** The output says "correctable" when
$10h_1 + 10h_2 - 1 > 0$, that is when $h_1 + h_2 > 0.1$, which means
$|a + b| > 1.1$.

**Four centres right, so why not 800?** The centres explain the design; 0.9938
measures it on the scattered points. The 5 misses are correctable sensors close to the
origin, such as $a = 0.32$, $b = 0.47$, where $a + b$ is only 0.79.

---

## Slide 18 — What the hidden layer actually did

![](xor_hidden_space.png)

Left: the sensors as measured, which no line separates. Right: the same 800 sensors
plotted by what the two hidden units output. The correctable clouds have gone to the
axes, around $(1, 0)$ and $(0, 1)$; both uncorrectable clouds sit together at the
origin. In the right-hand panel one line separates them.

**Did the network move the measurements?** No: it computed two new features from them,
the hidden units' outputs, and the right panel uses those as its axes.

**Why do the clouds become segments rather than points?** Take the cloud around
$(+1, +1)$: there $h_2$ is exactly 0 and $h_1 = a + b - 1$ varies with the scatter, so
the cloud lies along the $h_1$ axis. Atom 5 fits a line on $(h_1, h_2)$: 0.9738,
against 0.5000 on the raw drifts.

---

## Slide 19 — The hidden layer classified nothing

The idea to carry out of the first half. Students tend to picture a network as a
chain of classifiers, each improving the previous answer. That is not what happens:
the hidden layer changes the coordinates in which the data is described, and only
the output unit classifies, with a single straight line. Learning new coordinates
from data is called **representation learning**; computing the output layer by
layer is the **forward pass**.

**Is this representation learning, if we wrote the weights?** Not yet. Here the
coordinates were designed by hand; representation learning is training finding
such coordinates on its own, which starts on slide 23.

---

## Slide 20 — Without the ReLU, two layers are one

Why does a network need an activation at all? Take the ReLU away and keep every
weight. Each hidden unit then passes its weighted sum on unchanged,
$h_1 = a + b - 1$ and $h_2 = -a - b - 1$, and the output's sum becomes
$10(a + b - 1) + 10(-a - b - 1) - 1 = -21$ for every sensor: $a$ and $b$ cancel. The
network says $\sigma(-21) \approx 0$, "not correctable", to all 800, and scores
0.5000, since half of them are correctable.

In general, a weighted sum of weighted sums is one weighted sum. With
$h = xW^{[1]} + b^{[1]}$ and $z = hW^{[2]} + b^{[2]}$, substituting gives
$z = x\,W^{[1]}W^{[2]} + (b^{[1]}W^{[2]} + b^{[2]})$: one weight vector, one bias.
Any number of layers without an activation is one unit, a logistic regression with a
sigmoid output or a linear regression without one. It is the bend of the ReLU at 0
that lets several lines combine into a region.

---

## Slide 21 — Forward propagation, drawn

![](forward_pass_map.png)

The forward pass for a network with two inputs, four hidden units and one output,
with every symbol in its place. Weights live on the edges: the first weight matrix,
$W^{[1]}$, is $2 \times 4$, one column per hidden unit, plus four biases. Each hidden
node holds two numbers, its weighted sum $z$ and its activation $a = g(z)$. Stacking
all $m$ examples as rows turns those numbers into matrices; with the 2,250 training
sensors, $m$ is 2,250.

**Why is $W^{[1]}$ 2 × 4 and not "two examples, four features"?** Its two rows are the
two inputs and its four columns the four hidden units. The examples are the rows of
$X$.

**Are these $a$ and $b$ the drifts?** No: here $a$ is an activation and $b$ a bias.
From this slide on, the inputs are called $x_1$ and $x_2$.

---

## Slide 22 — Forward propagation, written

The same pass as a formula:
$A^{[1]} = g(XW^{[1]} + b^{[1]})$, $\hat{y} = \sigma(A^{[1]}W^{[2]} + b^{[2]})$. Read it as
two sentences: the hidden layer is a set of lines followed by a bend at zero; the
output is one more line followed by a probability. The output has shape $m \times 1$,
one prediction per example.

**What is the difference between Z and A?** $Z$ holds the weighted sums before the
activation; $A$ the activations after it, which become the next layer's inputs.

---

## Slide 23 — Letting training find the weights

Nobody writes weights by hand. scikit-learn's `MLPClassifier` (multilayer perceptron)
starts from small random weights, set by `random_state`, and improves them by the
gradient descent of lessons 3 and 4 on the cross-entropy; how the gradient reaches
every weight is the second half of the lesson. On the drift problem it is run 20
times per size, and scored on the 800 sensors it trained on: the question here is
about optimisation, whether training finds the lines at all, not about
generalisation.

Reaching 0.99: **6 of 20** runs with two hidden units, 17 with four, 20 with eight.
The median run with two units scores 0.7381.

**Why do different starts give different answers?** Lesson 4's cost, for logistic
regression, is convex: one valley, and every start reaches the same bottom. A
network's cost, as a function of its weights, has many valleys and flat stretches,
and where training stops depends on where it started.

**If two units can solve it, why does training so often fail?** Because "can" is about
what the network can represent and "does" is about where gradient descent stops. A
wider layer gives training more chances to start near a good set of lines, which is
why networks used in practice are wider than their task strictly requires.

---

## Slide 24 — Two units can; training often misses them

![](drift_boundaries.png)

Run 0 with two and with four hidden units, same 800 sensors and same
`random_state`. With two units training walled off three of the four clouds and
stopped there, at 0.7425: a **local minimum**, a valley of the cost that is not the
deepest, from which no small step leads down. With four units both correctable
clouds are enclosed: 1.0000.

**How is such a picture made?** The trained network's forward pass is run on a fine
grid of points covering the plane, and each point is coloured by whether
$\hat{y} > 0.5$; the boundary is where $\hat{y} = 0.5$.

**How would you notice this on a problem you cannot plot?** By training from several
starting points and reading the spread of the scores.

---

## Slide 25 — Back to the acceptance test: how many hidden units?

![](width_accuracy.png)

The data changes back to the circle no line could enclose. Each column is
`MLPClassifier` with one hidden layer of that size, trained five times on the 2,250
training sensors, `random_state` 0 to 4, and scored on the 750 test sensors: the dot
is the mean of the five, the grey bar runs from the worst run to the best. At three
and four units the worst run is still 0.7227 while the best is above 0.9; at eight
the runs agree and the mean is 0.9445, within three points of the ceiling. From one
line, 0.5507, that is 39 points.

**More hidden layers?** No: there is one hidden layer in every column. What grows is
its width, the number of hidden units.

**Does the bar disappear at eight units?** It shrinks: the five runs agree.

**Should we then pick sixteen, the best mean?** These are test-set scores, used here to
show how capacity changes the boundary, not to choose a width. To choose one
honestly, compare widths on validation data or by cross-validation (lesson 5) and
touch the test set once.

---

## Slide 26 — The boundary at 1, 2, 4 and 8 units

![](boundaries_by_width.png)

Run 0 at 1, 2, 4 and 8 hidden units, with the true circle dashed. One unit gives one
line, exactly like logistic regression; two give a strip; four a four-sided fence;
eight an eight-sided one, close to the circle.

**Will sixteen units give a sixteen-sided polygon?** Not necessarily: the number of
sides is not one per unit, as the next slide shows. Training decides where the lines
go and how much each counts.

---

## Slide 27 — Each hidden unit is a line

![](hidden_lines.png)

The eight hidden units of that network, each drawn as its own line from its column
of the first weight matrix, `coefs_[0]`; the network's boundary in black. Each unit
outputs zero on one side of its line and a value that grows with the distance on
the other; the output unit takes a weighted sum of those eight values.

**How do eight lines make that polygon?** Look at where the lines meet the boundary: at
its corners, not along its sides. Between two lines the same hidden units are on, so
the output's weighted sum is a flat plane there and the boundary is a straight side.
Crossing a line switches one unit on or off, the plane tilts, and the boundary turns
a corner. Notebook 01 counts the lines that cross the region the network accepts:
**4**, all within 0.19 of the centre and with negative output weights, each crossed
twice, which makes the 8 corners. The other four lie 1.84 to 4.38 from the centre;
they are on throughout the region, tilt the plane, and make no corner. The rule is
two corners for each line that crosses the region; "eight units, eight sides" is a
coincidence of this run.

**How does the closed fence arise?** The four central lines have negative output
weights: each lowers the output's sum on one side, in proportion to the distance.
Whichever way you leave the centre, some of them switch on, the sum falls below zero,
and the boundary closes.

---

## Slide 28 — Notebook 1, live

Notebook 01, from one line to a hidden layer, or atoms 1-7 of the atoms notebook,
which take the same steps one call at a time with a prediction before each. The
table on the slide maps each section to its slides and to the number to come back
with.

---

# Part III — How a network learns

## Slide 29 — Break

The second half answers the question the first half left open: the hand-built
network was written down, and `MLPClassifier` found good weights on its own. How does
training find them?

---

## Slide 30 — How a network learns: the loop

![](learning_loop.png)

Training repeats four steps. **Predict**: a mini-batch goes forward through the
network. **Measure**: the loss says how wrong the predictions are. **Blame**:
backpropagation works out how much each weight contributed to the error. **Adjust**:
every weight moves a little against its blame, scaled by the learning rate. After the
weights move, every prediction, error and blame has to be computed again; only the
weights carry over from one round to the next. Steps 1, 2 and 4 are lesson 3's
gradient descent; only step 3 is new.

**Doesn't backpropagation update the weights?** No. It computes the blame, step 3, and
nothing else; the update is step 4, the gradient step. The name is often used for
the whole of training, which is why the two get merged. In Keras the library
computes the blame and the optimiser takes the step.

**Do the weights converge to the optimal value?** To a good minimum, not necessarily
the best: lesson 3's cost had one minimum, a network's has many valleys, as the drift
runs showed.

---

## Slide 31 — Mini-batches and epochs

A **mini-batch** is a small random group of training examples; each round of the loop
uses one, which is cheap and points roughly the right way on average. An **epoch** is
one pass through all the mini-batches. On the acceptance test, `MLPClassifier` uses
its default mini-batch of $\min(200, m) = 200$ sensors: $2{,}250 / 200 = 11.25$, so
12 mini-batches per epoch, 11 full ones holding 2,200 sensors and a twelfth of the
remaining 50.

**Are the 200 drawn at random each time?** Not quite: at the start of each epoch the
training data is shuffled and then cut into consecutive blocks, so within one epoch
every sensor is used exactly once. Both scikit-learn and Keras do this by default.

**Why many epochs?** Each update is a small step; one pass is not enough to get down.
An epoch is 12 updates here, not one.

---

## Slide 32 — The loss and the gradient step

The loss is lesson 4's cross-entropy: $-\log \hat{y}$ if $y = 1$,
$-\log(1 - \hat{y})$ if $y = 0$. A weight's gradient, how fast the loss changes when
the weight changes, is called its **blame**. The update is lesson 3's:
$w \leftarrow w - \alpha \cdot \partial L / \partial w$, with $\alpha$ the learning rate.

For one sigmoid unit the result is strikingly simple: the blame of its output $z$ is
$\hat{y} - y$, and each weight's blame is that times its input.

**Where does "times its input" come from?** From the chain rule, which says that a
change passed through two links is multiplied by both:
$\partial L / \partial w_j = \partial L / \partial z \times \partial z / \partial w_j$.
The first link, sigmoid and cross-entropy together, simplifies to $\hat{y} - y$ (four
lines in the handout, section 4.3). The second is easy to see:
$z = w_1x_1 + w_2x_2 + b$, so nudging $w_j$ moves $z$ by $x_j$ times the nudge, and for
the bias the input is 1. A weight whose input was 0 took no part in the prediction
and gets no blame. Lesson 4 followed the same chain in three links, on its slide 25.

**Is $\hat{y} - y$ every weight's blame?** No, it is the blame of $z$, the output; each
weight multiplies it by its own input.

---

## Slide 33 — One step on one unit

![](one_unit_step.png)

One step with every number. The unit of slide 9, weights 1 and 2, bias −1, sees the
point $(1, 1)$, whose true label is 0. Forward, in blue: $z = 2$, $\hat{y} = 0.881$.
Backward, in rust: blame $0.881$, times each input; every input is 1 here, the bias's
always. A step with $\alpha = 0.5$ takes the prediction from 0.881 to 0.663, towards
the true label 0.

**Why does the bias change too?** It is a parameter like a weight, a weight on a
constant input of 1, so it gets a blame and a step.

---

## Slide 34 — Reading the step

The arithmetic of the figure. The loss for a true label of 0 is
$-\log(1 - 0.8808) = 2.127$. Each weight drops by $0.5 \times 0.8808 = 0.4404$, to
0.5596, 1.5596 and −1.4404; the new $z$ is $0.5596 + 1.5596 - 1.4404 = 0.679$, and
$\sigma(0.679) = 0.663$, with loss 1.089.

The last bullet is the contrast that shows the input at work. At $(1, 0)$, also with
true label 0, $z = 0$ and $\hat{y} = 0.5$; the blames are $0.5 \times 1$, $0.5 \times 0$
and $0.5 \times 1$. With $\alpha = 0.5$, $w_1$ goes to 0.75, the bias to −1.25, and
$w_2$ stays at 2, because its input was 0. The new prediction is $\sigma(-0.5) = 0.378$.

**The prediction is still above 0.5 after the step: did anything improve?** Yes: the
loss fell from 2.127 to 1.089. One step lowered the loss without changing the class,
and that is normal; training is many such steps over many examples.

<!-- example:begin -->

\Needspace{19\baselineskip}

### Worked example — one step where the other input is 0

The unit of slide 9 sees the point $(0, 1)$, true label $y = 0$, with $\alpha = 0.5$.

| | weight from $x_1$ | weight from $x_2$ | bias |
|---|---|---|---|
| input | 0 | 1 | 1 |
| blame = $(\hat{y} - y) \times$ input | 0.731 × 0 = 0.000 | 0.731 × 1 = 0.731 | 0.731 × 1 = 0.731 |
| new value = old − 0.5 × blame | 1 − 0.5 × 0.000 = 1.000 | 2 − 0.5 × 0.731 = 1.634 | −1 − 0.5 × 0.731 = −1.366 |

Forward first: $z = 0 + 2 - 1 = 1$ and $\hat{y} = \sigma(1) = 0.731$. After the
step, $\hat{y} = 0.567$ and the loss falls from $1.313$ to $0.837$. The weight
from $x_1$ has not moved: its input was 0.

<!-- example:end -->

---

## Slide 35 — Backpropagation: blame flows back

![](blame_backwards.png)

In a network, a hidden unit's weights do not touch the loss directly; they affect it
through the output. The error starts at the output and travels back along the same
connections the forward pass used. Each hidden unit receives its share of the
output's blame, in proportion to the weight connecting it to the output; it
multiplies that share by the slope of its own activation and passes it further back
the same way. One backward sweep gives every weight its blame.

**What travels backwards: the inputs, the labels?** Neither. What travels back is the
derivative of the loss, the blame; the values from the forward pass are only used to
compute it.

**How much blame reaches a unit connected to the output with weight zero?** None.

---

## Slide 36 — Backpropagation by hand: a sensor the network gets wrong

The next three slides do one backward sweep with every number, on a network you
already know: the hand-built drift network of slide 16, weights unchanged. The
sensor has $a = 1.5$, $b = -0.1$: the signs differ, so it is not correctable, $y = 0$.

Forward: $h_1 = \max(0, 1.5 - 0.1 - 1) = 0.4$ and $h_2 = \max(0, -1.5 + 0.1 - 1) = 0$;
the output's sum is $10 \times 0.4 + 10 \times 0 - 1 = 3$, and $\hat{y} = \sigma(3) =
0.953$: wrong, and sure. The loss is $-\log(1 - 0.953) = 3.049$.

---

## Slide 37 — The blame, sent back with every number

![](backprop_by_hand.png)

Four steps, right to left. **One**: at the output the blame is $\hat{y} - y = 0.953$.
**Two**: the blame reaching each hidden unit is that times the weight connecting it to
the output, $0.953 \times 10 = 9.53$. **Three**: each hidden unit multiplies it by its
ReLU's slope; unit 1 is on, slope 1, so 9.53 passes; unit 2 is off, slope 0, so its
blame is 0 and, from this sensor, it learns nothing. **Four**: every weight's blame is
its unit's blame times the input it carried: $9.53 \times 1.5 = 14.29$ for the weight
from $a$, $9.53 \times (-0.1) = -0.95$ for the weight from $b$, 9.53 for the bias; at the
output, $0.953 \times h_1 = 0.381$ and $0.953 \times h_2 = 0$.

The 9.53 is computed once and reused for all three weights of unit 1: that reuse is
why one backward sweep costs about one forward pass.

**Does an off unit never learn?** It learns nothing from *this* sensor. Whether a unit
is on depends on the example, and unit 1 here would be off for a sensor near the
origin while unit 2 would be on for one in the $(-1, -1)$ cloud. A unit learns from
the examples on its side of its line.

---

## Slide 38 — One step with α = 0.01, and what it changed

Each weight moves by 0.01 times its blame: $1 - 0.01 \times 14.29 = 0.857$ for the
weight from $a$, $1 - 0.01 \times (-0.953) = 1.0095$ from $b$, $-1 - 0.01 \times 9.53 =
-1.095$ for the bias; the output weight goes from 10 to 9.996 and its bias to
−1.0095; unit 2's weights stay where they were. Forward again:
$h_1 = 0.857 \times 1.5 + 1.0095 \times (-0.1) - 1.095 = 0.089$, the output's sum is
$9.996 \times 0.089 - 1.0095 = -0.115$, and $\hat{y} = 0.471$. One step, and the
network now calls this sensor not correctable; the loss falls from 3.049 to 0.637.

**Why such a small learning rate?** Because the blames are large: the output weight of
10 multiplied every hidden blame by 10. With $\alpha = 0.5$, as on the single unit, the
first weight would jump from 1 to $1 - 0.5 \times 14.29 = -6.1$.

**Does a computer get the same numbers?** Notebook 02 asks TensorFlow for the
gradients with `tf.GradientTape`, which records the forward pass and runs it
backwards, and prints the same blames, 14.29 among them, and the same 0.4712 after
one step.

---

## Slide 39 — What backpropagation buys

Backpropagation is the chain rule, applied one layer at a time from the output back.
Every weight's blame costs about as much as one more forward pass. The naive
alternative is to nudge one weight, run the forward pass again, see how much the loss
moved, and repeat for the next weight: one forward pass per weight, 33 for the
eight-unit sensor network ($2 \times 8 + 8$ into the hidden layer, $8 + 1$ into the
output), millions for a large network. Atom 10 does it both ways on the drift network:
the same blames, and 18 forward passes for 9 weights by nudging. scikit-learn and
Keras run backpropagation inside `fit`.

A historical note from the slide: the method was found more than once, but the 1986
paper in *Nature* by Rumelhart, Hinton and Williams made it the standard.

---

# Part IV — A network in Keras, on the digits

## Slide 40 — The digits: 64 pixels in, ten classes out

![](digit_examples.png)

The third problem, and the first real data: 1,797 handwritten digits shipped with
scikit-learn. Each image is 8 × 8 = 64 numbers, one per pixel, so a network has 64
inputs; and there are ten classes now, not two. The images are split three ways, as
lesson 5 asks, with the same share of each digit in every part: 1,077 to train, 360
to validate, 360 to test. Each pixel is divided by 16 so it lies between 0 and 1.

**Why three parts here, when the sensors had two?** Because today compares designs on
the digits: the validation images are used, repeatedly, to compare them, and the test
images are kept for one final score.

**Are these like the well-known MNIST digits?** No: those are 28 × 28; these are far
coarser, and some are hard for a person to read.

---

## Slide 41 — The digit network, drawn

![](digit_network.png)

The network as a picture. One image, eight by eight, unrolled into a row of 64
numbers: the 64 inputs. Every input is connected to every one of the 32 hidden units,
which use the ReLU; every hidden unit to every one of the 10 output units, one per
digit. Every line is one weight: $64 \times 32$ weights plus 32 biases is 2,080;
$32 \times 10$ plus 10 is 330; 2,410 in all. Under each column is the line of Keras
that creates it.

The unrolling shows one more thing: the network receives 64 numbers in a row, and
nothing tells it which pixels were neighbours in the image. That comes back at the end
of the lesson.

---

## Slide 42 — A network in Keras: 2,410 parameters

**Keras** is the interface; underneath it **TensorFlow** does the arithmetic and computes
the gradients. `keras.Sequential` lists the layers: `keras.Input` with 64 pixels, then
two `layers.Dense`, layers in which every unit sees every input. `compile` chooses the
loss, cross-entropy for integer class labels, and the optimiser; `fit` runs the loop
of slide 30 for 60 epochs, in mini-batches of 32, Keras's default:
$1{,}077 / 32 = 33.7$, so 34 mini-batches per epoch and $60 \times 34 = 2{,}040$ steps.

**What is Adam?** Adaptive moment estimation (Kingma and Ba, 2014): gradient descent with
two additions, made weight by weight. *Momentum*: the step follows a running average of
recent gradients, which smooths out the noise of small mini-batches. *An adaptive step*:
each weight's step is divided by the recent size of its own gradient, so a weight whose
blame has been large or jumpy takes shorter steps, and one whose blame has been small
and steady takes longer ones. The formulas are in the handout, section 5.2.

**Why 34 mini-batches and not 33?** 33 full ones hold 1,056 images; the remaining 21
make a 34th.

---

## Slide 43 — Softmax

The output layer computes ten numbers per image, one per digit. Softmax exponentiates
each one, so they are all positive, and divides by their total, so they add up to 1.
The prediction is the digit with the largest probability. The loss for ten classes is
still cross-entropy: minus the logarithm of the probability given to the true class.

**Is softmax a sigmoid applied to each of the ten scores?** No: all ten share one
denominator, which is why they add up to 1.

---

## Slide 44 — Softmax in words, with three scores

With three scores $z = (2, 1, 0)$: $e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$, total 11.107;
the probabilities are $7.389 / 11.107 = 0.665$, $2.718 / 11.107 = 0.245$ and
$1 / 11.107 = 0.090$, which add up to 1. The largest score gets the largest probability,
but not all of it, which is what makes softmax "soft".

**What if all ten scores are equal?** Each digit gets one tenth.

**What if I add 5 to every score?** Nothing changes: every exponential is multiplied by
$e^5$, the total too, and the ratios stay the same. Only the differences between the
scores matter, and that is the key to the next two slides.

<!-- example:begin -->

\Needspace{20\baselineskip}

### Worked example — softmax with a tie

Three scores, two of them equal.

| class | score $z$ | $e^z$ | probability $= e^z \div$ total |
|---|---|---|---|
| 0 | 1 | 2.718 | 2.718 ÷ 6.437 = 0.422 |
| 1 | 1 | 2.718 | 2.718 ÷ 6.437 = 0.422 |
| 2 | 0 | 1.000 | 1.000 ÷ 6.437 = 0.155 |

The total is $2.718 + 2.718 + 1 = 6.437$. The two tied scores share the top
equally, 0.422 each, and the probabilities add up to 1 (0.999 here only
because each is rounded to three places). Add 3 to every score and
nothing changes: each $e^z$, and the total, is multiplied by $e^3$.

<!-- example:end -->

---

## Slide 45 — Two classes: softmax is the sigmoid

With two classes, the probability softmax gives class 1 is
$e^{z_1} / (e^{z_0} + e^{z_1})$. Divide the top and the bottom by $e^{z_1}$: the top
becomes 1, the bottom $1 + e^{-(z_1 - z_0)}$. That is lesson 4's sigmoid, applied to
the difference of the two scores, $\sigma(z_1 - z_0)$. A check with numbers: scores
$(0, 1)$ give $e^1 / (e^0 + e^1) = 2.718 / 3.718 = 0.731$, and $\sigma(1) = 0.731$.

---

## Slide 46 — Why the difference of the two scores

Until now you have always seen $\sigma(z)$, with one $z$. With two output units each
class has its own score, each a weighted sum: $z_0 = w_0 \cdot x + b_0$ and
$z_1 = w_1 \cdot x + b_1$. Softmax only looks at their difference: scores $(0, 1)$ and
$(5, 6)$ both give 0.731. And the difference of two weighted sums is itself one
weighted sum, $z_1 - z_0 = (w_1 - w_0) \cdot x + (b_1 - b_0)$: call the weights
$w = w_1 - w_0$ and the bias $b = b_1 - b_0$ and it is the $z$ of slide 6.

So two output units are redundant for two classes: one unit computing the difference
directly does the same job, and that unit is logistic regression. That is why every
two-class network in the first half of the lesson had one sigmoid output, and why the
ten-class network needs softmax.

---

## Slide 47 — Softmax says how sure, not how right

![](softmax_example.png)

Two validation images and the ten probabilities the network gives each. The 6 above
gets 0.991 on one digit. The 1 below is a mistake: no digit gets more than a third,
and its most probable answer, a 4, gets 0.323. No probability is ever exactly 0,
since every exponential is positive.

**If a wrong answer gets 0.920, what does that number mean?** How sure the network is,
relative to the other nine digits; it does not mean that answers given with that
probability are right that often. Notebook 02 finds exactly that: a 4 called a 7 with probability
0.920, very sure and wrong.

---

## Slide 48 — Training, watched

![](training_curves.png)

Loss and accuracy after every epoch, on the training and the validation images, for
the network of slide 42, one run. Almost all the learning happens in the first ten
epochs. After the first epoch the network scores 0.2667 on the validation images;
after 60 it scores 0.9639, 13 of the 360 wrong, and 0.9926 on the images it trained
on. Each epoch on the horizontal axis is 34 gradient steps.

**Why do the two curves stay so close?** Because the network generalises: it is small,
2,410 parameters, the two parts come from the same distribution, and 60 epochs are
not enough to memorise. The training curve ends a little above, as it usually does:
the network has seen those images.

**Why is the validation curve above the training curve at first?** Because of how Keras
measures. The training figure is the average over the epoch's 34 mini-batches, taken
while the weights were still changing, so its early batches were scored with worse
weights; the validation figure is taken at the end of the epoch, with the final
weights. The training curve lags by about half an epoch, and while the network
improves fast that lag puts it underneath.

**Does `fit` use the validation images to train?** No: Keras only scores them after
each epoch; every gradient step uses the training images alone.

---

## Slide 49 — Where it goes wrong

![](softmax_confusion.png)

Lesson 4's confusion matrix, now with ten classes, on the 360 validation images. The
diagonal holds the correct answers; off it the mistakes are few and scattered, a 4
taken for a 7, a 9 for a 5, thirteen in all. No pair dominates.

**Why don't the columns add up to the same number?** It is the rows that count the
digits. Each row is a true digit and adds up to how many of that digit the 360 hold,
35 to 37, because the digits are not exactly balanced (row 1: $35 + 1 + 1 = 37$). Each
column is an answer of the network and adds up to how often it was given, which
depends on its mistakes (column 8: $1 + 1 + 1 + 33 = 36$). Some books put the true
class on the columns: read the axis labels first.

---

## Slide 50 — Is the hidden layer worth it here?

Three runs per row, on the validation images. The first row is the **baseline** of
lessons 4 and 5: always answer the digit most common in training, a 1 (109 of the
1,077, tied with 3, 5 and 6; the first of the tie is taken), right on the 37 ones
among the 360, $37 / 360 = 0.1028$. Softmax alone, with no hidden layer, is one
weighted sum of the pixels per digit: multiclass logistic regression, the **linear
model**, 0.9324. One hidden layer of 32: 0.9602, 2.8 points more. Of 64: 0.9611.

Keep the two references apart: the baseline says whether a model has learned
anything; the linear model says whether a hidden layer adds anything.

**Does doubling the width help?** $0.9611 - 0.9602 = 0.0009$, a third of an image of
360, far inside the run-to-run spread: the two rows cannot be ranked.

**So more units never help?** Not in general: on the sensors, going from four units to
eight is what made training reliable (slide 25). The right width depends on the
problem, and it is chosen as this table chooses it: on validation data.

---

## Slide 51 — 39 points against 2.8

![](hidden_layer_worth.png)

What a hidden layer is worth on each problem, side by side: on the sensors, one line
against eight hidden units, on the 750 test sensors, 39.4 points; on the digits,
softmax alone against 32 hidden units, on the 360 validation images, 2.8 points. On
this split that is $39.4 / 2.8 = 14$ times more; notebook 02 redraws the split and
gets 4.7 points, and $39.4 / 4.7 = 8.4$: about ten times either way. The two gaps are
measured on different sets, so compare their size, not their decimals.

**Are these properties of sensors and digits in general?** No: they are measured on
these data. The second split of the digits gives a different gap; what does not
change is its size, a few points against about 39.

---

## Slide 52 — When to reach for a hidden layer

The reason is the shape of each problem. A weight can only say "more is better" or
"more is worse". On the sensors neither is true. An offset fails when it is large in
either direction: $+0.6$ decibels fails, and so does $-0.6$, since
$(0.6/0.50)^2 = 1.44 > 1$. And how large is too large depends on the other offset: a
gain of 0.4 decibels passes with a phase of 0, since $(0.4/0.50)^2 = 0.64 < 1$, and
fails with a phase of 3 degrees, since $0.64 + (3/3.75)^2 = 1.28 > 1$. No weighted sum
can say "near the centre". On the digits each pixel votes much the same way whatever
the others do, ink in the middle saying "an 8, not a 0", and the sum of the votes
already scores 0.9324.

The predictable mistake is a hidden layer by reflex, after seeing one take the sensors
from 0.55 to 0.94. The instinct is sound when inputs act together; on data where a
weighted sum already works it buys little, here 2,410 parameters instead of 650 for
2.8 points.

**Why fit the linear model first: to save memory?** Not mainly: 650 parameters against
2,410 cost nothing here. The linear model is the reference that says whether a
network adds anything, it is simpler to read, and it is less able to memorise.

**How can I know before fitting?** The rule that makes the data, when you can read it,
suggests the answer; then a linear model and a small network, compared against the
spread of their runs, check it. That is exercise 9.

---

## Slide 53 — Notebook 2, live

Notebook 02, a network in Keras on the digits, or atoms 8-12. Its section 3 ends with
one cell in plain TensorFlow that repeats the backward sweep of slides 36-38 with
`tf.GradientTape` and gets the numbers computed there by hand.

---

# Part V — Training in practice

## Slide 54 — Two layers of 64 on the digits, at three learning rates

![](learning_rate_regimes.png)

The same network, two hidden layers of 64, trained with plain stochastic gradient
descent on the 1,077 training digits for 60 epochs at three learning rates, one run
each, scored after every epoch on the validation digits. At 0.001 accuracy is still
climbing when training stops. At 0.5 it settles high, 0.9722, after a spike at epoch
8, one step too long from which the network recovers. At 2.0 it never leaves 0.1,
what guessing scores on ten classes.

**Why is the 2.0 curve flat rather than jumping about?** The first steps were so long
that they threw the network into a region it never left. Notebook 03 looks inside: at
2.0, 49 of the 64 units of the first hidden layer and all 64 of the second output 0 for
every training image. A ReLU unit like that has slope 0 wherever it is used, so no
blame reaches its weights and it never recovers: it is **dead**. With the second layer
dead the network gives every image the same answer, a 4, which is right on the 36 fours
among the 360 validation images: 0.1000. Its loss, 2.3195, is close to
$\ln 10 = 2.303$, the loss of giving every digit one tenth.

**Isn't 0.001 the rate that worked with Adam on slide 42?** Yes, and that is the point:
here the optimiser is plain SGD, the same step for every weight, while Adam adapts each
weight's step. Learning rates do not carry over from one optimiser to another.

---

## Slide 55 — A rate too small looks like a network too small

Stochastic gradient descent (SGD) is lesson 3's update applied on each mini-batch, the
same $\alpha$ for every weight. At $\alpha = 0.001$, after 60 epochs, the network is low
on both sets: training loss 2.0456, validation accuracy 0.4806. Low on both usually
means a model too simple, and the reflex is a bigger network. Here that is the wrong
fix: the same network reaches 0.9722 at $\alpha = 0.5$. It had not finished learning.

**How do I tell the two apart?** The cheapest experiment: change the learning rate by a
factor of ten and train again. Try 0.001, 0.01, 0.1, 1: each rate ten times the last,
multiplications rather than additions, because the right rate can lie anywhere across
several orders of magnitude. Atom 13 does it in ten epochs: 0.1222, 0.6972, 0.9472, and
0.1000 at a rate too large.

---

## Slide 56 — Two layers of 512 on 300 digits: a network that memorises

![](overfitting_curves.png)

A much bigger network on much less data: two hidden layers of 512 units on only 300
training digits, drawn from the 1,077, for 200 epochs. Training accuracy reaches 1.000
within a few epochs and stays there; validation accuracy stays roughly level. The
validation loss is lowest at epoch 13 and then climbs for the rest of training.

**Is it the width, or the data?** The ratio: 301,066 parameters for 300 images, about
1,000 per image, enough freedom to store each answer.

**Doesn't overfitting make the validation accuracy fall?** Not here: it stays level. The
validation loss climbs. Watch only the accuracy and you would see nothing wrong.

---

## Slide 57 — Reading the two curves

The network has stored the 300 answers outright: training accuracy 1.000. The
validation loss goes from 0.1725 at its lowest, epoch 13, to 0.2419 by epoch 200,
1.4 times higher, while the validation accuracy stays near 0.95 (0.9472 at the end).
Accuracy only asks whether the top answer is right; the loss also asks how confident
the network was.

**Does a rising loss mean the network is less sure of its wrong answers?** The
opposite. An image's loss is $-\log$ of the probability given to its true class; the
surer the network is of a wrong answer, the less is left for the true class: a
probability of 0.10 costs $-\log 0.10 = 2.30$, one of 0.01 costs
$-\log 0.01 = 4.61$. On the right answers the loss is already near 0 and can fall only
a little; on the wrong ones it can rise without limit.

**So it gets more images wrong?** About the same ones: about 19 of the 360 at the end.
It gets them wrong with growing conviction that it is right. Atom 14 shows the same
thing on three images: accuracy fixed, mean loss rising.

**How does this connect to lesson 5?** Lesson 5 saw the gap between training and
validation grow with the complexity of the model; here the model is fixed and the gap
grows with the epochs. That is what makes the first remedy possible: stop at the right
epoch.

---

## Slide 58 — Two remedies for overfitting

**Early stopping** watches the validation loss after every epoch; when it has not
improved for 25 epochs, training stops and the weights of the best epoch are put back.
It needs a validation set, lesson 5's discipline again. **Dropout**: during training,
for every example separately, a random 40% of the hidden activations are set to 0 and
the rest are scaled up by $1/0.6$, so their total is unchanged on average; for
predictions it is off.

Both are **regularisation** in lesson 3's sense: they limit what the network can
memorise without changing the data. Early stopping limits how far the weights travel
from their small starting values; dropout stops any unit from relying on a particular
other one.

**Does early stopping stop at exactly the best epoch?** No: it waits 25 epochs past it,
then restores the best weights.

---

## Slide 59 — Dropout, drawn

![](dropout_drawn.png)

A small network with two hidden layers of five, so every unit is visible. For one
training example two of the five units in each layer are off: their outputs are 0, so
their connections carry nothing. For the next example a different two are off. At
prediction time every unit works. Nothing is removed for good: each example trains a
different thinned network, so what is learned is spread over many units.

**Does dropout delete 40% of the network?** No: it switches off a different 40% for
every example, during training only. Atom 15 shows the zeros and the 1.667s that
Keras's `Dropout(0.4)` produces.

---

## Slide 60 — Two remedies, measured

Three runs each, on the 360 test images. Read the spread before the means: the best
and worst rows differ by $0.9481 - 0.9426 = 0.0055$, under two standard deviations of
the runs, $2 \times 0.0035 = 0.0070$. The three methods cannot be ranked. What early
stopping certainly bought is time: 39 epochs instead of 200.

**Why more epochs with dropout?** With a patience of 25 the epochs run are the best
epoch plus 25: 39 means the best was epoch 14 on average, in line with the minimum at
epoch 13 of slide 56; with dropout $50.3 - 25$ is about epoch 25, because dropout slows
learning and the validation loss keeps improving for longer.

**Why a larger spread with the remedies?** First measure it: one image of 360 is
$1/360 = 0.0028$ of accuracy, so the standard deviations are half an image and a
little over one, from three runs each, too few to trust the difference. If there is a
reason, it is that after 200 epochs the three networks have finished memorising and
settle on similar answers, while early stopping halts each run at a different epoch.

**Can we say dropout gives the best model?** No: ask how many runs, and what spread.

---

## Slide 61 — The sigmoid's slope is at most 1/4; an active ReLU's is 1

![](activation_functions.png)

Left: the sigmoid and the ReLU. Right: their slopes. The sigmoid's slope,
$\sigma(z)(1 - \sigma(z))$, is never above one quarter, $0.5 \times 0.5$ at its very
centre, and much smaller elsewhere: at $z = 5$ it is 0.0066, the sigmoid almost flat
("saturated"). The ReLU's slope is exactly 1 wherever the unit is active, so the blame
passes through unchanged, and 0 where it is not.

**For $z < 0$ the ReLU learns nothing?** From that example, nothing: its slope is 0. But
whether a unit is on depends on the example, and it learns from the examples on its
side of its line. The real risk is a unit off for every example: it never receives
blame again, and is dead, as most units were at the learning rate of 2.0 on slide 54.
A common fix is the **leaky ReLU**, which keeps a small slope, such as 0.01, for
negative $z$.

**Why is the ReLU's slope 0 on the negative side?** Its output is constant there: a
small change in the input changes nothing.

---

## Slide 62 — Six sigmoid layers, plain SGD: stuck at chance for 80 epochs

![](deep_training_curves.png)

Six hidden layers of 32, identical in everything but the activation, trained with
plain SGD at learning rate 0.1 for 80 epochs, one run each. The sigmoid network sits on
0.1, guessing on ten classes, for all 80 epochs; the ReLU network, same size, same
data, same optimiser, learns normally. The code for the two differs by one word.

**Does this prove sigmoids cannot be used in deep networks?** No: it is one experiment,
plain SGD at 0.1 for 80 epochs. Better starting weights or another optimiser can
rescue a sigmoid network; the ReLU removes the cause.

---

## Slide 63 — The vanishing gradient

Backpropagation multiplies the blame by each layer's activation slope (slide 35). With
sigmoids that slope is at most 1/4, so through six layers the slopes alone cut the
blame to at most $0.25^6 = 1/4096$: too little reaches the first layers to move them.
They keep their random starting weights, the layers after them receive inputs that
carry nothing learned, and the network never leaves chance: 0.1000 against the ReLU's
0.9611. This is the **vanishing gradient**.

**Is it because the sigmoids are saturated?** Saturation would make it worse, but it is
not needed: Keras starts the weighted sums near 0, where the sigmoid is steepest, and
six factors of 1/4 are already enough. The trouble is the product of many small
factors, not one flat sigmoid. Atom 16 measures it: with sigmoids the first layer's
blame is 0.0006 of the last hidden layer's; with the ReLU, 0.57.

**Is 1/4096 the whole gradient?** No: the blame is also multiplied by the weights it
crosses on the way back; small starting weights shrink it further, large ones could
enlarge it. 1/4096 describes the mechanism, not a bound.

**Can a ReLU network never have a vanishing gradient?** It removes one cause, the
repeated small slope; an inactive ReLU has slope 0, and the weights still matter. The
problem comes back in lesson 10's deeper networks.

---

## Slide 64 — Shuffle the pixels the same way in every image, then train again

![](permuted_pixels.png)

One fixed reordering of the 64 pixels, applied to every image, in training, validation
and test alike; then the one-hidden-layer network of slide 42 is trained again, from
scratch, on the shuffled images. Top row: five digits as written; bottom row: the same
five shuffled. To a person they are noise.

Most people expect the score to collapse, and that is reasonable: we read digits
through their shapes.

**Why the same shuffle for every image?** So that each input position keeps one meaning
throughout: a different shuffle per image would destroy the information, not rename it.

---

## Slide 65 — Nothing notices

The network scores 0.9602 on the digits as written and 0.9630 shuffled, three runs
each: 0.0028 apart, exactly one image of the 360, inside the standard deviations of
0.0052 and 0.0035. A dense layer gives every pixel its own weights and has no
built-in notion of which pixel is next to which: there is an equivalent network, the
same weights with the rows of $W^{[1]}$ reordered like the pixels, and training finds
one as good. **Permute the pixels consistently and nothing here notices.**

**Train on the originals and shuffle only the test images: same score?** No: it
collapses. Atom 17 shows it with logistic regression: 0.9556 either way when trained
on what it is scored on, 0.0639 when only the validation images are shuffled.

**So a dense network cannot recognise shapes?** It recognises digits well; it can learn
spatial patterns from examples, but nothing in its structure assumes them. For a
photograph, where neighbouring pixels are the whole point, that is a strange blindness,
and lesson 10 is about networks built to use it.

---

## Slide 66 — Notebook 3, live

Notebook 03, training in practice, or atoms 13-17. If time is short, its sections 3
and 5 are the two to keep: the validation loss climbing while the accuracy stays level,
and the sigmoid network stuck at 0.1000.

---

# Part VI — The close

## Slide 67 — What to take away

Four sentences for the whole lesson. One neuron is one line; a hidden layer gives the
data new coordinates in which one line is enough. Learning is a loop: predict, measure
the loss, backpropagation shares out the blame, a gradient step moves the weights. Low
accuracy on training and validation alike: try learning rates a factor of ten apart
before a bigger network. Fit the linear model first: its gap to a network says whether
one is needed.

---

## Slide 68 — 39 points against 2.8

The number to carry out: 39 points against 2.8. The 39.4 is $0.9445 - 0.5507$ on the 750
test sensors; the 2.8 is $0.9602 - 0.9324$ on the 360 validation images. A network is
not better than a linear model in general; it is better when the inputs act in
combination, when one input's vote depends on another, and the only way to know which
kind of problem you have is to measure the gap.

**So networks are only useful when features interact?** That is not what the two numbers
say. They say that the value of a hidden layer depends on the problem, and that you
measure it rather than assume it.

---

## Slide 69 — Where networks went next

![](network_timeline.png)

Not examinable. Today's networks had one or two hidden layers and a few thousand
weights; the networks behind image recognition, game playing and protein science are
much bigger, and built from the same pieces: weighted sums, the ReLU, softmax,
backpropagation, Adam. In 1986 Rumelhart, Hinton and Williams made backpropagation the
standard. In 1989 Cybenko proved that one hidden layer of sigmoid units can approximate
any continuous function as closely as wanted, given enough units; in practice "enough"
can be enormous, one reason networks went deep instead. The same year LeCun and
colleagues trained a convolutional network to read handwritten ZIP codes. In 2012
AlexNet won the ImageNet image-recognition contest by a wide margin, with the ReLU,
dropout and graphics processors; in 2015 ResNet trained a network 152 layers deep; in
2016 AlphaGo beat Lee Sedol at Go; in 2020 AlphaFold predicted the shapes of proteins
from their sequence.

None of this contradicts the lesson's number: these results came from depth, very large
data sets and a structure matched to the data, and for images that structure is
convolution, lesson 10.

---

## Slide 70 — Homework

Exercise 9, discussed at the start of the next lesson. Two batches from Meridian's two
final-test stations, 3,000 units each, eight columns, a pass rate near 0.50, the same
3% rig error: everything is the same except the shape of the rule that decides the
verdict. On one line a hidden layer is the difference between a model that works and
one that does not; on the other it buys nothing measurable.

What is assessed is the prediction made before fitting: read the generator's
description, plot the columns, say which line needs a hidden layer and why, and only
then measure, with the number of runs and the spread, against the ceiling of 0.97.

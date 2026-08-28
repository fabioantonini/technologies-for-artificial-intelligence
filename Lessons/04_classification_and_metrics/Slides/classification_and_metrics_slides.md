---
title: "Lesson 4: Classification and Evaluation Metrics"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "16 October 2026"
---

# Before we start

- Exercise 3 was due today
- Ridge, Lasso, and the choice of the penalty strength

::: notes
Collect exercise 3. If you have looked at any of it, name the recurring mistake
rather than praising the good answers - it is worth more to the room.

The one to watch for: choosing the penalty by looking at the test error. It is
the natural thing to do because the test error is the number they were told to
care about, and it is exactly the sin lesson 1 named. Today's lesson ends up in
the same place from a different direction, when we choose a decision threshold.
Lesson 5 gives them the machinery to do both honestly.
:::

# Today: from a number to a decision

- Logistic regression: the model and why the cost had to change
- The confusion matrix, and why accuracy lies
- Precision, recall, F1, and the threshold nobody chose
- Receiver operating characteristic (ROC) curves, and where they mislead
- Choosing a threshold from money

::: notes
Agenda. Flag the proportion out loud, because it is the point of the lesson:
about a third of today is the model, two thirds is how to tell whether it works.

That is not padding. Fitting a classifier is one line of scikit-learn. Deciding
whether the thing you fitted is any good, and reporting that honestly, is where
the difficulty actually lives - and it is where working data scientists most
often get it wrong in public.

If anyone asks why this comes before k-nearest neighbours and support vector
machines: because every one of those needs the same evaluation apparatus, and it
is better learned once, here, on the simplest possible model.
:::

# The question has changed

- Lesson 3: given a house, **how many euros?**
- Lesson 4: given a disk drive, **will it fail?**

::: notes
Draw the distinction sharply. Regression predicts a quantity; classification
predicts a category. The change sounds cosmetic and forces two separate changes
of machinery.

First, the model has to change, because a straight line is unbounded and a
probability is not. Second, the evaluation has to change, because "how far off
were we on average" means nothing when the answer is yes or no.

Ask the room for other examples of the second kind before moving on. You want
them to notice that the interesting class is nearly always the rare one: fraud,
disease, intrusion, equipment failure. That observation is the whole second half
of today.
:::

# The fleet

- **8,000 disk drives** in a data centre
- Six SMART counters each: Self-Monitoring, Analysis and Reporting Technology
- Label: did it fail within thirty days?

::: notes
SMART is the diagnostic data a drive keeps about itself: reallocated sectors,
spin retries, read error rate, temperature, hours in service. Expand the acronym
on the slide and out loud - it is the kind of thing that gets skipped and then
quietly blocks a student for the rest of the hour.

The data is synthetic, and say why: the coefficients that generated the labels
are written down in disk_data.py. Same argument as lesson 3 - a real dataset
lets you admire an estimate, a synthetic one lets you check it.

One column, seek_error_rate, was generated with a coefficient of exactly zero.
Do not reveal that yet. It comes back in the last section of notebook 1.
:::

# 306

- **306 of the 8,000 drives failed.** 3.8% of the fleet
- Which means 96.2% did not

::: notes
Put this number on the board and leave it there. It is the number the whole
lesson turns on, and they will meet it again as an accuracy score in an hour.

Ask, without answering: if a model has to guess for each drive, and it wants to
be right as often as possible, what should it say?

Somebody will say "healthy, every time". Agree, and move on. Do not explain the
consequence yet - it lands far harder in section 5 when they see 96.2% printed
as a model's accuracy score.
:::

# What goes wrong if we just fit a line

![](linear_vs_logistic.png)

::: notes
This is notebook 1, section 1. Ordinary least squares on a 0/1 label, one
feature. Nothing prevents it and the code runs.

The left panel: the fitted line gives a NEGATIVE probability to 3,606 of the
8,000 drives - 45% of the fleet. Not a small negative number. A negative
probability. And extend it slightly past the data, to 32 reallocated sectors,
and it predicts more than 1.

Say clearly that this is not a defect better fitting would remove. A straight
line is unbounded, a probability is not; no slope and intercept reconcile them.
The shape of the function has to change, not its parameters.

The right panel is where we are going. Same data, a curve that cannot leave the
interval.
:::

# The sigmoid

![](sigmoid_and_odds.png)

::: notes
Left: the curve. Steep in the middle, flat at both ends. Make the case that the
flatness is the right behaviour, not a compromise - going from 0 to 4
reallocated sectors should change your mind a lot; going from 40 to 44 should
barely register, because you had already concluded the drive was finished.
Evidence has diminishing returns.

Right: the same relationship inverted. This is the slide that explains WHY this
particular curve and not some other S-shape. A probability is trapped in [0,1].
The odds - p over 1 minus p - remove the ceiling. The log of the odds removes
the floor too, leaving something that runs over all of the real line.

And that is exactly what a linear model can safely predict. So the sigmoid is
not a squashing trick bolted onto linear regression: it is what you get when you
model the log-odds linearly and then ask for the probability back.
:::

# The model, in one line

$$P(y = 1 \mid x) = \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

::: notes
Read it aloud as a sentence: take the linear model of lesson 3, and instead of
calling its output a prediction, call it the log-odds and convert.

Name the three properties we will use, without deriving them here - the handout
section 2.2 does that: sigma of 0 is a half; sigma of minus z is 1 minus sigma
of z; and the derivative is sigma times one minus sigma.

That last one is the one to flag. It is largest in the middle, one quarter, and
it decays to zero at both ends. In fifteen minutes that fact will be
simultaneously the reason the sigmoid works and the reason squared error does
not.
:::

# What the coefficients mean

- The model is linear in the log-odds, so `exp(coefficient)` multiplies the **odds**
- Features standardised, so one unit = one standard deviation
- A coefficient of 1.80 means **six times the odds** of failing

::: notes
This is the interpretability that keeps logistic regression in production in
medicine and credit scoring. A coefficient of 1.80 means: a drive one standard
deviation worse on that counter has exp(1.80), about six times, the odds of
failing.

Now the mistake they will make, and it is a reasonable one. They will read "six
times the odds" as "six times as likely". For rare events those are nearly the
same number, so here it is almost right - which is exactly why the habit
survives until it does damage.

Give them the counterexample out loud: odds of 9, a probability of 0.90,
multiplied by 5 gives odds of 45 - a probability of 0.978. The odds went up
fivefold; the probability moved eight points. Odds ratios multiply odds.
:::

# The truth, and what we recovered

| Feature | True | Fitted | Odds × |
|---|---|---|---|
| reallocated_sectors | 1.80 | 1.66 | 5.28 |
| spin_retry_count | 1.15 | 1.17 | 3.22 |
| read_error_rate | 0.85 | 0.87 | 2.39 |
| temperature_c | 0.45 | 0.47 | 1.59 |
| power_on_hours | 0.35 | 0.38 | 1.46 |
| seek_error_rate | **0.00** | **0.02** | **1.03** |

::: notes
Notebook 1, last section. This is the payoff of using synthetic data: we are not
admiring the estimates, we are marking them.

Go down the column. Everything is close. Nothing is exact, and say why - 6,000
training drives carrying 230 failures is a modest amount of evidence, and 230 is
the number that matters, not 6,000.

Then the last row, which is the one to pause on. seek_error_rate was generated
with a coefficient of exactly zero: a real-looking counter with no connection to
failure. The model estimates 0.02, an odds multiplier of 1.03. It did not fall
for it.

Then ask the room the harder question: would you have noticed, if I had not told
you? Leave it hanging. It is lesson 5.
:::

# The intercept is the base rate

- Fitted intercept: **−6.09**
- $\sigma(-6.09) = 0.0023$: a 0.2% chance for an average drive

::: notes
This is the slide that quietly explains everything in the second half of the
lesson, so do not rush it.

An average drive has a 0.2% chance of failing in thirty days. For the model to
reach an even bet - probability one half, log-odds zero - the evidence has to
move the log-odds by more than six. Only badly degraded drives manage that.

So the model will assign small probabilities to almost everything, and a
threshold of 0.5 will flag almost nothing. That is not the model being timid. It
is the model being correct about a rare event. Everything that looks like a
failure of the classifier in the next hour traces back to this number.
:::

# Fitting means minimising something, but what?

- Squared error worked for lesson 3
- Why not just apply it to the sigmoid's output?

::: notes
Put the question to the room and take answers before showing the next slide.
This is a good place for them to be wrong out loud, because the honest answer is
not obvious and the usual first guesses ("it's not differentiable", "it doesn't
work for categories") are wrong.

Squared error on the sigmoid IS differentiable, and it does produce a working
model on easy problems. The reason to reject it is subtler and more interesting,
which is what makes it worth ten minutes.
:::

# Maximum likelihood, in one sentence

- The model has a knob; each setting assigns a probability to what happened
- **Turn the knob so that what actually happened is least surprising**

::: notes
Say this before any algebra. It is the whole idea, and once they have it the
derivation is bookkeeping.

Each drive contributes one factor: the probability the model gave to what
actually occurred. Multiply across the fleet and you have the probability of the
whole dataset under that setting of the parameters. Choose the setting that
makes it largest.

Two practical notes worth saying aloud. We take the logarithm because it turns
the product into a sum, which differentiates one term at a time - and because
with 8,000 factors below 1 the product is about 10 to the minus 1000, which a
64-bit float stores as exactly zero. Every real implementation works in log
space, and not for elegance.

Then negate, so we minimise as in lesson 3.
:::

# Cross-entropy, also called log loss

$$J(w, b) = -\frac{1}{m}\sum_{i=1}^{m} \left[\, y^{(i)} \log p^{(i)} + \left(1 - y^{(i)}\right)\log\left(1 - p^{(i)}\right) \right]$$

::: notes
Do not derive it here - the handout, section 3, does it in full. Show it and
read it.

The reading that makes it stick: the cost of an outcome is the SURPRISE of
seeing it. Something you called certain costs nothing. Something you gave a
probability of one in a thousand costs log 1000, about 6.9. Something you
declared impossible costs infinity.

Only one of the two terms is ever active, because y is 0 or 1. The bracket is a
way of writing an if-statement in algebra, nothing more.

And the headline: nobody designed this function. It fell out of asking for the
parameters that make the data least surprising.
:::

# The gradient is the one you already know

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x_j^{(i)}$$

::: notes
Prediction error times feature, averaged. Character for character the gradient
of lesson 3, with a different model and a different cost function.

The cancellation is worth showing on the board if there is appetite: the sigma
times one-minus-sigma from the chain rule cancels against the one-over-sigma
from the logarithm. Handout section 3.4 has it.

If someone asks whether that is a coincidence: no. Both are generalised linear
models fitted by maximum likelihood - Gaussian noise for regression, Bernoulli
outcomes for classification - and the whole exponential family gives this form.
They do not need that theory to use either method, but it is why the same
gradient descent code, unchanged, trains both.
:::

# So why not squared error

![](loss_comparison.png)

::: notes
Left panel, the costs. A drive failed and the model said 0.0001. Squared error
charges 0.9998. An honest "don't know" - 0.5 - costs 0.25. So the worst possible
answer is four times worse than a shrug. Log loss charges 9.2 against 0.69, and
the ratio grows without bound.

Right panel is the real argument, so spend the time here. The gradient is what
the optimiser actually uses. Squared error's gradient through the sigmoid
carries a factor p times one-minus-p - which is exactly the sigmoid derivative
from ten minutes ago - and that factor goes to zero as p goes to zero.

Say it as a sentence: the wronger the model gets, the less squared error asks it
to change. At p = 0.001 the gradient is a five-hundredth of log loss's. Log loss
goes the other way, approaching its maximum push exactly where the error is
worst.

That is the failure. Not that squared error scores badly - that it stops
teaching.
:::

# And it is convex, which squared error is not

- Log loss is **convex** in the weights, so any local minimum is global
- Squared error through a sigmoid is not

::: notes
State it, do not prove it; the handout section 4.3 has the argument.

The consequence is practical: gradient descent on log loss cannot get trapped.
Squared error through a sigmoid has no such guarantee and is in general
non-convex.

Now the caution that catches people, and it is worth ten seconds because they
will hit it in the notebook. Convex does NOT mean there is a closed-form
solution. Setting the gradient to zero gives X-transpose times sigma of Xw minus
y equals zero, which is not linear in w - the sigmoid is in the way.

There is no normal equation for logistic regression. It has to be solved
iteratively, which is why LogisticRegression has a max_iter argument and
LinearRegression does not. If a student ever sees a convergence warning, that is
what it is about.
:::

# Notebook 1, live

- Fifteen lines of NumPy, then one line of scikit-learn
- Same answer

::: notes
Run notebooks/01. Twenty minutes. The parts worth stopping on:

The linear-fit failure - let them see 3,606 negative probabilities printed
rather than being told.

The gradient/loss figure, which is the argument you just made, in code they can
change. Invite them to change it: what happens to the squared-error gradient at
p = 0.5?

The coefficient recovery table, including the decoy.

If time is short, skip the descent curve - it is the least surprising cell and
they saw the same picture in lesson 3.
:::

# The decision boundary

![](decision_boundary.png)

::: notes
Two features, so the model can be drawn. The black line is where the predicted
probability is exactly 0.5 - equivalently, where the log-odds are zero.

First observation: it is STRAIGHT. Logistic regression is a linear classifier.
The curve is in the probabilities, not in the boundary. That surprises people
who expected the sigmoid to bend it.

Second observation, and this is the bridge to the rest of the lesson: almost
every gold point - every drive that actually failed - is on the blue side of the
line. At a threshold of 0.5 this model calls nearly the whole fleet healthy,
including most of the drives that failed.

Ask: is that a bad model? Take answers. Then say: the probabilities are fine.
The threshold is a habit. Telling those two apart is the next two hours.
:::

# Break

- Ten minutes

::: notes
Ten minutes. When they come back, the first slide is the trap, and it works best
if they have not been thinking about it.
:::

# Two models

| Model | Accuracy | Failures caught |
|---|---|---|
| Predict "healthy" for everything | **96.20%** | **0 of 76** |
| Logistic regression, threshold 0.5 | 97.70% | 43 of 76 |

::: notes
This is the number to remember from lesson 4. Put it up and say nothing for a
moment.

A model that could be written `return 0` - no learning, no data, no features  - 
scores 96.20%. The model they just built, which finds more than half the failing
drives, scores 97.70%. One and a half percentage points separate a real model
from a worthless one.

The arithmetic is not subtle: 1,924 of the 2,000 test drives are healthy, so
answering "healthy" collects them all for free. Only 3.8% of the test set is
left to compete over.

This is the 306 from the first ten minutes, coming back as an accuracy score.
:::

# Why everyone makes this mistake once

- Accuracy is the fraction of predictions that were right
- On a **balanced** problem that is exactly what you want
- The interesting class is almost always the **rare** one

::: notes
Be careful to defend the instinct rather than mock it. Accuracy is not a naive
metric, and students who reached for it were not being lazy - on a balanced
problem it is the correct summary and everything else is a complication.

It stops being informative precisely when one class is rare. And then list the
problems: fraud, disease screening, intrusion detection, equipment failure,
spam. In every one of them the class you care about is the small one.

The general statement, worth putting on the board: with a positive rate pi, the
trivial always-negative classifier scores 1 minus pi. Here 96.2%. In some fraud
problems, where the rate is one in a thousand, 99.9%.

Accuracy is not wrong. It is answering a question nobody asked.
:::

# Four numbers instead of one

![](confusion_matrix.png)

::: notes
The fix is to stop collapsing four outcomes into one number.

Side by side, the two models are obviously not comparable - and the number that
separates them, 0 caught against 43, is nowhere in the accuracy figure.

Teach them to read the bottom-left cell first when misses are expensive: 33
drives failed and were called healthy. That is the cell that costs money, and it
is the one a single-number metric hides.

Worth naming the convention while it is on screen, because half the world uses
the transpose: scikit-learn puts truth in rows and predictions in columns. Any
time they read a confusion matrix from a paper, check which way round it is
before drawing conclusions.
:::

# The four outcomes, and what they cost

- **False positive**: we replace a healthy drive. A technician's hour, €140
- **False negative**: a drive dies in service. Callout, rebuild, risk, €2,600

::: notes
The names are jargon until they are attached to consequences, so attach them
here and keep the numbers visible - they come back at the end of the lesson when
we choose a threshold.

The essential point: these two errors are NOT the same size, and no
single-number metric knows that. Accuracy treats them as identical. F1 treats
them as identical. Only a cost, supplied from outside the data, distinguishes
them.

Ask for a domain where the ratio goes the other way - where a false positive is
the expensive one. Medical screening that triggers an invasive biopsy is the
usual answer; content moderation that removes a legitimate post is another.
:::

# Precision and recall: look at the denominator

![](metric_denominators.png)

::: notes
This picture is the antidote to the most reliable confusion in the lesson.
Both metrics divide the true positives. The difference is entirely in what goes
underneath, and the shading shows it.

Precision divides by the COLUMN - everything we flagged. When this model raises
an alarm, how often is it right? That is the technician's question: how much of
my time are you wasting?

Recall divides by the ROW - everything that actually failed. Of the drives that
were going to fail, how many did we find? That is the operations manager's
question: how exposed am I still?

Give them the hook and make them repeat it: precision is about the alarms,
recall is about the failures. When in doubt, find the denominator and ask which
population it is.
:::

# Our model, in those terms

- Precision **0.768**: of 56 drives flagged, 43 really failed
- Recall **0.566**: of 76 drives that failed, we found 43
- Specificity 0.993: of 1,924 healthy drives, we left 1,911 alone

::: notes
Read each line as an English sentence with the counts in it, not as a ratio.
That is the habit worth building: a metric that cannot be said as a sentence
about actual objects has not been understood.

Specificity is recall for the negative class, and one minus it is the false
positive rate, which comes back in the ROC section. Mention it now so the term
is not new later.

If someone asks which of the three to report: the answer is that it depends on
who is asking, which is the honest answer and also the lesson.
:::

# F1: why the harmonic mean

| Model | Precision | Recall | Arithmetic | Harmonic |
|---|---|---|---|---|
| Flag every drive | 0.038 | 1.000 | 0.519 | **0.073** |

::: notes
Take the stupidest possible model - flag every single drive - and score it.
Recall is a perfect 1.0. Precision is 0.038.

The ordinary average gives that model 0.519, which looks respectable. Anyone
scanning a results table would not blink at 0.52.

The harmonic mean gives it 0.073, which is the truth.

The reason: the harmonic mean is dominated by the smaller of the two numbers. A
model is only as good as its weaker side, and F1 refuses to let a perfect score
on one axis buy a pass on the other.

Mention F-beta in one sentence: beta above 1 weights recall more, which is what
you want here where misses cost nineteen times more. F2 is the usual choice.
:::

# Read the minority class, not the average

- `weighted avg` F1: **0.975** (dominated by the healthy class)
- `macro avg` F1: **0.820** (both classes get an equal say)

::: notes
classification_report prints three summary rows and students quote whichever is
biggest. Give them the rule.

The weighted average weights each class by how common it is, so on this problem
it is essentially the healthy class talking, and it reads close to accuracy  - 
inheriting exactly the blindness we spent twenty minutes on.

The macro average gives the two classes equal weight, so the rare class actually
registers. 0.820 against 0.975 for the same model.

The rule: on an imbalanced problem, quote the macro average or the minority
class directly. Reporting the weighted average is not incorrect - it is the
number that makes a mediocre model look finished, and choosing it without saying
so is how a result gets oversold.
:::

# Notebook 2, live

- The confusion matrix by hand, then from scikit-learn
- Then the threshold sweep

::: notes
Run notebooks/02. Twenty minutes.

Have them build the four counts by hand before calling confusion_matrix. It is
four comparisons and it takes two minutes, and it is what makes the jargon stop
being jargon.

The cell to linger on is the threshold sweep table. Do not explain it first  - 
let them read the columns and say what they see. Somebody will notice that
accuracy barely moves. That is the moment to make the next slide.
:::

# The threshold is a choice nobody made

![](threshold_sweep.png)

::: notes
The title of that figure is the point: nothing about the model changes along the
x-axis. Same coefficients, same probabilities, same drives. All that moves is
where we draw the line between "leave it" and "replace it".

That 0.5 is predict()'s default and nothing more. It is the right choice when
the two errors cost the same AND the classes are balanced. Neither is true here,
and neither is true in most real problems.

So when a paper or a colleague reports 77% precision, the useful follow-up is:
at what threshold, and why that one? Most of the time there is no answer,
because nobody chose it.
:::

# What the sweep actually costs

| Threshold | TP | FP | FN | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| 0.02 | 69 | 284 | 7 | 0.195 | 0.908 | 0.855 |
| 0.10 | 61 | 102 | 15 | 0.374 | 0.803 | 0.942 |
| 0.30 | 53 | 25 | 23 | 0.679 | 0.697 | 0.976 |
| 0.50 | 43 | 13 | 33 | 0.768 | 0.566 | **0.977** |
| 0.90 | 14 | 0 | 62 | 1.000 | 0.184 | 0.969 |

::: notes
Read it column by column with the room.

Lower the threshold: recall rises, precision falls. At 0.02 we catch 69 of 76
failures and replace 284 healthy drives to do it. At 0.90 every alarm is
justified and we catch fewer than a quarter.

Now the accuracy column, which is the slide's real argument. From 0.10 upwards
it moves by three percentage points while recall falls from 0.80 to 0.18. And
its maximum is at 0.5 - the threshold that misses 33 of the 76 failures.

Say the consequence plainly: tune on accuracy and it will quietly recommend
catching fewer failures, because every alarm it drops was a possible false
positive.
:::

# The precision-recall curve

![](precision_recall_curve.png)

::: notes
The sweep traced a path through precision-recall space; this is that path drawn
directly. Every threshold at once.

The part to teach is the baseline. A model with random scores has precision
equal to the positive rate - 0.038 here - at every recall. That flat red line,
not 0.5, is what "no skill" looks like on an imbalanced problem.

That is why this curve is so much more informative than accuracy when positives
are rare: the floor is where the difficulty actually is. Our model's area under
it, the average precision, is 0.717 against a floor of 0.038.

The three marked dots are the same three thresholds from the last two slides, so
they can see the correspondence.
:::

# Comparing models without choosing a threshold

- We keep evaluating at one threshold, but which?
- What if we score the whole **ranking** instead?

::: notes
Set up the next section as a question rather than announcing a tool.

Everything so far has needed a threshold. That is fine when you know the costs,
and we will come back to it. But often you want to compare two models before
committing - is model A better than model B, at any operating point?

The move is to stop asking "how many did it get right" and start asking "did it
put the failures at the top of the list". That is a question about ranking, and
ranking has no threshold in it.
:::

# The ROC curve

![](roc_curve.png)

::: notes
Receiver operating characteristic - expand it, and say where the name comes
from, because it makes the idea concrete. Wartime radar: operators tuning
receivers to spot aircraft without chasing flocks of birds. Same problem shape,
literally the same trade-off.

The two axes: true positive rate, which is recall under another name - the
failures we catch. False positive rate - the healthy drives we condemn.

The intuition worth saying slowly: at threshold 1 both are zero, flag nothing.
At threshold 0 both are one. The entire content of the curve is the ORDER in
which they get there. A good model raises TPR fast while FPR is still low,
because its highest scores really are the failures. A useless model raises them
together and traces the diagonal.

So the curve asks: as we grow more willing to raise alarms, do we catch failures
faster than we annoy technicians?
:::

# What the area means

- **AUC = the probability that a random failing drive scores above a random healthy one**
- Ours: 0.949

::: notes
This is the definition to remember, and it is much more useful than "area under
the curve", which is a geometric fact rather than a meaning.

Say it concretely: pick one drive that failed and one that did not, at random.
95 times out of 100 the model gave the failing one a higher score.

That makes AUC a measure of RANKING quality. It is threshold-free because
ranking is threshold-free.

The identity with the Mann-Whitney U statistic is in the handout, section 8.2,
for anyone who wants the proof. But notebook 3 does something better than a
proof for this audience: it draws 200,000 random pairs and counts. 0.9488
against roc_auc_score's 0.9493.
:::

# What AUC does not tell you

- Divide every predicted probability by 10
- AUC is **unchanged**; every decision changes

::: notes
Short slide, important consequence. AUC depends only on the ordering, so it is
invariant under any monotone transformation of the scores.

Divide all the probabilities by ten and the ranking is identical, so the AUC is
identical - while the model is now badly calibrated and every threshold-based
decision it makes is different.

The consequence: AUC says nothing about whether "0.7" means anything. If you
need the probabilities to be believable - because you are computing expected
costs, as we will in ten minutes - AUC is not the check you want.

Good place to mention calibration exists as a topic, without going into it.
:::

# Where ROC misleads

![](roc_vs_pr_imbalance.png)

::: notes
Same model. Not retrained. We keep every healthy drive in a fleet of 60,000 and
thin the FAILURES until they are 0.4% instead of 3.8% - the direction reality
moves in fraud, disease, hardware failure.

Left: the three ROC curves sit on top of each other. AUC 0.968, 0.965, 0.974. It
reports an excellent model throughout, and at the rarest setting a marginally
better one, which is sampling noise on 231 positives.

Right: the precision-recall curves are in three different worlds. Average
precision falls from 0.706 to 0.361, and precision at the same threshold falls
from 0.79 to 0.26 - two alarms in three now false.

Three curves on the left, one model, no problem. Three curves on the right, same
model, a disaster. Let that sit for a second before explaining.
:::

# Why the two disagree

- FPR divides by **all healthy drives**: an enormous, growing denominator
- Precision divides by **the alarms raised**, where the false ones dominate

::: notes
The explanation is entirely in the denominators, and it is worth writing on the
board.

A thousand false alarms among a hundred thousand healthy drives is a false
positive rate of 0.01 - invisible on the ROC axis, a rounding error. The same
thousand false alarms, set against the alarms actually raised, may be most of
them.

So ROC is measuring the false alarms against a population nobody experiences.
The technician does not see the 99,000 drives left alone; they see the alarm
queue.

This is the second predictable mistake of the lesson, and again defend the
instinct: AUC is threshold-free, comparable across models and datasets, bounded
in a familiar range. Those are real virtues. It simply answers a question about
ranking when the question was about the alarm queue.
:::

# The rule

- On an imbalanced problem, report the **precision-recall curve alongside AUC**
- Never instead of it, and never AUC alone

::: notes
Short and quotable, and this is the thing to have on the board when they write
their exercise reports.

If someone asks which single number to use when a single number is demanded:
average precision, because its baseline moves with the problem. But push back on
the premise - the reason to demand a single number is usually to avoid thinking,
and this lesson is about the thinking.
:::

# Choosing the threshold from money

![](cost_curve.png)

::: notes
Finally, the honest way to pick a threshold. It needs one ingredient no metric
can supply: what the errors cost. We wrote those down an hour ago - €140 for a
needless replacement, €2,600 for a drive that dies in service.

The curve is the total cost on the test set as the threshold moves. The default
0.5 costs €87,620. The cheapest, at 0.08, costs €45,540.

Choosing the threshold on purpose HALVED the cost of the same model. No
retraining, no new features, no better algorithm. One number, taken from the
economics instead of from the library default.

Note the flatness around the minimum, and say why it is good news: the exact
threshold does not have to be right. Which is fortunate, because cost estimates
never are.
:::

# And it has a closed form

$$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{140}{140 + 2600} = 0.051$$

::: notes
Derive it in one line on the board, because it is two steps and it lands: flag
the drive if the expected cost of flagging is lower than the expected cost of
not. Flagging costs C-FP if the drive was healthy, which happens with
probability 1 minus p. Leaving it costs C-FN if it fails, probability p. Set one
below the other and solve.

Then read the formula rather than the number, which is the part worth
remembering: the optimal threshold does not depend on the model, the dataset, or
the class balance. Only on the RATIO of the two costs.

If the errors cost the same, t-star is 0.5 and the default was right all along.
As misses grow more expensive, it falls towards zero.

Empirically we found 0.08 against a theoretical 0.051, and the curve is flat
between them.
:::

# The caveat you must state

- We chose the threshold **by looking at the test set**
- That is precisely what lesson 1 forbade

::: notes
Say this out loud rather than leaving it in the handout, because it is the same
mistake you may have just returned to them on exercise 3.

The threshold is a parameter. Choosing it by looking at the test set means the
test set has influenced the model, and the number it then reports is optimistic.

Done properly: choose the threshold on a validation set, then measure once on
test. Lesson 5 builds exactly that machinery, and this is the concrete reason it
is needed - not a hygiene rule, a thing that changes the number you report.

Until then, treat the 0.08 as a demonstration of a method, not a result anyone
could publish.
:::

# Class weights: the same lever, pulled earlier

| Model | Recall | Precision | AUC | Cost |
|---|---|---|---|---|
| Plain, threshold 0.50 | 0.566 | 0.768 | 0.949 | 87,620 |
| Plain, threshold 0.08 | 0.855 | 0.349 | 0.949 | 45,540 |
| Balanced weights, 0.50 | 0.895 | 0.249 | **0.950** | 49,500 |

::: notes
class_weight="balanced" makes each rare example count as many, here about 25.
It changes the training rather than the decision.

Point at the AUC column: 0.949 against 0.950. Reweighting taught the model
essentially nothing new about disk failure. What it did was move where the model
puts its 0.5 line - the same lever as the threshold, pulled at a different
moment.

That matters because class weighting is usually presented as a REMEDY for
imbalance. It is not a remedy; it is a reparameterisation of the same decision.
The remedy, where one exists, is more positive examples.

Where it earns its place: when a downstream tool insists on predict() and gives
you no threshold to move. Then this is how you shift the operating point.

Same caution for resampling and SMOTE - and it must happen inside the
cross-validation fold, for the reasons lesson 2 gave about imputation.
:::

# Notebook 3, live

- ROC by hand, the 200,000-pair experiment, the cost curve
- Then the multiclass matrix

::: notes
Run notebooks/03. Twenty minutes if the clock allows, otherwise show the pairs
experiment and the imbalance comparison and leave the rest for them.

The pairs experiment is the cell worth protecting if time is short: it turns a
definition they would otherwise memorise into something they watched happen.
:::

# More than two classes

![](multiclass_confusion.png)

::: notes
Real fleets are not healthy-or-dead. A drive can be degraded: readable, but
reporting errors, and worth replacing at the next maintenance window rather than
at three in the morning.

For K classes the model produces one score per class and normalises them - the
softmax, which for two classes reduces exactly to the sigmoid. The cost
generalises to categorical cross-entropy, and it is the same idea: the negative
log of the probability assigned to what actually happened. Handout section 10.

Read the off-diagonal cells rather than the accuracy. Degraded is the hard
class, and understandably - it sits between two neighbours that both look like
it from a distance.

Precision and recall are computed one class at a time: each class in turn is the
positive one, the rest are negative.
:::

# Macro against weighted, one last time

- macro avg F1: **0.707**
- weighted avg F1: **0.883**

::: notes
The same choice as before, now with a much starker gap on three classes.

Macro treats the three conditions as equally important. Weighted lets the
healthy class - five drives in six - do most of the talking.

Which to quote is a question about the application, not about statistics. If
missing a failure matters as much as missing a healthy drive, macro is the
honest one. Say which you used, always.
:::

# What to take away

- **96.20% accuracy, nothing caught.** Accuracy measures the majority class
- Precision is about the **alarms**, recall is about the **failures**
- The **threshold is a decision**, and it has a formula
- **AUC measures ranking** and hides imbalance

::: notes
Close on the number they started with. 306 failures out of 8,000; 96.2%
accuracy for a model that does nothing.

If they remember one thing from today, it should be the reflex to ask "what's
the base rate?" before believing any accuracy figure. That reflex is worth more
than any of the formulas.

The second thing: every metric on the slide depended on a threshold, and the
threshold has an answer - C-FP over C-FP plus C-FN - that comes from the
business, not from the data.
:::

# Homework

- **Exercise 4**, due Friday 23 October
- Build a classifier, choose a threshold from a stated cost ratio, defend it

::: notes
Set it explicitly and say the deadline out loud: next Friday, 23 October.

Emphasise which part carries the marks. The classifier is a few lines and
everyone will get it. The paragraph defending the threshold choice is where the
learning is, and it is where the marks are.

Warn them about the trap deliberately planted: the dataset has a decoy column,
as this one did, and the exercise asks them to report AUC and average precision
together. Anyone reporting AUC alone will be reporting a number that looks
better than the model is.

Next week: experimental methodology. Cross-validation, bias-variance, and the
honest way to make every choice we made by hand today.
:::

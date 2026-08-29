---
title: "Lesson 5: Experimental Methodology"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "23 October 2026"
---

# Before we start

- Exercise 4 was due today
- The threshold you chose, and how you chose it

::: notes
Collect exercise 4. The paragraph defending the threshold was the part that
carried the marks, so say one sentence about what you saw across the class.

The recurring mistake to name, if it appeared: choosing the threshold by
sweeping it on the test set. The exercise sheet said that losing marks for it
was avoidable by simply declaring it - and today is the lesson that shows why
it matters rather than asserting that it does.
:::

# We broke a promise on purpose

- Lesson 4 chose a threshold **by looking at the test set**
- Lesson 1 forbade exactly that
- Today we pay the debt

::: notes
Be explicit that this was deliberate. In lesson 4 there was no machinery
available to do it properly, and inventing half of that machinery mid-lesson
would have buried the actual subject.

Keep the debt visible through the whole lesson, because nearly every error today
has the same shape: a choice was made by looking at data that was then used to
report the result.

Section 8 of the handout pays it off, and slide 39 shows the number.
:::

# Today: how to tell whether it works

- One split is a lottery, and how wide the lottery is
- Cross-validation: what it estimates, and what it does not
- Bias, variance, and the floor you cannot go below
- Learning curves: which fix to buy
- Leakage that survives cross-validation

::: notes
Agenda. Say why this lesson sits in the middle of the course, because it is a
deliberate choice and the syllabus says so.

They can already build models - lessons 3 and 4 gave them two, scikit-learn
gives a hundred more for one line each. Producing a model that appears to work
is an afternoon's work and it is not the job. Knowing whether it does work is
considerably harder.

And everything after today makes it harder still: lessons 6 to 10 add more
flexible methods, and every increase in flexibility adds ways for a result to be
wrong while looking right.
:::

# A test score is a measurement

- You would not report a poll of forty people to three decimals
- Holding data out makes the estimate **unbiased**
- It does nothing to make it **precise**

::: notes
This distinction is the whole first half of the lesson, and it is routinely
missed - because "test set accuracy" sounds like a property of the model rather
than a measurement with a standard error.

Lesson 1's rule was about bias: never let the test set influence the model, and
the number you get is centred on the truth. Correct, and it says nothing at all
about how far from the truth any particular measurement lands.

Ask the room, before the next slide: how far apart do you think two honest
splits of the same dataset can be? Take a guess out loud. They will underestimate
it.
:::

# The fleet, cut down

- Lesson 4's drives, reduced to **800**, of which **29 fail**
- Not to exaggerate: 800 rows with 29 positives is the ordinary case

::: notes
Justify the reduction explicitly, because a sharp student will otherwise suspect
the demonstration is rigged.

With 8,000 drives a single split is fairly stable, and lesson 4's numbers were
trustworthy. Eight hundred records with 29 positives is a far more common
situation than eight thousand - it is what a real pilot study looks like - and
what follows is simply what the honest version of that situation looks like.

Same data, same model, same code as last week. Only the sample size changed.
:::

# 200 honest splits, one dataset

![](split_lottery.png)

::: notes
Two hundred legitimate experiments. Every one done correctly: nothing fitted
before the split, the test set never inspected, the metric computed properly.

Worst 0.885. Best 1.000. Mean 0.955.

Sit on the 1.000 for a moment. That split reports a PERFECT classifier - every
failing drive ranked above every healthy one - for a model whose honest
performance is about 0.95. Ten of the two hundred scored above 0.99.

Then the point that matters most, and say it slowly: the right-hand tail is the
dangerous end, not the left. A disappointing score makes you frown and keep
working. A delightful score makes you stop, write it down, and put it in the
report. The splits that flatter you are the ones you are least likely to
question.
:::

# Why so wide?

- Each test set holds **7 failures**
- Every AUC (area under the receiver operating characteristic curve) rests on seven

::: notes
This is the arithmetic behind the previous slide, and it generalises into the
single most useful diagnostic question in the course.

The precision of a classification metric is governed by the count of the RARER
class, not by the size of the dataset. Eight thousand drives at 3.8% gives 306
positives and a stable estimate. Eight hundred gives 29, and it does not.

Give them the question to carry away: how many positive examples are in the test
set? If the answer is under about fifty, treat every metric as provisional.

Worth connecting to lesson 4: this is the same reason precision collapsed there
while AUC did not notice.
:::

# Now the expensive part: choosing

| Model | Mean AUC | Times declared best |
|---|---|---|
| C = 0.01 | 0.9549 | 91 |
| C = 1 | 0.9549 | 71 |
| C = 100 | 0.9544 | 38 |

::: notes
Reporting a noisy number is bad. Choosing with one is worse, because noise does
not average out when it is used to decide - it decides.

Read the first column, then the last. The three models are the SAME: their mean
performance agrees to three decimal places, a difference an order of magnitude
smaller than the spread of a single measurement.

And yet a winner is declared every single time. Those counts carry no
information about the models. They record which model happened to suit which
random quarter of the data.

Then the sentence to land: run this once, which is what real life allows, and
you will report that one of these is best with a number to support it. That
conclusion is noise wearing the clothes of a result.
:::

# The same three models, drawn honestly

![](single_split_selection.png)

::: notes
Left: the win counts, against the dashed line showing what pure chance would
give. Right: the actual distributions, sitting on top of each other.

The right panel is the truth about these three models. The left panel is what a
single experiment reports.

If anyone asks why C=0.01 won more often than the others - it is not evidence of
anything. Run the whole thing with different seeds and the ordering moves. That
is exactly the point.
:::

# k-fold cross-validation

![](kfold_diagram.png)

::: notes
The idea in a sentence: if one measurement is noisy, take several and average
them. The difficulty is that you cannot afford several test sets - every row
spent on testing is a row not spent on training.

Cross-validation resolves it by rotating the role of the test block. Cut the
data into k parts, train on k-1, test on the one left out, repeat until every
part has been the test set exactly once.

Then the clause that makes it legitimate rather than a trick, and say it
explicitly because students suspect otherwise: every row is predicted exactly
once, by a model that never saw it. This is not testing on training data with
extra steps.
:::

# What it actually estimates

- The performance of a **procedure**, not of one fitted model
- Slightly **pessimistic**: each fold trains on 80% of the data

::: notes
Two consequences, both of which surprise people.

First: there are k different fitted models in the calculation, and the reported
number belongs to none of them. It answers "if I apply this procedure to a
dataset of about this size, what should I expect?" - which is the question you
actually have.

Second: every fold trains on four fifths of the data, so it estimates the
performance of a model trained on less data than the one you will eventually
ship. The bias is in the safe direction, and it shrinks as k grows. That is the
argument for leave-one-out, where k equals the number of rows.

Handout section 4.2 has both in symbols.
:::

# Report the spread, but not as a confidence interval

- Any two training folds share **three quarters** of their rows
- The scores are correlated, so the usual standard error is optimistic

::: notes
The temptation is to quote s over root k and call it a confidence interval. Do
not, and give them the specific reason rather than a warning.

That formula assumes the k scores are independent. They are not: for k=5, any
two training sets have three quarters of their rows in common. Positively
correlated measurements carry less information than independent ones.

The handout gives the variance with the correlation term, and the result of
Bengio and Grandvalet from 2004: no unbiased estimator of that variance exists
in general.

Practical instruction: quote the mean and the spread, call the spread a spread,
and treat small differences between models as unresolved.
:::

# Stratify when a class is rare

| Fold | Plain KFold | StratifiedKFold |
|---|---|---|
| 1 | 7 failures of 160 | 5 of 160 |
| 2 | **1 failure of 160** | 6 of 160 |
| 3 | 7 of 160 | 6 of 160 |
| 4 | 5 of 160 | 6 of 160 |
| 5 | 9 of 160 | 6 of 160 |

::: notes
Point at fold 2 of the left column. An AUC computed from a single positive
example is not a measurement of anything - and nothing in the code warns you.

StratifiedKFold keeps each fold's class balance equal to the whole dataset's. It
costs nothing and it is one word in the call.

The rule: stratify whenever a class is rare. Which, as lesson 4 established, is
whenever the problem is interesting.
:::

# How much does it help?

| | Single split | 5-fold CV |
|---|---|---|
| worst | 0.9119 | 0.9439 |
| best | 1.0000 | 0.9588 |
| spread | 0.0881 | 0.0149 |

::: notes
Forty seeds, changing nothing but the seed.

The spread falls from 0.088 to 0.015 - about seven times more stable. And note
that both centre in the same place: cross-validation is not more pessimistic, it
is less arbitrary.

What it costs is k fits instead of one. For everything in this course that is
seconds. For a large network it is a real decision, and the usual compromise is
a single validation set with enough rows that its own error bar is tolerable.

What it does NOT fix is the whole last hour of today.
:::

# How many folds?

- **k = 5 or 10** in practice
- Larger k: less pessimistic, more expensive, folds more correlated
- k = m is **leave-one-out**

::: notes
The question always comes, so answer it before it does.

Larger k means each fold trains on more data, so the estimate is less
pessimistic - the bias shrinks. It also means more fits, and the training sets
overlap even more heavily, so the fold scores are more strongly correlated and
the spread you observe understates the uncertainty by more.

Five and ten are the conventional choices and the difference between them rarely
matters. Leave-one-out is the extreme: nearly unbiased, m fits, and notoriously
high variance for the estimate itself.

If you have the compute, repeated k-fold - the whole procedure run with several
different shuffles, averaged - buys more than moving from 5 to 10.
:::

# When the rows are not independent

- Ten readings from the **same drive**, split at random
- Nine in training, one in test
- The model recognises the drive, not the failure

::: notes
Everything so far assumed rows are independent draws. That assumption is
violated constantly and silently, and random k-fold gives no warning.

The concrete case, which they will meet: your table has ten telemetry readings
per drive. A random split puts nine of a drive's readings in training and one in
test. The model does not need to learn anything about failure - it learns to
recognise that drive, and the test row is nearly a duplicate of a training row.

The score is excellent and the model is worthless on a drive it has never seen,
which is the only situation anyone cares about.

Other instances worth naming out loud: several images of the same patient,
several sentences from the same document, repeated measurements of the same
machine, any panel data.
:::

# Split by the thing you must generalise to

- Same entity in two folds → **GroupKFold**
- Predicting the future → **TimeSeriesSplit**

::: notes
The rule that covers both: split along the axis you need to generalise across.
If the model must work on a drive it has never seen, no drive may appear in both
training and test. GroupKFold takes a group label and guarantees that.

Time is the other case, and it is stricter. With a random split the model trains
on Wednesday and tests on Tuesday, which is not the problem anyone has. Real
deployment always predicts forward, so validation must too: TimeSeriesSplit
trains on a prefix and tests on what comes next.

The tell that you needed one of these and did not use it is the same in both
cases - a validation score that is excellent and a production score that is not.
That gap is the most common cause of a model failing after deployment.
:::

# Notebook 1, live

- The split lottery, the three identical models, k-fold by hand

::: notes
Run notebooks/01. Twenty minutes.

Have them run the five-seed cell first and watch seed 3 return 1.000 before you
say anything about it.

The cell worth protecting if time runs short is the 200-split model selection  - 
it is the one that changes behaviour rather than just adding knowledge.
:::

# Break

- Ten minutes

::: notes
Ten minutes. The second half is bias-variance, which needs them awake.
:::

# Where does the error come from?

- **Bias**: the average model is wrong
- **Variance**: the individual models disagree
- **Noise**: the target itself is uncertain

::: notes
Set up the decomposition in words before any algebra.

Imagine drawing many training sets from the same source and fitting the same
model to each. You get many models, and they fail in two distinguishable ways.

They may agree with each other and all be wrong - a straight line fitted to a
curve gives nearly the same line every time and every one misses the curvature.
That is bias: consistently, confidently wrong.

Or they may disagree wildly - a twelfth-degree polynomial on twenty-five points
gives a different wild curve every time. That is variance: unreliable, whatever
the average looks like.

Underneath both sits noise in the measurements, which is not a property of the
model at all.
:::

# Three hundred parallel universes

![](many_universes.png)

::: notes
Each faint line is a model fitted to a DIFFERENT training sample from the same
curve. In real life you drew exactly one of them and never knew about the rest.

Left, degree 1: the lines lie almost on top of each other - stable - and the red
average is nowhere near the black truth. That gap is bias.

Middle, degree 3: average tracks the truth, spread is modest. Both small.

Right, degree 12: through the middle it follows the truth better than degree 1
ever could, and at the edges the lines fly off and drag the average off the top
of the plot with them. That scatter is variance.

Two things to point at in the right panel. The scatter is worst WHERE THE DATA
RUNS OUT - beyond about 30 degrees nothing constrains the fit. Extrapolation is
where flexible models embarrass themselves.

And any one of those faint lines is a model somebody actually deployed.
:::

# The decomposition

$$\mathbb{E}\left[(y - \hat{f})^2\right] = \text{bias}^2 + \text{variance} + \sigma^2$$

::: notes
State it, do not derive it - handout section 5.2 does that in five lines, and it
is worth reading because the derivation is short and the cross terms vanishing
is the whole trick.

The important claim is that this is an IDENTITY, not an approximation or a
useful metaphor. Notebook 2 checks it on 300 independently drawn training sets
and the two sides agree to three times ten to the minus twelve - floating-point
equality.

That matters pedagogically: they have seen this picture in every textbook drawn
as a schematic. Here it is measured.
:::

# Measured, on 25 observations

| Degree | bias² | Variance | Noise | Total |
|---|---|---|---|---|
| 1 | 5,250.8 | 723.8 | 484.0 | 6,458.6 |
| 2 | 59.6 | 70.8 | 484.0 | **614.3** |
| 5 | 8.9 | 368.9 | 484.0 | 861.8 |
| 12 | 153,537.3 | 32,174,344.9 | 484.0 | 32,328,366.2 |

::: notes
Noise is 484 in every row - point at the column and say that it never
moves, because it is a property of the data and not of the model.

Degree 1 is a bias problem: the largest bias² and the smallest variance. Stable
and wrong.

Degree 12 is a variance problem, and the magnitude deserves a pause: 32 million,
four orders of magnitude larger than anything else. On 25 points a degree-12
polynomial nearly interpolates, and an interpolating model is a machine for
amplifying noise.

Someone will notice that degree 12 also has the worst bias², which contradicts
"flexible models have low bias". It does not: with variance this large the
average of the fitted curves is not a meaningful curve, so the bias term
inherits the instability. Read the two together.
:::

# The picture everyone has seen, now measured

![](bias_variance_tradeoff.png)

::: notes
Bias falls monotonically with complexity; variance rises; the total is a U whose
minimum is the model you want.

The y-axis is logarithmic, and say why: otherwise degree 12 flattens everything
else onto the baseline. That in itself tells them something about the scale of
the failure.

The dotted line is the noise floor at 484.
:::

# The floor you cannot go below

- Noise variance is **484** at every complexity
- A reported error below the floor means **contamination**, not excellence

::: notes
The noise column never moves, because it is a property of the data rather than
of the model. No model, no algorithm and no quantity of data brings the expected
error below it.

The corollary is a practical diagnostic worth writing down: if you know your
measurement precision and your model beats it, something has leaked. This is a
real check used in engineering and in the physical sciences, and it catches
errors nothing else catches.

Connects forward to the last hour, where we manufacture exactly that symptom
deliberately.
:::

# The best model depends on how much data you have

| Degree | m = 25 | m = 60 | m = 150 |
|---|---|---|---|
| 2 | **614.3** | 570.6 | 555.3 |
| 5 | 861.8 | **540.9** | **506.9** |
| 12 | 32,328,366.2 | 7,626.9 | 899.0 |

::: notes
The same experiment at three sample sizes. The best degree moves from 2 to 5 as
the data grows, because there is now enough of it to pin down the extra
coefficients and the variance has fallen accordingly.

Degree 12 improves by a factor of four thousand between 25 and 150 observations
and is still the worst model in its column.

Then the sentence worth putting on the board: "which model is best" is not a
question about models. It is a question about models AND the quantity of data
you have.

This explains something they will meet constantly - a paper reports a large
model beating a small one on a dataset far bigger than theirs, and it does not
reproduce for them. Neither experiment was wrong.
:::

# Learning curves: the same diagnosis, without knowing the truth

![](learning_curves.png)

::: notes
Everything so far needed the true function. Learning curves need only data you
have, which is why they are the practical tool.

Train on 15% of the data, then 25%, and so on, plotting training score and
cross-validated score against the number of examples used.

Left, one feature only: the curves MEET, and they meet LOW - 0.715 against 0.718
 -  and the validation curve is flat from the first point to the last.

Right, six features plus 150 columns of pure noise: training exactly 1.000
against validation 0.847, a gap of 0.15, and the validation curve still climbing
at the right edge.

Say that the right-hand setup is not contrived. It is what a wide feature table
looks like when most columns carry nothing, which is most feature tables.
:::

# Which fix to buy

- Curves **meet low** → bias. More data will **not** help
- Wide gap, validation **still rising** → variance. More data **will**

::: notes
This slide is the practical payoff of the entire second hour, so do not rush it.

High bias: the model is equally mediocre on data it has seen and data it has
not, because it extracted everything available and that was not enough. What
helps is a more flexible model, better features, less regularisation. What does
NOT help is more data - the curve is already flat, and more rows land on the
same plateau.

Emphasise that this is the expensive mistake. Collecting data is the slowest and
costliest item on the list, and the plot says in advance that it will buy
nothing.

High variance: the rising curve is itself the evidence that more data will help.

Name the predictable mistake and defend the instinct: prescribing data for a
bias problem is what almost everyone does, and the reasoning is sound - more
data almost always helps and never makes a model worse. It just does not help
THIS failure, and fifteen lines of code tell you which failure you have.
:::

# Notebook 2, live

- Measuring bias and variance; the learning curves

::: notes
Run notebooks/02. Twenty minutes.

The cell to protect is the decomposition table where the sum equals the measured
error to twelve decimal places. Let them see the identity hold rather than
telling them it does.

If time is short, skip the three-sample-size comparison and assign it as reading
 -  the table is in the handout.
:::

# Now: what cross-validation does not protect you from

- 2,000 columns of **pure random noise**
- No relationship to the label whatsoever
- The honest score of any model on it is **0.500**

::: notes
Set this up carefully, because the demonstration only lands if they believe the
data is genuinely empty.

Two thousand columns drawn from a standard normal, independent of the label and
of each other. There is nothing to find. Any model's honest AUC is one half.

Tell them what is coming so they can watch for it: we are going to write code
that looks completely reasonable, and get 0.93.
:::

# The code almost everyone writes

```python
selector = SelectKBest(f_classif, k=10).fit(noise, y)
chosen = noise.loc[:, selector.get_support()]
cross_val_score(model, chosen, y, cv=folds)
```

::: notes
Read it aloud as the sentence a practitioner would say: "two thousand features
is too many, so let me keep the ten most predictive ones, then cross-validate a
model on those."

Every line is defensible in isolation. Feature selection is normal. Ten from two
thousand is reasonable. Cross-validating afterwards is exactly what we spent the
morning recommending.

Do not reveal the bug yet. Ask them to find it before the next slide.
:::

# 0.931

- Cross-validated AUC **0.931** on data with no signal
- Four of five folds between 0.93 and 0.98

::: notes
The result. Tight fold agreement, which is exactly what a trustworthy result
looks like.

Now the bug, which is on the first line: SelectKBest was shown EVERY row,
including the rows that would later serve as test folds. It searched two
thousand columns for the ones that best matched labels it had already seen, and
handed the winners to cross-validation.

The framing worth using: cross-validation did not fail. It was lied to. It
faithfully measured a procedure that had already peeked.

This is the single most common serious error in applied machine learning, and it
is committed in published work regularly.
:::

# Fix it, and look what happens

- Selection moved **inside** the pipeline, refitted per fold
- 20 different fold seeds: **0.658**
- The truth is still 0.500

::: notes
The fix is one Pipeline: selection becomes part of the model, so it is refitted
on the training rows of each fold.

The leak is mostly gone - 0.93 down to 0.66. And it has NOT come back to 0.5.
Twenty different fold seeds all agree that a signal-free dataset supports a
model better than chance.

Let that be uncomfortable for a moment before explaining. Most treatments of
this topic stop at "put it in a Pipeline" and imply the problem is solved.
:::

# Why it does not return to 0.5

- With 2,000 columns and 800 rows, some columns match the label **by accident**
- That accident belongs to the **sample**, not to the split
- Every fold contains it

::: notes
This is the most useful idea in the lesson, so take the time.

With two thousand columns and eight hundred rows, some columns correlate with
the label purely by chance across the whole dataset. That accident is a property
of this particular sample. It is present in every subset of it - every training
fold and every test fold alike.

So a selector fitted honestly on four fifths finds those columns, and they still
work on the remaining fifth, because the spurious correlation was never
fold-specific in the first place.

The statement to write on the board: cross-validation protects you from leaking
between folds. It cannot protect you from having searched a large space of
possibilities on a small sample. No rearrangement of the same 800 rows removes a
pattern that is genuinely in those 800 rows and absent from the world.

What helps: a test set held out before any of this and used once; fewer
candidates; or more rows. Nothing else.
:::

# Hyperparameter search is the same problem

| | AUC |
|---|---|
| best of 25 candidates | **0.7999** |
| average candidate | 0.7265 |
| nested cross-validation | 0.6699 |
| the truth | 0.500 |

::: notes
Choosing between configurations is choosing, and choosing on data costs the same
honesty. Twenty-five combinations on the same signal-free table.

The number a practitioner reports is best_score_, and it is the MAXIMUM of 25
noisy estimates. The maximum of a set of noisy numbers is biased upward even
when every one measures the same quantity - and here they nearly do, since the
average candidate is 0.73 and the worst is 0.68.

So 0.7999 is not the performance of the chosen configuration. It is the
performance of whichever configuration got luckiest.

The optimism is +0.13, which is larger than most differences anyone reports
between competing methods.
:::

# Nested cross-validation

![](nested_cv_diagram.png)

::: notes
Measure the search, not the winner.

Outer loop holds out a fifth. The entire search runs inside the remaining four
fifths. Score the search's chosen model on the held-out fifth. Repeat five
times.

The inner loop may overfit its own data as much as it likes; the outer block was
never part of it.

One clarification that always comes up, so pre-empt it: nested cross-validation
is NOT a way to choose hyperparameters - it produces k possibly different
winners. It is a way to estimate what choosing costs. You choose on all the data
afterwards, and you report the nested figure.
:::

# Four ways to measure the same worthless model

![](leakage_ladder.png)

::: notes
The summary of the whole last hour, on one axis. Same data, containing nothing.

0.500 is the truth. 0.670 is nested cross-validation. 0.800 is what the search
reports. 0.931 is selection before cross-validation.

Say clearly: nobody in this picture wrote dishonest code. The difference between
0.93 and 0.50 is entirely a matter of which choices were made while looking at
which data.

Each step to the left is a piece of discipline, and each one costs a chunk of
the score you would otherwise have reported. That is what discipline feels like
from the inside, and it is why it is unpopular.
:::

# The debt, paid

| Policy | Cost on the test set |
|---|---|
| threshold 0.50, the default | 10,540 EUR |
| threshold chosen on **validation** | **4,700 EUR** |
| threshold chosen on **test** | 3,300 EUR |

::: notes
Lesson 4's threshold, done properly. Split three ways: train to fit, validation
to choose, test to report once.

The honest threshold is WORSE on the test set than the cheating one. It is
supposed to be - and this is the slide where that finally makes sense to them.

The cheating figure describes a threshold tuned to this particular test set, and
there is no reason it survives contact with the next two hundred drives. The
honest figure is an unbiased estimate of what happens next.

Note the cost: we now train on 56% of the data instead of 75%, because the
validation set has to come from somewhere. On a small dataset that hurts - which
is the argument for choosing by cross-validation on the training portion and
keeping the test set whole.
:::

# Three sets, three jobs

- **Train**: fit the parameters
- **Validation**: make every choice: model, hyperparameters, threshold
- **Test**: report. Touched **once**

::: notes
The organising principle of the whole lesson, on one slide. Put it on the board
and leave it there.

Every choice belongs to the validation set. Which model, which penalty, which
features, which threshold, when to stop training, whether to try a different
approach entirely - all of it.

The test set has exactly one job and one use. The moment you look at it and
change something in response, it has become a validation set, and you no longer
have a test set at all.

The honest thing to say aloud, because they will do it anyway: everyone looks
twice sometimes. The requirement is that you SAY SO in the report. A stated
weakness costs a little credibility; an unstated one costs all of it when
somebody else fails to reproduce your number.
:::

# A fixed seed buys repeatability, not stability

- 30 seeds, all perfectly reproducible: **0.913 to 1.000**
- `AUC 1.000 (random_state=3)` is reproducible **and** misleading

::: notes
Every one of those thirty runs is exactly reproducible by anyone with the same
code. They disagree with each other by nearly nine points.

So reproducibility is necessary and nowhere near sufficient. Report the seed AND
the spread.

Contrast the two ways of writing the same experiment up. "AUC 1.000,
random_state=3" - fully reproducible, thoroughly misleading. "AUC 0.951 plus or
minus 0.019 over 5-fold cross-validation, seed 0" - reproducible and honest.

The second one is also the format the final project will be marked in.
:::

# What to fix, and what to write down

- Every `random_state`, and `np.random.default_rng(seed)`
- **Library versions**: defaults change between releases
- What you could **not** fix: threads, GPU non-determinism

::: notes
Practical checklist. The container exists for the second item: it pins the
environment so that their result and the marker's result are the same result.

Worth saying that a result nobody can reproduce in two years was not really
reproducible - and that scikit-learn genuinely does change defaults between
minor versions, which has invalidated published comparisons.

Commit the code that produced the number, not a tidied-up version of it.
:::

# Notebook 3, live

- The 0.931, the fix, the nested search

::: notes
Run notebooks/03. Twenty minutes if the clock allows.

The cell to protect is the first one - let them watch 0.931 appear on empty data
before any explanation. Nothing else in the lesson changes behaviour as
reliably.
:::

# What to put in a report

- The metric, the **spread**, and how it was estimated
- How many **positives** the test set held
- Every seed, and every library version
- Which choices were made on which data

::: notes
A checklist they can apply to the final project and to anything they read.

The four items map to the four failures of today: a number without a spread
hides the lottery; a metric without a positive count hides why the spread is
wide; a missing seed makes it unreproducible; and an unstated selection
procedure hides the optimism.

Turn it around as a reading habit too. When they meet a paper reporting a single
figure with no spread, no fold count and no statement about how the
hyperparameters were chosen, they now know exactly which questions are
unanswered - and roughly how large the correction is likely to be.
:::

# What to take away

- **A test score is a measurement.** 0.885 to a perfect **1.000**, same model
- **Choosing on one split chooses noise**
- **bias² + variance + noise** is an identity, and the noise floor never moves
- **Curves that meet low mean bias**: more data will not help
- **Cross-validation protects against fold leakage, nothing more**

::: notes
Close on the number they opened with. Seed 3, AUC 1.000, on a model whose honest
performance is 0.951.

If they remember one habit from today, make it this: before believing any
reported score, ask what the base rate is, how many positives are in the test
set, and which choices were made while looking at which data.

That third question is the one that separates a result from a story, and it is
the question the final project will be marked on.
:::

# Homework

- **Exercise 5**, due **Friday 30 October 2026, 09:00**
- A result that is too good. Find out why, then produce the honest number

::: notes
Set it explicitly and say the deadline out loud: next Friday, 30 October,
09:00, before lesson 6 starts.

The shape of it: they are given a notebook that reports an excellent score, and
the score is wrong. They have to find the reason, fix it, report the honest
number, and say how much honesty cost.

Warn them that there is more than one problem planted, and that the marks are
for the diagnosis rather than for the final figure - which will be much worse
than the one they start with, and is supposed to be.

Next week: k-nearest neighbours, Naive Bayes and support vector machines. Three
new model families, all of them evaluated with what we built today.
:::

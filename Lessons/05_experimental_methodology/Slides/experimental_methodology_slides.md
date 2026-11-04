---
title: "Lesson 5: Experimental Methodology"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "6 November 2026"
---

# Before we start: lesson 4, concluded

- First, lesson 4's last slides: receiver operating characteristic (ROC) curves, the area under the curve (AUC), the threshold from costs, notebook 3
- Exercise 4 is set at their end; exercises 4 and 5 are both discussed on 13 November
- Then lesson 5, with a few parts left for self-study, marked (optional)

::: notes
Lesson 4 ran over two Fridays: on 30 October it reached its slide 42, so this
Friday opens with its remaining slides, about forty-five minutes, from its slide
43 to its homework slide, where exercise 4 is set officially. Its optional slides
(55, 58 to 60) can be skipped.

Lesson 5 then has about two hours and a quarter instead of three. Nothing is cut
from the material: the slides marked (optional) - 19 and 20, 31 and 32, 34 and 35, 54
and 55 - and notebook 2 are left for self-study, and the handout covers them all.
Say so at the start, so nobody thinks they missed something.

Exercises 4 and 5 are both discussed at the start of lesson 6, Friday 13
November. Students have already loaded exercise 4's data and fitted a logistic
regression, so the two together are a fair week's work.
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

Section 8 of the handout pays it off, and slide 52 shows the number.
:::

# Today: how to tell whether it works

- How far apart can two honest splits be?
- Cross-validation: what it estimates, and what it does not
- Bias, variance, and the error no model can remove
- Learning curves: more data, or a different model?
- Leakage that survives cross-validation
- Lesson 4's threshold, done properly

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

The last bullet covers the end of the lesson: lesson 4's threshold chosen on a
validation set, the three sets and their three jobs, and what to put in a report.

One word to flag now, because it changes meaning halfway through. "Bias" is used
twice today. On slide 5 and in the leakage part it is the bias of an ESTIMATE: a
measured score that is systematically too high or too low. On slides 25 to 40 it
is the bias of a MODEL: a model too simple to follow the truth, wrong in the same
way on every training set. A predictable slip is to merge them - "cross-validation
removes bias" - when it removes neither: it makes the estimate more precise, which
is a third thing.
:::

# A test score is a measurement

- You would not report a poll of forty people to three decimals
- Holding data out makes the estimate **unbiased**: right on average over all the test sets you might have drawn, for the model you fitted
- It does nothing to make it **precise**: close to the truth on the one split you did draw

::: notes
This distinction is the whole first half of the lesson, and it is routinely
missed - because "test set accuracy" sounds like a property of the model rather
than a measurement with a standard error.

Lesson 1's rule was about bias: never let the test set influence the model, and
the number you get is centred on the truth. Which truth, precisely: the test score
is an empirical risk, computed on the test rows, and what it estimates without
bias is the EXPECTED risk R of lesson 1, the error over the whole distribution of
drives the model will ever meet. Averaged over all the test sets you might have
drawn, the score equals R. That holds because the test rows played no part in
choosing the model. It says nothing at all about how far from R any particular
measurement lands.

The poll in numbers: forty people, a proportion near one half, standard error
square root of 0.25 ÷ 40 = 0.079. The second decimal is already uncertain, so the
third is decoration.

A predictable slip: "the test score is an unbiased estimator of the empirical
risk". The reasoning is natural - both are averages over rows - but the test score
IS an empirical risk; what it estimates is the expected one.

Ask the room, before the next slide: how far apart do you think two honest
splits of the same dataset can be? Take a guess out loud. They will underestimate
it.
:::

# The fleet, cut down

- Lesson 4's drives, cut to **800**, of which **29 fail** (3.6%)
- Why 800: real projects often have a few hundred rows, a few dozen positive
- In code: `train_test_split(X, y, test_size=0.25, stratify=y)`; lesson 4's pipeline fitted on the training part; `roc_auc_score` on the test part

::: notes
The model is lesson 4's, unchanged, make_pipeline(StandardScaler(),
LogisticRegression()): scaled features, logistic regression, scored
by AUC (area under the receiver operating characteristic curve) on a stratified
75/25 split - the In code line names each piece. The failure rate is about
lesson 4's: 29 of 800 is 3.6%, against 306 of 8,000, 3.8%.

Justify the reduction explicitly, because a sharp student will otherwise suspect
the demonstration is rigged.

With 8,000 drives a single split is fairly stable, and lesson 4's numbers were
trustworthy: a 25% test set held 306 × 0.25 = 76.5, so 76 failures in lesson 4's split. At 800 drives it
holds 7. Eight hundred records with 29 positives is a far more common
situation than eight thousand - it is what a real pilot study looks like - and
what follows is simply what the honest version of that situation looks like.

Same data, same model, same code as last week. Only the sample size changed.
:::

# 200 honest splits, one dataset

![](split_lottery.png)

::: notes
Two hundred legitimate experiments. Every one done correctly: nothing fitted
before the split, the test set never inspected, the metric computed properly. On
each split the model is refitted on that split's training part, so these are 200
slightly different models from one procedure, and the mean is the procedure's.

How to read it: each bar is a bin about 0.004 of AUC wide ((1.000 - 0.885) ÷ 28
bins), not one exact value; its height is how many of the 200 splits landed in it.

Worst 0.885. Best 1.000. Mean 0.955.

Sit on the 1.000 for a moment. That split reports a PERFECT classifier - all 7
failing drives ranked above all 193 healthy ones - for a model whose mean over the
200 splits is 0.955. Ten of the two hundred scored above 0.99.

Then the point that matters most, and say it slowly: the right-hand tail is the
dangerous end, not the left. A disappointing score makes you frown and keep
working. A delightful score makes you stop, write it down, and put it in the
report. The splits that flatter you are the ones you are least likely to
question.
:::

# Why do the 200 scores spread so widely?

- Each test set takes 25% of the 29 failures: 0.25 × 29 = 7.25, so **7 failures** among 200 drives
- AUC = pairs ranked right ÷ 1,351, the 7 × 193 = 1,351 pairs of one failing and one healthy drive
- For example, one failing drive slipping below 20 more healthy ones costs 20 ÷ 1,351 = 0.015 of AUC

::: notes
This is the arithmetic behind the previous slide, and it generalises into the
single most useful diagnostic question in the course.

The precision of a classification metric is governed by the count of the RARER
class, not by the size of the dataset. Eight thousand drives at 3.8% gives 306
positives and a stable estimate. Eight hundred gives 29, and it does not.

Give them the question to carry away: how many positive examples are in the test
set? If the answer is under about fifty, treat every metric as provisional.

The third bullet makes the count concrete, with lesson 4's reading of AUC as the
fraction of failing-healthy pairs ranked the right way round: there are only 7 x
193 = 1,351 pairs, and they come in seven bundles of 193, one per failing drive.
One drive ranked a little lower moves twenty pairs at once and the AUC by 0.015 -
over half of the 0.024 standard deviation of slide 7. The twenty is an
illustration, about a tenth of the 193 healthy drives, not a measured value. In
numbers: a perfect ranking is 1,351 ÷ 1,351 = 1.000; twenty pairs turned wrong give
1,331 ÷ 1,351 = 0.985.

Two predictable slips. "Each test set holds 7% failures": it holds 7 failures,
7 ÷ 200 = 3.5%. And "the AUC counts how many times a failing drive outscores a
healthy one": it is that count divided by 1,351, a fraction between 0 and 1.

Worth connecting to lesson 4: this is the same reason the precision METRIC (the
share of flagged drives that really fail) collapsed there while AUC did not
notice. Say "metric" aloud: slide 5 used "precise" for the error bar of an
estimate, a different thing.
:::

# Parameters and hyperparameters

- **Parameters**: estimated by the fit, $w$ and $b$
- **Hyperparameters**: not estimated by the fit; some fixed before it (penalty $\lambda$, neighbours, tree depth), some after it (the threshold)
- Training error would always pick $\lambda = 0$, so they are chosen by **comparing** models on data the fit did not see
- scikit-learn's C plays the role of 1/$\lambda$: smaller C, stronger penalty

::: notes
One slide, because every choice in the rest of the lesson is a choice of one of
these, and the word has been used loosely since lesson 2.

Parameters are what the fit estimates: coefficients and intercept.
Hyperparameters are what it does not - lesson 3's penalty, lesson 6's neighbours,
lesson 7's tree depth, and lesson 4's threshold.

Neighbours and tree depth are named ahead of their lessons: say so, they only
need to know that such settings exist.

Then the second bullet's last clause: some hyperparameters are fixed before the
fit (lambda), some after it (the threshold). That is worth a sentence, because the obvious shorthand is wrong and a
sharp student will catch it. "Fixed before the fit" is true of the penalty,
which sits inside the cost being minimised, and false of the threshold, which is
chosen afterwards on probabilities the fit has already produced. What they share
is not timing: it is that neither is estimated the way a coefficient is.

Then the sentence that makes the distinction matter rather than taxonomic: you
cannot estimate a hyperparameter by minimising training error, because that
objective already knows its favourite answer - no penalty at all, the biggest
tree available, a threshold tuned to the rows already seen. So the only way to
choose one is to compare, and comparison needs data the model has not seen. That
is what the rest of today is about, and why it costs what it costs.

Last bullet is the trap in the next table: C is the reciprocal of the penalty,
so C = 0.01 is the most heavily regularised of the three, not the least.
(scikit-learn multiplies the data term by C instead of the penalty by lambda; up
to the factor m the two are the same thing, which is why "plays the role of" and
not "equals".)

Two predictable slips. "Hyperparameters are chosen before the fit": the
threshold is chosen after it. And "fit once on the training data, then tune every
hyperparameter on that model": true for the threshold, false for lambda, which
needs one fit per candidate. The next slide draws both recipes.
Handout section 3.
:::

# Choosing a hyperparameter on a validation set: two recipes

![](two_recipes.png)

::: notes
First the third set, because this is where it enters. Training error cannot
choose a hyperparameter - it always prefers the most flexible model - and the
test set must not, or it is spent (slide 3's broken promise). So a part of the
non-test data is held out for choosing: the VALIDATION set. Training fits,
validation chooses, test reports once. The sizes are slide 52's: of the 800
drives, 200 test, 150 validation, 450 to train.

Two recipes, and the difference is where the hyperparameter acts.

Left, a hyperparameter fixed before the fit. Lambda sits inside the cost being
minimised, so one fit cannot serve every value: each candidate needs its own fit.
Three values of C, three fits on the 450, three scores on the 150. Step 3 is the
one to say slowly: what wins is a VALUE of C, not a model - the three fitted
models are thrown away. The winning C is refitted once on all 600 non-test
drives, fit number 4, and the test set is scored once.

A predictable slip: "identify the winning model, then refit it". The winner is a
number; the refit is a new model, fitted on more data than any candidate saw.

Right, a hyperparameter chosen after the fit. The threshold acts on probabilities
the model has already produced, so one fit serves every candidate: fit once on
the 450, predict on the 150 validation drives, try every threshold on those
probabilities, keep the cheapest. Slide 52 does it for lesson 4's costs.

Both end the same way: the test set is touched once, at the end, after every
choice has been made. Slide 13 replaces the single validation set with
cross-validation; the recipe stays the same. Handout section 3.
:::

# On each of slide 7's 200 splits, three models compete

| Model (C = 1, the default, is slide 7's) | Mean test AUC over the 200 splits | Splits it won outright (31 were ties) |
|---|---|---|
| C = 0.01 | 0.9549 | 89 |
| C = 1 | 0.9549 | 42 |
| C = 100 | 0.9544 | 38 |

::: notes
Remind them of the reciprocal as they read the first column: the top row,
C = 0.01, is the most heavily penalised model, not the least.

Reporting a noisy number is bad. Choosing with one is worse, because noise does
not average out when it is used to decide - it decides.

Say what was done first: the same 200 splits as slide 7, the three models
fitted on each training part and scored on its test part, and on each split the
one with the highest AUC declared the winner. 89 + 42 + 38 = 169 outright wins,
plus 31 ties, makes 200. The ties are real: AUC moves in steps of 1 ÷ 1,351, and
C = 1 and C = 100 - both barely penalised - often rank the 200 test drives in
exactly the same order: 29 of the 31 ties are between those two.

Read the first column, then the last. The three models are equally good: their
mean AUCs differ by at most 0.0005, about fifty times less than the standard
deviation of a single measurement, 0.024.

And yet a winner is declared on almost every split. If someone asks why
C = 0.01 wins most often: it is not chance, it holds on other seeds too (seeds 200
to 399 give 86, 41 and 41 outright wins). C = 0.01 ranks the drives a little differently from
the two near-twins, so it wins or loses alone, while C = 1 and C = 100 split their
wins between them. The counts say the models are DIFFERENT; they say nothing
about which is BETTER - for that, the means, and the means are equal.

Then the sentence to land: run this once, which is what real life allows, and
you will report that one of these is best with a number to support it. That
conclusion is noise wearing the clothes of a result.
:::

# The same three models: who wins, and how they score

![](single_split_selection.png)

::: notes
Left: what each winner wins by - on each of the 200 splits, the winner's AUC
minus the runner-up's, 0 on a tie. Work one split aloud before the histogram.
Seed 1, the first split without a tie: C = 0.01 scores 0.9756, C = 100 0.9667,
C = 1 0.9660. The winner is C = 0.01, the runner-up C = 100, and the margin is
0.9756 − 0.9667 = 0.0089; the third model does not count. The histogram is that
subtraction, done on all 200 splits. The median margin is 0.0015; the dashed line is the
0.024 that a single measurement wobbles by (slide 7), sixteen times wider. The
decision is made inside the noise. Right: the 200 test AUCs of each model, as a box plot - say how to
read one, it is the first in the course: the box holds the middle half of the 200
scores, the line inside it is the median, the whiskers reach the rest, and the
circles are the few scores unusually far out. The three boxes sit at the same
height.

The right panel is the truth about these three models. The left panel is how
little separates the winner a single experiment reports.
:::

# k-fold cross-validation: the validation set, rotated

![](kfold_diagram.png)

::: notes
The idea in a sentence: if one measurement is noisy, take several and average
them. The difficulty is that you cannot afford several test sets - every row
spent on testing is a row not spent on training.

Introduce it in five steps. The problem: slide 7, one measurement is noisy, so
take several and average. The obstacle: several test sets cost rows. The trick:
rotate. The legitimacy: every drive is predicted exactly once, by a model that
never saw it. The pay-off: slide 19, where 40 seeds of a single split range from
0.912 to 1.000 and cross-validation from 0.944 to 0.959.

Cross-validation resolves it by rotating which part is held out. Cut the data
into k parts, the folds, train on k-1, score on the one left out - the validation
fold - and repeat until every part has been the validation fold exactly once.
Here: 800 drives in 5 folds of 160; each round trains on 640 and scores the 160
held out. Read the figure row by row: each row is one round, each column one fold,
and each round ends with one score; the estimate is the mean of the five.

Keep the two words apart: a FOLD is a part of the data, a ROUND is one fit and
one score. One model per round. With k-fold there are as many rounds as folds,
which is why people say "one model per fold" - a predictable slip that later
makes "every fold trains on 80%" sound like a fold is a model. scikit-learn calls
a round a split: n_splits.

Where it runs: on all the data that is not the test set. It REPLACES the fixed
validation set - a predictable slip is "k-fold runs on the validation set". Two
uses. To MEASURE a procedure, with nothing chosen (slides 14 to 22): run it on all
800 drives, no test set needed. To CHOOSE: slide 10's left recipe with the
validation set replaced by the folds - set the test set aside first, run 5-fold
cross-validation on the rest for each candidate (three values of C: 3 × 5 = 15
fits), keep the value with the best mean, refit it on all the non-test data
(fit number 16), score the test set once. scikit-learn's GridSearchCV runs this
loop (slide 47); with refit=True, its default, best_estimator_ is fit 16.

Say why it is labelled validation and not test, because the code will contradict
the slide. Lesson 1's three roles: training fits, validation chooses, test
reports once. This block is scored k times and, by the end of the lesson, is what
a search chooses on. scikit-learn calls it test - test_index, mean_test_score -
and there the word means only "not used to fit this round's model", never the
test set reported once. Handout section 4.1.

Then the clause that makes it legitimate rather than a trick, and say it
explicitly because students suspect otherwise: every row is predicted exactly
once, by a model that never saw it. This is not testing on training data with
extra steps.
:::

# What it actually estimates

- **0.9514** (seed 0, mean of five round AUCs) scores a **procedure**, not one model
- Slightly **pessimistic**: each model trains on 80% of the data
- **Nothing chosen**: this is the measurement. **Something chosen**: test set aside first (slide 10)
- `cross_val_score` gives scores, no model; the model you ship is `fit` on all 800

::: notes
The number on the slide: the five round AUCs at seed 0 are 0.929, 0.958, 0.984,
0.939 and 0.947; their mean is 4.757 ÷ 5 = 0.9514, their standard deviation
0.0187. The procedure is the recipe as a whole - scaling, model, its settings -
refitted from scratch in every round.

Two consequences, both of which surprise people.

First: there are k different fitted models in the calculation, and the reported
number belongs to none of them. It answers "if I apply this procedure to a
dataset of about this size, what should I expect?" - which is the question you
actually have. The next slide shows the five models.

The fourth bullet is the line students most need: two uses of the same tool.
When nothing is being chosen, this is the measurement: the model you ship is the
same recipe refitted on all 800 drives, and 0.9514 is its estimate. When
something is chosen - a C, a set of features, a threshold - the test set is set
aside first and the choice is made inside the rest (slide 10's recipe).

The last bullet is the same thing in code, in two phases. cross_val_score clones
the recipe, fits and scores it five times, and throws the five models away: it
returns scores, not a model. The model you ship is built afterwards, by an
ordinary fit on all the data. An analogy that works: five test cakes, each with
four fifths of the ingredients, to judge the recipe; the cake you bring to the
wedding is a sixth, made with all of them.

A predictable slip: "if the cross-validated score is not acceptable, change the
recipe and run the cross-validation again". Deciding on the score is itself a
choice: try several recipes on the same folds, keep the best, and its score is
inflated (slide 48 shows how much). If more than one recipe will be tried, set
the test set aside before the first one.

Second: every round's model trains on four fifths of the data, so it estimates the
performance of a model trained on less data than the one you will eventually
ship. The bias is in the safe direction, and it shrinks as k grows. That is the
argument for leave-one-out, where k equals the number of rows.

Handout section 4.2 has both in symbols.
:::

# Five rounds, five different models; you ship a sixth, fitted on all 800

![](five_round_models.png)

::: notes
The evidence for the previous slide's first bullet, from notebook 1. Seed 0,
5-fold, C = 1: each blue dot is one round's model, the red diamond the model you
would ship, the same recipe refitted on all 800 drives. Coefficients on
standardised features; each round fits its own scaler too, which moves them only
slightly.

Point at power_on_hours: from 0.010 in round 3 to 0.353 in round 2. Then
seek_error_rate: it changes sign, from -0.178 to +0.057. The five models even
disagree on individual drives: at threshold 0.5 they flag 15, 17, 13, 14 and 13
drives, and disagree on 8 of the 800.

So 0.9514 is the average of five scores of five different models, and the model
you ship is a sixth that no round ever scored. The number is a property of the
recipe - scale, then logistic regression with C = 1, trained on 640 drives - not
of any model. Slide 19 adds a second piece of evidence: 40 seeds give 40 sets of
five models, and the mean barely moves.

The six rows are the six features of the drive; the intercept is not drawn.

A predictable slip: "the model you ship is one of the five, or their average".
The diamond sits among the dots, which invites it; but it is a new fit of the
same recipe on all 800 drives, and the five round models only measured the
recipe.
:::

# Report the spread, but not as a confidence interval

- On **another fleet of 800**, how far would 0.9514 move? A 95% **confidence interval** answers
- Tempting: ± 1.96 × s ÷ √5 = **± 0.016** (s = 0.0187, sd of the 5 round scores)
- **Too narrow**: the 5 scores share training data, and s rests on 5 numbers
- So: report mean and s, call s a **spread**, treat small differences as unresolved

::: notes
Pose the question first, it is the first bullet: we have 0.9514 from one fleet;
on another fleet of 800 drives, how far would it move? That is what a confidence
interval answers. Recall it from lesson 1 and slide 5's poll: the standard error
is how far the estimate would move if the whole experiment were repeated on a new
dataset, and estimate plus or minus 1.96 standard errors is an approximate 95%
interval - approximate because it leans on the normal curve and on a standard
error that is itself right.

The temptation is to quote s over root k - the standard deviation of the round
scores divided by the square root of the number of rounds - and call it a
confidence interval. Here: 0.0187 ÷ √5 = 0.0084, so 0.9514 ± 1.96 × 0.0084 =
± 0.016. Do not, and give them the specific reasons rather than a warning.

That formula assumes k independent scores. Two things break it. The training
sets overlap - with 5 folds each is 4 folds, and any two share 3 of them, so
three quarters of their rows - and correlated scores carry less information than
independent ones. And s itself comes from only five numbers, so it is a rough
estimate, too small more often than not. On this fleet the second reason does
most of the work: measured over 300 new fleets, the correlation between round
scores is only 0.023, because each score is dominated by its own 160 test drives,
disjoint from the others. On other data the first can dominate, and Bengio and
Grandvalet showed in 2004 that no unbiased estimator of that variance exists in
general.

A predictable slip: "the models train on correlated data, so the scores are
correlated". The training sets overlap, but each score is measured on its own
disjoint fold. The next slide measures the damage either way.

The last bullet is the rule; the next slide is the evidence for it.
:::

# 300 new fleets: twice the naive spread; compare the widths

![](naive_interval_too_narrow.png)

::: notes
Notebook 1 does here what real data never allows: it draws 300 new fleets of 800
drives from the same generator and runs the same 5-fold cross-validation on each.
The histogram is the 300 cross-validated AUCs.

The black bar is the real spread: their standard deviation is 0.0165, so 95% of
them lie within ± 1.96 × 0.0165 = ± 0.032. The red bar is what s ÷ √k promised on
our fleet: ± 0.016, half as wide. Over the 300 fleets the naive 95% interval
contains the average in only 80% of cases; even with the small-sample t value
for 4 degrees of freedom, 2.776 instead of 1.96, only 88%.

The two bars have different centres - 0.9514 is our fleet's cross-validation,
0.961 the mean over the 300 new fleets - and that is not the point: compare
their widths.

Why it matters: an interval too narrow makes noise look like a difference. Two
models at 0.951 and 0.935 look separated at ± 0.016, not at ± 0.032.

This is also what slide 19's 0.0032 is NOT: that is how much cross-validation
moves when only the fold cut changes on the same 800 drives. 0.0165 is how much
it moves on new data.
:::

# Stratify when a class is rare

| Fold (seed 3) | KFold: folds cut at random | StratifiedKFold: each fold keeps 3.6% failures |
|---|---|---|
| 1 | 7 failures of 160, AUC 0.979 | 5 of 160, AUC 0.983 |
| 2 | **1 failure of 160, AUC 1.000** | 6 of 160, AUC 0.969 |
| 3 | 7 of 160, AUC 0.942 | 6 of 160, AUC 0.902 |
| 4 | 5 of 160, AUC 0.965 | 6 of 160, AUC 0.945 |
| 5 | 9 of 160, AUC 0.943 | 6 of 160, AUC 0.969 |

::: notes
Point at fold 2 of the left column: one failing drive, and a perfect 1.000. That
single drive happened to rank above all 159 healthy ones - 159 pairs, where a
stratified fold has about 6 × 154 = 924. An AUC computed from a single positive
example is defined, but it rests on one drive: all 159 pairs share it - and
nothing in the code warns you.

The counts: the 29 failures are among all 800 drives, so a fold of 160 should
hold 29 ÷ 5 = 5.8 of them, which as whole drives is 5 or 6. Both columns hold the
same 29 in total (7 + 1 + 7 + 5 + 9 = 5 + 6 + 6 + 6 + 6 = 29); only the dealing
differs. A predictable slip is "each fold of 160 holds 29 failures".

StratifiedKFold keeps each fold's class balance equal to the whole dataset's: it
shuffles the failing drives and the healthy ones separately and deals each class
into the folds in turn. It costs nothing and it is one word in the call. It does
not promise a smaller spread on every seed - here the stratified scores spread a
little more - it promises that no score rests on one drive.

The rule: stratify whenever a class is rare. Which, as lesson 4 established, is
whenever the problem is interesting.
:::

# Change only the seed, 40 times: how far does the score move? (optional)

![](cv_vs_single_split.png)

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

What to look at: the two dashed lines, the means, almost on top of each other;
then the widths. The single split spreads from 0.912 to 1.000, cross-validation
stays between 0.944 and 0.959. The same 40 seeds as numbers, test AUC:

- single 75/25 split: worst 0.9119, best 1.0000, mean 0.9600, best − worst
  0.0881, standard deviation 0.0221;
- 5-fold cross-validation: worst 0.9439, best 0.9588, mean 0.9532, best − worst
  0.0149, standard deviation 0.0032.

Forty seeds, changing nothing but the seed. The single-split column is the first
40 of slide 7's 200 splits, seeds 0 to 39, which is why its mean is 0.960 and not
slide 7's 0.955: forty draws centre less exactly than two hundred. Cross-validation has a seed of its
own - shuffle differently and the folds change - so it reduces the variance of
the estimate, it does not abolish it. Each number in the right column is already
a mean of five round AUCs, so the column is 40 × 5 = 200 fits.

What the seed changes is the reason for the gap. For a single split it picks
which 200 drives are tested, so which 7 of the 29 failures count. For
cross-validation it only changes how the 800 are cut into folds: every drive is
still predicted once, and all 29 failures count, each in the AUC of its own fold.

A predictable slip: "the mean of the five fold AUCs is the AUC of all 800
out-of-fold predictions". Every drive is predicted once, so it sounds right;
but pooling ranks drives scored by different models against each other, so
the two are different calculations (handout section 4.7).

The range, best minus worst, falls from 0.0881 to 0.0149, and the standard
deviation, last number in each line, by a factor of about seven: 0.0221 ÷ 0.0032 = 6.9 (the
notebook, unrounded, prints 6.8). Say which one "seven" refers to: a student who
divides the two ranges gets 0.0881 ÷ 0.0149 = 5.9, and will ask.

What 0.0032 is not: the error bar of 0.9514. It is how far cross-validation moves
when only the fold cut changes on the same 800 drives. On new data it moves by
0.0165 (slide 17). A predictable slip: "cross-validation is accurate to ± 0.003".
It is less arbitrary, not that precise.

Then the centres, because someone will have noticed in notebook 1 that
cross-validation came out lower: 0.953 against 0.960. That gap is about two
standard errors on forty draws, and with a different forty seeds it reverses
sign. So cross-validation is neither more optimistic nor more pessimistic here -
it is less arbitrary. The histogram on the slide puts both on the same bins, so
the bar heights compare directly; it comes from notebook 1 and is in handout
section 4.7.

What it costs is k fits instead of one. For everything in this course that is
seconds. For a large network it is a real decision, and the usual compromise is
a single validation set with enough rows that its own error bar is tolerable.

What it does NOT fix is the subject of slides 42 to 51.
:::

# How many folds? (optional)

- **k = 5 or 10** in practice
- Larger k: less pessimistic, more fits to run, round scores can be more correlated
- k = m (m the number of rows) is **leave-one-out**: for very few rows and a per-row metric; with AUC it returns nan (one row holds no pair)

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

The question always comes, so answer it before it does.

Larger k means each round's model trains on more data - 80% at k = 5, 90% at
k = 10 - so the estimate is less pessimistic: the bias shrinks. A predictable slip
is "larger k is more optimistic": cross-validation errs low, and a larger k
errs less. It also means more fits, and the training sets overlap even more
heavily - any two share 75% of their rows at k = 5, about 90% at k = 10 - so the
models are nearly the same and the round scores can be more strongly correlated
(on this fleet the correlation is small, slide 16). A predictable slip: "the
scores are more correlated because the validation fold is smaller". It is the
overlap of the training sets that makes the models alike.

Five and ten are the conventional choices and the difference between them rarely
matters. Leave-one-out is the extreme, and the third bullet says when it is
worth it: very few rows and a metric computed row by row (mean squared error,
accuracy). Each model trains on m - 1 rows, so it is nearly unbiased, and there
is only one way to cut, so no seed: with 30 patients, losing a fifth of them per
fold costs a lot, losing one costs almost nothing. The price is m fits and
notoriously high variance for the estimate itself - the m models are nearly the
same, and each score rests on one row. With AUC it cannot work at all: a fold of
one row holds no failing-healthy pair, so its AUC is undefined - cross_val_score
with LeaveOneOut returns nan for every round. On these 800 drives: five or ten
folds, or repeated folds.

Is k a hyperparameter? No - another predictable slip. It is a setting of the
measurement, not of the model: the model you ship is refitted on all the data
whatever k was. Fix it beforehand, and never choose it by the score it gives.

If you have the compute, repeated k-fold - the whole 5-fold procedure run with
several different shuffles, averaged; in scikit-learn
RepeatedStratifiedKFold(n_splits=5, n_repeats=10) - buys more than moving from 5
to 10. Measured on 40 seeds: 5-fold, 5 fits, standard deviation 0.0033; 10-fold,
10 fits, 0.0036, and its mean moves by only 0.0007; 5-fold repeated twice, the
same 10 fits, 0.0025; repeated ten times, 50 fits, 0.0009. What repetition
removes is the dependence on the fold cut; the uncertainty about new data, 0.0165,
stays.

A predictable slip: "ten repeats of 5-fold give me 50 independent tests". The
software returns 50 scores, and more measurements usually buy precision; but
all 50 reuse the same 800 drives, so repetition removes the dependence on the
fold cut and nothing else.
:::

# When the rows are not independent

- Suppose ten readings per drive (lesson 4's table has one)
- A random 5-fold split puts about 8 of a drive's 10 readings in training, 2 in validation
- Those 2 are near-copies of the 8, same label: the model recognises the drive (1-nearest neighbour: 1.000 split at random, 0.620 by drive)

::: notes
Everything so far assumed rows are independent draws. That assumption is
violated constantly and silently, and random k-fold gives no warning.

The concrete case, which they will meet - a hypothetical one here, since the
fleet table has one row per drive: suppose it had ten telemetry readings per
drive. A random 5-fold split puts about eight of a drive's ten readings in
training and two in validation, in every round. The model does not need to learn anything about failure - it learns to
recognise that drive, and the test row is nearly a duplicate of a training row.

The score is excellent and the model is worthless on a drive it has never seen,
which is the only situation anyone cares about. Why the test row is a near
duplicate: a drive's ten readings have nearly the same values and the same label,
so the model can look the answer up instead of learning the pattern. It is a form
of leakage: the test row is not independent of the training rows.

The third bullet carries the evidence; the nearest neighbour is named ahead of
lesson 6 - say only that it copies the label of the closest training row.

It bites flexible models. Simulated in the handout, section 4.6 - the 800 drives
with ten noisy readings each, 8,000 rows: a 1-nearest-neighbour classifier
(lesson 6) scores 1.000 with rows split at random and 0.620 with each drive's
readings kept together, the same as with one row per drive. A random forest
(lesson 7): 0.999 against 0.896. Logistic regression, six coefficients, cannot
memorise 800 drives: 0.960 against 0.950.

Other instances worth naming out loud: several images of the same patient,
several sentences from the same document, repeated measurements of the same
machine, any panel data.
:::

# Split by the thing you must generalise to

- `drive_id`: one value per row, the drive's serial number; not a feature, passed apart
- **GroupKFold** keeps a drive's rows in one fold: 640 drives train, 160 others validate
- `cross_val_score(model, X, y, groups=drive_id, cv=GroupKFold(5))`
- Test split too: `GroupShuffleSplit`; rare class: `StratifiedGroupKFold`

::: notes
The rule: split along the axis you need to generalise across. If the model must
work on a drive it has never seen, no drive may appear in both training and
validation. GroupKFold takes a group label and guarantees that.

Read the bullets in order. First what drive_id is: a column with one value per
row, the serial number of the drive the reading came from - ten rows with 0, ten
with 1, up to 799. It comes from the data, as a medical record carries the
patient's id; it is NOT a feature, because a model that sees it can recognise the
drive again, so it is passed apart, with groups=. Then what GroupKFold does with
it: every row with the same value goes into the same fold, so each round trains
on 640 drives (6,400 readings) and validates on 160 others (1,600). Atom 7 prints
the first rows with the drive_id column, then runs this exact call. Groups may differ in
size - one drive with ten readings among drives with one each is fine; the
predictable slip is "GroupKFold needs equal-size groups". What unequal groups
change is weight: that drive counts ten times in training and in the score, so
aggregate to one row per drive or weight its readings 1/10.

Two cautions. GroupKFold is not stratified: with 29 failing drives a fold can get
few of them; StratifiedGroupKFold does both. And the test set must be grouped too:
train_test_split takes no groups, GroupShuffleSplit does.

The tell that you needed GroupKFold and did not use it: a validation score that
is excellent and a production score that is not.
That gap is among the most common causes of a model failing after deployment.
:::

# Notebook 1, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| two honest experiments, 5-6 | `train_test_split(..., random_state=seed)`, `make_pipeline(StandardScaler(), LogisticRegression())`, `roc_auc_score` → seed 3 gives 1.000 |
| how wide, 7-8 | the same, 200 times → 0.885 to 1.000 |
| choosing on one split, 9-12 | three values of C, a winner per split → 89, 42, 38 outright wins, 31 ties |
| k-fold by hand, 13-17 | `StratifiedKFold`, then `cross_val_score` → mean 0.9514; 300 new fleets → 0.0165 against 0.0084 |
| stratify, 18-19 | `KFold` against `StratifiedKFold` → 1 failure, AUC 1.000; 0.0881 against 0.0149 |


::: notes
The table is the notebook's map: each section, the calls it makes, the slides
that introduced the idea, and the number to come back with.

On 6 November, with lesson 4 taking the first hour: run atoms 2 and 3 of
Consolidation/05_lesson_5_atoms.ipynb instead (the five round AUCs and their
mean; five rounds, five models), and leave the notebook to self-study with this
table as its map. With a full three hours: run notebook 01. Fifteen minutes.

Have them run the five-seed cell first and watch seed 3 return 1.000 before you
say anything about it.

The cell worth protecting if time runs short is the 200-split model selection - it
is the one that changes behaviour rather than just adding knowledge.
:::

# Break

- Ten minutes

::: notes
Ten minutes. The second half is bias-variance, which needs them awake.
:::

# Error too high: more data, or another model?

- Daily energy use from the outdoor temperature, as in lesson 3
- A straight line on 25 days: 5-fold cross-validated mean squared error (MSE) **6,064.3 kWh²**, a typical miss of √6,064.3 ≈ 78 kWh
- **More days** or **another model**: each cures a different error
- Find where the error comes from: first in a lab, then on real data

::: notes
This is the question the second hour answers, so pose it as a real decision.
Ask the room: more data or another model? Most will say more data. Do not
correct them yet; slide 37 gives the answer on these same 25 days.

Where the number comes from: one sample of 25 days from the curve of the next
slide. Not 25 consecutive days - one month never spans −5 to 38 °C - but 25
days scattered over the year, each with its mean outdoor temperature drawn at
random, uniformly, between −5 and 38 °C. Then 5-fold cross-validation, so each round fits the line on 20 days and
scores it on the other 5; 6,064.3 is the average of the five validation MSEs.
Notebook 2, section 4, prints it. It is a VALIDATION error, computed inside the
training data: the test set is not touched while we diagnose (lesson 1's rule,
and slide 53's three sets).

The arc, in five steps, so you can say where we are at any moment:
1. The question (this slide).
2. The lab, slides 26 to 35: a world where we wrote the truth ourselves, so we
   can see where the error comes from. There it splits exactly into three parts,
   bias², variance and noise, and each needs a different cure.
3. What follows in the lab: the total is a U over complexity; there is a floor
   no model goes below; the best model moves with the amount of data.
4. Back to real data, slides 36 to 39: we do not know the truth, but training
   error against validation error shows the same two parts, and learning curves
   add the training-set size.
5. The decision, slide 40: which fix to buy - and the answer to this slide.

Why go through the lab at all, if cross-validation does the diagnosis? Because
cross-validation gives two numbers, and the decomposition is what says what they
mean. The formula is the WHY, cross-validation the HOW. Four things only the
formula gives: why training error measures bias and the gap measures variance;
why more data cures variance and not bias; that there is a floor, so a test
error below it is leakage; why the best model depends on the amount of data. A
doctor reads symptoms in the clinic, but learnt what they mean in physiology,
where the mechanism is known; the lab is our physiology.

History, if there is a minute: the decomposition is old in statistics, and Geman,
Bienenstock and Doursat brought it into machine learning in 1992, in "Neural
networks and the bias/variance dilemma". Handout section 5.
:::

# A lab where we know the truth

- Truth: f(t) = 240 + 0.55 (t − 18)² + 12 sin(t/4) kWh, t = daily mean temperature (°C); noise 22 kWh (standard deviation)
- 300 new training sets of 25 days at −5 to 38 °C, each drawn afresh (slide 25's is the first)
- Polynomials of degree 1 to 12: `make_pipeline(PolynomialFeatures(d), LinearRegression())`

::: notes
The lab of the arc. Slide 25 asked where the error comes from; here we build a
world in which that can be seen, because we know the answer. A model trained on
some days is asked about a new day; its error is measured squared, as lesson 3's
mean squared error (MSE). The next slides split that squared error into three
parts - two that belong to the model, one to the data - and slides 30 and 31
measure each with numbers.

Then the data. One observation is one day: that day's mean outdoor
temperature and its total energy use. The 18 in the formula is the classic
balance temperature of degree days, the measure energy utilities have used since
the early twentieth century: below about 18 °C of daily mean a building heats,
above it cools, and both cost energy. A caveat if someone checks the range: a
daily MEAN of 38 °C is extreme - Italian heat waves stay around 33-35 °C as
daily means - and the temperatures are drawn uniformly, so a −5 °C day is as
likely as a 20 °C one, unlike a real year. The range is chosen to show the
curvature and the edges, not for realism; the data are synthetic.

Lesson 3 also predicted energy from temperature, with the same noise of 22 kWh,
so the setting is familiar. A predictable mistake is to think this is lesson 3's
dataset; the axes and the noise invite it. It is not: lesson 3 used a plain
parabola, 240 + 1.15 (t − 18)², 30 days, temperatures from −5 to 35 °C. Here
the parabola is gentler and carries a seasonal sine term, so no polynomial of
any degree matches the curve exactly, and every degree keeps some bias. The
function is `true_curve` in `Notebooks/methodology_data.py`; handout section 5.3.

Why a curve we wrote: to measure how far models are from the truth we need the
truth, and to know the floor we need the noise level. Real data gives neither.

Then the procedure. Draw 25 days at random, fit the polynomial, keep the model;
do it 300 times. For each degree that gives 300 models, all trained the same way
on different days. In real life you would have one of them. Slide 25's 25 days
are the first of the 300 (same generator, same seed).

A predictable mistake: "the 300 models come from different splits of slide 25's
25 days". Natural, since slides 13 to 22 always split one dataset. But the 300
training sets are NEW, independent samples, each drawn afresh from the curve -
new temperatures, new noise - which only a curve we wrote allows. Splits of one
sample would not do: models trained on parts of the same 25 days share most of
their data, so they agree more than models from truly different samples, and
the variance would come out too small. Every error that
follows is computed from these 300 models, at 220 test temperatures spaced
evenly from −4 to 37 °C.
:::

# Three hundred parallel universes: one model each

![](many_universes.png)

::: notes
Before any definition, look at the models; the figure's title says how to read
it. Each faint line is a model fitted to a
DIFFERENT training set of 25 days from the same curve. Forty of the 300 are drawn, so the panel stays readable; the red
average is taken over all 300. In real life you drew exactly one of them and
never knew about the rest.

Left, degree 1: every line misses the curvature in the same way, and the red
average is nowhere near the black truth. That gap, between red and black, is
bias. The lines also fan out, more than degree 2's: that fan is variance, not
bias. A predictable mistake is to read bias off the fan - "the lines diverge,
so the bias is high" - which is natural, since both are error. But bias is
about where the average sits, and variance about how far the lines sit from it.
Slide 32 gives the numbers, bias² 5,250.8 and variance 723.8, and its notes
explain why a straight line pivots.

Middle, degree 2: average tracks the truth, spread is modest. Both small. It is
the best model of the table on slide 32 and of the U on slide 33, which is why
it is the one drawn here.

Right, degree 12: through the middle it follows the truth better than degree 1
ever could, and at the edges the lines fly off and drag the average off the top
of the plot with them. That scatter is variance.

Two things to point at in the right panel. The scatter is worst at the EDGES of
the temperature range, at both ends: there every training point lies on one side,
so nothing holds the polynomial down on the other. And the plot runs a little
beyond the data. The 25 temperatures are drawn between −5 and 38 °C, but the
coldest of 25 such draws is on average −5 + 43 ÷ 26 = −3.3 °C and the warmest
38 − 43 ÷ 26 = 36.3 °C (for n uniform draws the expected minimum sits one
(n + 1)-th of the range above the bottom), while the test temperatures run from
−4 to 37 °C. So at the two ends of the plot a model is often extrapolating: in
more than half of the training sets the coldest day is warmer than −4 °C,
(42 ÷ 43)²⁵ = 0.55. The middle is where the days are dense - about
25 × 8 ÷ 43 = 4.7 of them fall above 30 °C.

And any one of those faint lines is a model somebody actually deployed.
:::

# The bias–variance decomposition: expected error = bias² + variance + noise

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \text{bias}^2 + \text{variance} + \sigma^2$$

::: notes
Name it first: this is the bias-variance decomposition of the expected squared
error. It holds for squared error, mean squared error (MSE); for accuracy or AUC
there is no split this clean.

This is what the experiment measures. The left side is slide 25's question made precise:
the squared error of a model on a new day, averaged. The right side is the
split, three terms the next slide defines and slides 30 and 31 measure.

State it, do not derive it - handout section 5.2 does that in five lines, and it
is worth reading because the derivation is short and the cross terms vanishing
is the whole trick.

Read the symbols on the previous slide's figure. Pick one test temperature x,
say 30 °C, and imagine a vertical line through it in each panel.
- x is the input, the temperature: 30 °C.
- f-hat(x) is where ONE faint line crosses that vertical: the prediction of one
  model, fitted to one random training set of 25 days.
- y is the ENERGY measured on a new day at 30 °C: the truth, 330.5 kWh, plus
  fresh noise. Not the temperature.
- E averages over everything that is random: which of the 300 training sets the
  model came from, and the noise in the new day.
So the left side is the average squared miss of a model trained on random days,
predicting a new day at x. On the right, bias² is the squared gap between the red
and black curves at x, variance the spread of the faint lines around the red one
at x, and sigma squared the noise variance, 484. The next slide says what each
term means and why bias appears squared; slide 30 does it with numbers.

The important claim is that this is an IDENTITY, not an approximation or a
useful metaphor: the two sides are equal, exactly. The handout (end of section
5.2) and notebook 2 check it two ways without using it; there is no need to
quote those checks in class.

That matters pedagogically: they have seen this picture in every textbook drawn
as a schematic. Here it is measured.
:::

# What each term means

- A new day at x = 30 °C: y = its energy, truth plus fresh noise (kWh); f̂(x) = ONE model's prediction; 𝔼 = average over models and noise
- **Bias** = average prediction − truth (kWh, signed); the mean squared error, expanded, yields bias²
- **Variance** = average of (prediction − average prediction)²: the models' disagreement
- **Noise** σ² = 22² = 484: the days' scatter; known because we added it

::: notes
The symbols first, because the formula slide could not carry them. x is the
input, a temperature such as 30 °C. y is the energy measured on a new day at
that temperature, in kWh: the truth, 330.5 at 30 °C, plus fresh noise - so y is
random too. f-hat(x) is the prediction of ONE model, one faint line read at x.
The expectation averages over two random things: which of the 300 training sets
the model came from, and the noise of the new day.

Two predictable mistakes, both natural on a first read. "y is the temperature,
30 °C": no, the temperature is x; y is what we predict. And "f-hat is the 300
models, and we predict with their average": no, in real life you train ONE
model and predict with it. Each line is the model you might have had; the 300
are averaged only to MEASURE how wrong such a model is. The average model exists
only in this lab, to measure bias.

A third, the most natural of all: "the left side is (true value − prediction)²",
because that is how everyone pictures an error. It is (y − prediction)², with y
measured on a new day, truth plus fresh noise - and that is where sigma² comes
from. Put f(x) inside the square instead and only bias² + variance remain. In the
lab the formula fixes y's distribution, not y: two new days at 30 °C give two
different energies around 330.5, which is why even a perfect model misses by
484 on average.

On real data y is the one thing you can measure; f and sigma are what only this
lab knows.

Then the three terms of the formula, in words, now that the picture has shown them.

Bias first. It is a gap with a sign: the average of the 300 predictions minus
the truth, in kWh. At 30 °C, for the straight line, 303.6 − 330.5 = −26.9 kWh:
the average line sits below the curve there. Why does it appear squared? Not by
analogy with the error, and not by choice: it is what the mathematics produces.
We set out to average the squared error, (y − f-hat)²; expand that average -
handout section 5.2 does it in five lines - and one of the terms that comes out
is exactly (truth − average model)², the bias squared, in kWh² like the other
two. That is why the three can be added. A bias of +26.9 and one of −26.9 cost the same.

Variance: how far the individual models sit from their own average, squared and
averaged - the fan of the previous figure. Bias asks where the average model
sits; variance asks how far the models sit from that average. Separate
questions: a model can have both, and degree 1 here does.

Noise: the scatter of the days around the true curve, 22² = 484. It belongs to
the data, not to the model, so no model removes it. We know it only because we
generated the data; on real data it cannot be separated out.

So bias and variance belong to the model, the noise to the data. The next slide
puts numbers on all three, at 30 °C.
:::

# The straight line (degree 1) at 30 °C, in three numbers

| Step, degree 1 | Formula | Value |
|---|---|---|
| The truth | f(30), known because we wrote f | 330.5 kWh |
| The 300 predictions | one per model | 300.8, 317.1, 304.8, and so on, from 215.6 to 364.7 kWh |
| Their average | sum ÷ 300 | 303.6 kWh |
| **bias²** | (303.6 − 330.5)² | **724 kWh²** |
| **Variance** | sum of (prediction − 303.6)² ÷ 300 | **856 kWh²** |
| **Noise** | 22² | **484 kWh²** |
| Expected error | bias² + variance + noise | **2,064 kWh²** |

::: notes
The bridge from the picture to the table on slide 32: the formula holds at one
test temperature at a time, so do it at one. The 300 degree-1 models of slide 27,
each asked for its prediction at 30 °C. The numbers are printed by notebook 2,
section 2, "First at one temperature".

Read the last row as one sentence: a straight line trained on 25 random days,
asked about a new day at 30 °C, misses by a squared error of about 2,064 kWh² on
average - 724 because the line cannot bend, 856 because it depends on which 25
days it saw, 484 because the new day is itself noisy.

The kWh² values are rounded to whole numbers so that the column adds up on the
slide: 724 + 856 + 484 = 2,064. Back in kWh: √2,064 ≈ 45 kWh, the typical miss of
a straight line on a new day at 30 °C.

A predictable slip in the variance row: squaring after summing, "(sum of
predictions − 303.6)² ÷ 300". The square goes on each term first; summed first,
the differences from the average cancel to zero. Noise is the one term nobody estimates here: we
added it, so we know it. On real data it cannot be separated out.

If someone asks whether the formula really holds: the handout (end of section
5.2) checks it without the formula, on simulated new days. Leave the numbers
there; this slide carries one example, degree 1 at 30 °C.
:::

# Degree 1 at 220 temperatures, −4 to 37 °C, then averaged (optional)

![](error_by_temperature.png)

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout. If skipped, go from slide 30 straight to the U of
slide 33, and say that each of its points is the 30 °C computation averaged over
220 temperatures.

The same computation as the previous slide, repeated at every temperature of a
grid, for degree 1 only. Why a grid: the error depends on the temperature, so to
get ONE number per degree - the table of the next slide - the 30 °C computation
is repeated at 220 temperatures evenly spaced from −4 to 37 °C (one every 0.19
°C) and averaged; in effect, an average over the whole range. 220 is arbitrary:
any fine grid gives nearly the same averages. −4 to 37 sits just inside the
data's −5 to 38, so the models are not judged where no day can fall. The grid is
TEST_TEMPERATURE in methodology_data.py. Notebook 2 draws the figure, right after
the 30 °C example.

Red is bias²: zero where the straight line happens to cross the curve, near 5
and 28 °C, and large at the two ends and in the middle, where the line
passes above the bottom of the curve. Its average over the 220 temperatures, the
dashed line, is 5,250.8. So 30 °C was a lucky spot for degree 1: its bias²
there, 724, is far below the average.

Blue is variance: lowest in the middle, where the days are dense, and rising
toward both ends, where the line pivots. Its average is 723.8.

The dotted line is the noise, 484 everywhere: sigma squared does not depend on
the temperature. The total at each temperature is not drawn; it is the sum of
the three curves.

The averages make the first row of the next slide: 5,250.8 + 723.8 + 484.0 =
6,458.6. Every row of that table is built the same way, one degree at a time.

And the caveat, once more: we can split the error into three only because we
know the truth and the noise. On real data y is measured, but f and sigma are
not, so you never see the three parts. Slide 36 says what you can measure
instead, and slide 37 shows it gives the same diagnosis.
:::

# Every degree, on 25 days (optional)

| Degree | bias² (kWh²) | Variance (kWh²) | Noise (kWh²) | Total = bias² + variance + noise (kWh²) |
|---|---|---|---|---|
| 1 | 5,250.8 | 723.8 | 484.0 | 6,458.6 |
| 2 | 59.6 | 70.8 | 484.0 | **614.3** |
| 3 | 44.5 | 125.8 | 484.0 | 654.4 |
| 5 | 8.9 | 368.9 | 484.0 | 861.8 |
| 9 | 80.1 | 65,186.0 | 484.0 | 65,750.1 |
| 12 | 153,537.3 | 32,174,344.9 | 484.0 | 32,328,366.2 |

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout. If skipped, go from slide 30 straight to the U of
slide 33, and say that each of its points is the 30 °C computation averaged over
220 temperatures.

Each row is averaged over the 300 training sets of 25 days and the 220 test
temperatures, as slide 31 showed for degree 1. The count, if asked: the SAME 300
training sets serve every degree (the notebook restarts from seed 0), so the
degrees differ only in the model, a fair comparison; 6 degrees × 300 = 1,800
models, each asked at 220 temperatures, 396,000 predictions. No new day is
simulated: bias² and variance need only the predictions and the truth, and the
noise enters exactly, as 484. The totals add the unrounded parts, so a student adding the printed
ones may be off in the last digit: 59.6 + 70.8 + 484.0 = 614.4, printed 614.3.

Noise is 484 in every row - point at the column and say that it never
moves, because it is a property of the data and not of the model.

Degree 1 is a bias problem: bias² of 5,250, two orders of magnitude above the
next row. Wrong everywhere, in the same way.

Degree 2 is the "about right" panel of slide 27, and the minimum, 614.3;
degree 3 is close behind, 654.4.

Do not say degree 1 has the smallest variance - it does not, 723.8 against degree 2's
70.8, and it is on the slide. Every universe draws its own 25 temperatures, and
a straight line that cannot follow the curve pivots according to where they
fell. Handout section 5.3 measures what happens when the temperatures are held
fixed instead: the variance column then climbs from the first row.

Degree 12 is a variance problem, and the magnitude deserves a pause: 32 million,
four orders of magnitude larger than anything else. On 25 points a degree-12
polynomial nearly interpolates, and an interpolating model is a machine for
amplifying noise.

Someone will notice that degree 12 also has the worst bias², which contradicts
"flexible models have low bias". Bias and variance stay separate quantities;
what fails is the measurement. A few wild fits dominate both columns, the
average of 300 such curves is itself unstable, and another seed gives very
different numbers for both (handout section 5.3). Read the two together:
degree 12 is unusable, and its exact digits mean nothing.
:::

# The bias–variance trade-off: the total error is a U

![](bias_variance_tradeoff.png)

::: notes
Too rigid on the left, too flexible on the right: say it while pointing. Read it as the answer to the lab: degree 1, on the left, fails by bias (bias²
5,250.8 against variance 723.8); degree 12, on the right, by variance (32 million);
degree 2 is best, 614.3. Each point averages the 30 °C computation of slide 30 over
220 test temperatures.

The textbook picture of this trade-off, measured instead of drawn. Many of them
will meet the drawn version later, in every machine-learning text. The textbook version: bias falling
monotonically, variance rising, the total a U. Two of those three are not what
the measurement gives.

Bias² falls only as far as degree 5 - 5,250 down to 8.9 - and then climbs to
153,537, for the reason in slide 32's notes: at that variance the average of
the fits is not a meaningful curve. And variance starts at 723.8, drops to 70.8
at degree 2, and only then rises.

What does survive is the part the lesson needs: the total error is a U with its
minimum at degree 2. Say that the cartoon is a well-specified model on a fixed
design, and that real measurements are lumpier - it is a better lesson than the
tidy version, and it is on the slide in front of them.

The y-axis is logarithmic, and say why: otherwise degree 12 flattens everything
else onto the baseline. That in itself tells them something about the scale of
the failure.

The dotted line is the noise floor at 484.

Only degrees 1, 2, 3, 5, 9 and 12 are measured; the segments between them only
join the points, nothing was measured at 4, 6 to 8, 10 or 11.
:::

# The floor you cannot go below (optional)

- Noise variance is **484** = 22², the noise put into the data, at every complexity
- A **test** error well below a known floor means leakage or a mistake in the evaluation, not excellence

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

The noise column never moves, because it is a property of the data rather than
of the model. No model, no algorithm and no quantity of data brings the expected
error on NEW data below it. Training error is another matter: degree 12 on its own
25 points goes far below 484, by fitting the noise. That is why the bullet says
test error.

The corollary is a practical diagnostic worth writing down: if you know your
measurement precision and your model beats it, something has leaked. This is a
real check used in engineering and in the physical sciences, and it catches
errors nothing else catches. Slide 37 shows the training side: degree 5 trains
at 262.4, below 484, on days it has seen.

Two conditions, both on the slide in short. "Well below", not just below: the
floor bounds the EXPECTED error, and a small test set can dip under it by luck -
on 5 test days even the true curve scores below half the floor about one time in
five (handout section 5.3). And "known": here 484 is known because we added it;
on real data the floor comes from outside, the instrument's stated precision or
repeated measurements under the same conditions. Without it the check cannot be
made.

Connects forward to the last hour, where we manufacture exactly that symptom
deliberately.
:::

# The best model depends on how much data you have (optional)

| Degree; total error at training-set size m (bold: lowest in its column) | m = 25 | m = 60 | m = 150 |
|---|---|---|---|
| 1 | 6,458.6 | 5,998.6 | 5,861.5 |
| 2 | **614.3** | 570.6 | 555.3 |
| 5 | 861.8 | **540.9** | **506.9** |
| 12 | 32,328,366.2 | 7,626.9 | 899.0 |

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

The same experiment at three sample sizes. The best degree moves from 2 to 5 as
the data grows, because there is now enough of it to pin down the extra
coefficients and the variance has fallen accordingly.

Read the first row against slide 25's question. The straight line goes from
6,458.6 to 5,861.5 with six times the data: more days barely help, because its
error is bias, and bias does not shrink with m. Degree 2 on the same 25 days is
already at 614.3.

Degree 12 improves by a factor of four thousand between 25 and 60 observations
(32,328,366.2 ÷ 7,626.9 = 4,239), and of thirty-six thousand by 150 (÷ 899.0 =
35,960). At 150 it has overtaken the straight line, 899.0 against 5,861.5:
variance shrinks with data, bias does not (handout section 5.4).

Then the sentence worth putting on the board: "which model is best" is not a
question about models. It is a question about models AND the quantity of data
you have.

This explains something they will meet constantly - a paper reports a large
model beating a small one on a dataset far bigger than theirs, and it does not
reproduce for them. Neither experiment was wrong.
:::

# Without the truth: two errors you can still measure

- One training set, f and σ unknown: cross-validation inside it, test set untouched, gives two errors
- **Training error**, on the days the model was fitted to: it has followed them, so it hides its variance
- **Validation error**, on the fold it did not see: it pays bias, variance and noise
- Training error too high for the task: **bias**. Validation far above training: **variance**

::: notes
In code, notebook 2 section 4: cross_validate(model, X, y,
cv=KFold(5, shuffle=True), scoring="neg_mean_squared_error",
return_train_score=True) - the same cross_validate as cross_val_score, asked to
return the training scores too.

Back to the world of slide 25. Everything from slide 26 to 35 needed the true
curve and 300 training sets; real life gives one training set and no truth. What
it does give is the measured y on every day, and that is enough for two errors.

Training error: score each round's model on the days it was fitted to. A model
that has seen those days has already bent towards them, noise included, so the
spread that variance measures - how differently it would come out on other days -
does not show. What is left is roughly its bias plus the noise it did not chase.

Validation error: score it on the fold it did not see. That is a new day for the
model, so it pays everything the formula of slide 28 lists.

Hence the reading, which is the formula put to work: training error high, bias;
a gap between the two, variance. Without the lab this rule would be a recipe to
memorise; with it, it is a consequence.

What is unknown is the TRUE rule f, not our model: our model we choose.

"High" against what? In the lab the reference was the noise, 484; on real data
sigma is unknown, so the reference comes from outside. First, the error the task
needs. Take the root of the mean squared error, the root mean squared error
(RMSE), which is in kWh, the typical daily miss: the straight line trains at
√5,155.7 = 72 kWh and validates at √6,064.3 = 78 kWh. If the building manager
needs forecasts within about 50 kWh to buy energy ahead, the line already misses
by more on the days it has SEEN - the easiest test there is - so it cannot do
better on new ones: too rigid, bias, and more days will not change that. Degree
2 trains at √312.8 = 18 and validates at √450.4 = 21 kWh, both within 50.
Second, the instrument's known precision, when you have it. Third, a more
flexible model: if degree 2 trains at 312.8 and the line stays at 5,155.7, the
line cannot follow the data. And both problems can come together: training high
AND validation far above it. Pure bias is training high with validation close.

"Too high for the task" includes a comparison with a more flexible model: if
degree 2 trains at 312.8 where the line stays at 5,155.7, the line cannot follow
the data - that comparison is what separates bias from noise.

A predictable slip: "the training error is low, so the model has low variance".
The second bullet says why not: the training error HIDES variance, it does not
measure it. Degree 12 has a tiny training error and an enormous variance.

A predictable mistake: "on real data f is unknown, so the decomposition is only a
proof that cannot be used". It is reasonable - none of slide 32's columns can be
computed on real data. But the decomposition is what says what the two measurable
errors mean, and why the cures of slide 40 work.

The test set stays out: both errors come from cross-validation inside the
training data. Slide 37 does it on the 25 days of slide 25. Handout section 5.5.
:::

# One sample of 25 days, 5-fold cross-validation

| Degree | Training MSE, kWh² | Validation MSE, kWh² | Reading | What the lab measured |
|---|---|---|---|---|
| 1 | 5,155.7 | 6,064.3 | both high and close: **bias** | bias² 5,250.8 |
| 2 | 312.8 | **450.4** | lowest validation: **the best** | lowest total, 614.3 |
| 5 | 262.4 | 615.1 | training below the noise, 484; gap 615.1 − 262.4 ≈ 353: **variance** | variance 368.9 against degree 2's 70.8 |

::: notes
One sample of 25 days from the curve of slide 26, drawn with seed 0, 5-fold
cross-validation with KFold(shuffle=True, random_state=0); each number is the
average over the five rounds. Notebook 2, section 4.

Read it row by row against the lab.

Degree 1: both errors high and close together. The line is just as wrong on days
it has seen as on days it has not, which is what bias looks like. The lab said
bias² 5,250.8. Slide 25's 6,064.3 is this row: so the answer to slide 25 is
already here - the straight line's error is bias, and more days will not cure it.
Slide 35's first row confirms it in the lab: six times the data, 6,458.6 down to
5,861.5 only.

Degree 2: the lowest validation error, 450.4, the best of the four; in the lab
it was the lowest total too. Best on VALIDATION, not on training: degree 5 trains
lower, 262.4 against 312.8. A predictable mistake is to pick the model with the
lowest training error - natural, it looks like the most accurate - and it would
pick degree 5, the one chasing the noise. The choice is always made on the
validation error.

Degree 5: training 262.4 is BELOW the noise variance of 484. No model can predict
new days better than the noise allows, so a training error below it means the
model has fitted part of the noise. Validation 615.1, a gap of 615.1 − 262.4 ≈
353 on the slide (352.6 from the unrounded errors): variance.

Degree 3 sits between, 306.1 against 525.6; degree 12 is left out because, with
unscaled powers up to t to the twelfth, the solver loses precision and its
training error is not trustworthy.

The numbers differ from the lab's, and must: these are one sample and five folds
of 20 days, the lab is an average over 300 samples. What carries over is the
ordering and the pattern, and that is all a diagnosis needs.

With 150 days (handout section 5.5): degree 5's gap falls from 352.6 to 58.3 -
more data cured its variance - while degree 1 stays high, 7,281.1 against
8,058.6.
:::

# Learning curves: would more data help?

- Slide 37's two errors, repeated at growing training-set sizes: `learning_curve(model, X, y, cv=folds, scoring="roc_auc")`
- On slide 6's 800 drives (lesson 4's fleet, cut down), scored by the area under the curve (AUC): **higher is better**, so poor = **low** curves
- Two models broken on purpose: one feature only; 6 features + 150 noise columns, almost no penalty

::: notes
Slide 37 answered "bias or variance?" at one size. The question of slide 25 was
about buying data, and that needs the errors at several sizes.

Why switch to the drives: the method is the same for any score, and on the
drives we can show both failures with real features. Say clearly that the score
flips direction: an error is better low, an AUC is better high. So on the next
slide "bad" is LOW, the mirror image of the energy table. A predictable mistake
is to call the AUC "the error" - natural after twelve slides of errors - and then
read the curves upside down. The AUC is a score: 1 is perfect, 0.5 is chance.

The data: here the demo uses all 800 drives with 5-fold cross-validation,
because nothing from it is reported as a final score. In practice set the test
set aside first and run the learning curve on the rest.

The two models: the rigid one sees only power_on_hours; the flexible one sees the
six features plus 150 columns of random noise with C = 1000, so almost no
penalty.
:::

# Learning curves: left the curves meet low, right a gap that stops closing

![](learning_curves.png)

::: notes
The figure's title repeats the reading key: AUC, higher is better.

What the x axis counts: the 800 drives are cut into 5 folds of 160. In each
round one fold of 160 validates and the other four, 640 drives, are the training
part; the curve trains on a growing share of those 640, from 96 up to all 640,
and always scores the same 160 held-out drives. Each point is the mean of the 5
rounds, so each model is refitted 8 sizes × 5 rounds = 40 times.

5-fold cross-validation on the 800 drives, so each round's training part is 640
drives. Train on 15% of it, 96 drives, then on larger shares, eight sizes up to all
640, plotting training score and cross-validated score against the number of
examples used. The first bullet of the next slide names the scikit-learn call.

Left, one feature only, power_on_hours, logistic regression with C = 1: the curves
MEET, and they meet LOW - 0.715 against 0.718 - and the validation curve is
exactly flat, 0.718 at every size. Say why, because a sharp student will ask:
with a single feature, logistic regression ranks the drives in that feature's own
order whatever coefficient it learns, so its AUC is power_on_hours' own AUC and
no amount of data can move it. It is the extreme case of bias, chosen to make the
shape unmistakable.

Right, six features plus 150 columns of pure noise, logistic regression with
C = 1000, almost no penalty: training exactly 1.000 against validation 0.847, a
gap of 0.15.

Then read the right-hand curve to the end, because this is where the room gets
it wrong. It climbs to
0.857 at 251 examples and then stops: 0.834, 0.847, 0.835, 0.832, 0.847. It ends
below its own high point.

Someone will say it is still rising, because the last segment goes up by 0.014.
Have the answer ready: across the five rounds that step is -0.030, +0.050,
+0.006, -0.009, +0.053. It is a wiggle inside a band of about 0.04, not a trend.
A slope smaller than its band is not a slope.

A predictable slip on the x axis: "the point at 640 is the model I deploy, on all
800 drives". The rightmost point is the largest training part a round holds, 640
of the 800; the deployed model is refitted afterwards on all of them.

A predictable slip on the left panel: "the two curves meet, so it is bias". The
title says meet LOW, and the word is the diagnosis: a good model's curves also
approach each other as the data grows, but high. Here they meet at 0.715 against
0.718.

Say that the right-hand setup is not contrived. It is what a wide feature table
looks like when most columns carry nothing, which is most feature tables.
:::

# Which fix to buy

- Curves meet at a **poor** score: bias; more data will **not** help
- Wide gap, validation **still rising** by more than its spread: more data **likely** helps
- Wide gap, validation **flat**: smaller C or **remove features**
- Slide 25's answer: bias; degree 2, same 25 days: MSE **450.4**, not 6,064.3
- The test set: scored once, after the fix

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

High variance splits in two, and the slope is what splits it. A curve still
rising is itself the evidence that more data will pay - the only case where
"collect more" is a plan rather than a hope. A curve that has flattened, which
is what our right-hand panel does after 251 examples, says the gap is real and
more rows of the same kind have stopped buying anything: what is left is
stronger regularisation, fewer features, a simpler model. Here the 150 noise
columns are the problem, and no quantity of rows removes a column.

Name the predictable mistake and defend the instinct: prescribing data for a
bias problem is what almost everyone does, and the reasoning is sound - more
data almost always helps and never makes a model worse. It just does not help
THIS failure, and fifteen lines of code tell you which failure you have.

Two wording points. Bias is the curves meeting LOW, at a poor score: two curves
meeting high, near 1, are a model that works. And "remove features" means
removing columns - here the 150 noise columns. A predictable mistake is to hear
"drop" and think of dropout, the neural-network technique of lesson 9; it has
nothing to do with it. The stronger penalty is lesson 3's lambda: in
LogisticRegression it is set through C, and a smaller C is a stronger penalty;
the right-hand model used C = 1000, almost none.

Close the arc: this was slide 25's question. The straight line on 25 days had
both errors high and close, so its problem was bias. A month spent collecting
days would have bought almost nothing (slide 35: 6,458.6 down to 5,861.5 with six
times the data); changing the model to degree 2 brings the validation error from
6,064.3 to 450.4 on the same 25 days.

"Meet at a poor score" is written that way on purpose: for an error, poor is
HIGH, as in the energy table; for an AUC, poor is LOW, as in the drive curves.

And the test set. Every number in this diagnosis came from cross-validation
inside the training data. The test set is scored once, after the fix is chosen.
A predictable mistake is to diagnose on the test error - it is the number
everyone looks at, so the instinct is natural. But if the final test score
disappoints and you go back and change the model, the test set has become a
validation set (slide 53, "Three sets, three jobs"): its number is no longer
honest, and the report must say so or use fresh data.
:::

# Notebook 2, self-study (optional)

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 300 parallel universes, 26-27 | `PolynomialFeatures` then `LinearRegression`, in `make_pipeline`, fitted 300 times |
| the three parts measured, 28-34 | NumPy on the 300 predictions → sum and measured error agree to 2.73e-12 |
| more data moves the answer, 35 | the same at 25, 60 and 150 days → best degree 2, then 5 |
| without the truth, 25 and 36-37 | `cross_validate` with `KFold`, `return_train_score=True` → degree 1: 5,155.7 against 6,064.3 |
| learning curves, 38-40 | `learning_curve` with `StratifiedKFold` → 0.715 against 0.718; 1.000 against 0.847 |


::: notes
The table maps each section to its calls and its number. On 6 November this
notebook is left for self-study: show the table, say what each section measures,
and that its numbers are on slides 25 to 40, which were just shown.

When there is a full three hours, run it live, for twenty minutes.

The cell to protect is the decomposition table where the sum equals the measured
error to twelve decimal places. Let them see the identity hold rather than
telling them it does.

If time is short, skip the three-sample-size comparison and assign it as reading
 -  the table is in the handout.
:::

# Now: what cross-validation does not protect you from

- Slide 6's 800 drives and labels; features: **2,000 columns of random noise**; honest AUC **0.500**
- Keep the 10 columns with the highest F-test (failing against healthy means, relative to the spread in each group): `SelectKBest(f_classif, k=10)`
- Cross-validate a logistic regression on them: 5 rounds, train on 640, score 160

::: notes
Set this up carefully, because the demonstration only lands if they believe the
data is genuinely empty.

Two thousand columns drawn from a standard normal, independent of the label and
of each other; the labels are the real ones, the 29 failures among slide 6's
800 drives. There is nothing to find. Any model's honest AUC is one half.

Tell them what is coming so they can watch for it: we are going to write code
that looks completely reasonable, and get 0.93.

The second bullet names the selector. f_classif scores each column on its own:
how far apart the mean of the 29 failing drives and the mean of the 771 healthy
ones are, divided by how spread the values are inside each group - an
analysis-of-variance (ANOVA) F-test; with two classes it is the t statistic
squared. A large F means the column seems to separate the classes. SelectKBest
computes it for all 2,000 columns and keeps the k largest. Point out the clash of
letters: this k counts columns, not folds.

Why pure noise still offers columns that look good: the mean of 29 random values
is not exactly 0, it wobbles with standard deviation 1 ÷ √29 = 0.19. Across 2,000
columns a few drift far by luck; the ten chosen here have the failing drives'
mean shifted by 0.52 to 0.77, between 2.8 and 4.1 times what chance usually
allows. They look like signal and are the luck of those 29 drives.

The procedure in four steps, so nobody wonders where the training happens:
(1) the data, 800 drives, their real labels, 2,000 noise columns; (2) the
selector looks at all 800 rows once and keeps 10 columns; (3) inside
cross_val_score, 5 rounds: each trains a logistic regression on 640 drives using
only those 10 columns and scores the AUC on the other 160; (4) the five AUCs are
averaged. Cross-validation IS training and scoring five times, which is why the
word alone hides the training.
:::

# The code almost everyone writes

```python
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
sel = SelectKBest(f_classif, k=10)
sel.fit(noise, y)
X = noise.loc[:, sel.get_support()]
cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
```

::: notes
The first two lines are the model and the folds, as in the first half of today:
standardise, then logistic regression; five stratified folds. Then read the rest
aloud as the sentence a practitioner would say: "two thousand features
is too many, so let me keep the ten most predictive ones, then cross-validate a
model on those."

Every line is defensible in isolation. Feature selection is normal. Ten from two
thousand is reasonable. Cross-validating afterwards is exactly what the first half of
today recommended.

Do not reveal the bug yet. Ask them to find it before the next slide.
:::

# 0.931

- Cross-validated AUC **0.931**, the mean of five folds: 0.928, 0.979, 0.975, 0.943, 0.832, on data with no signal
- The bug is `sel.fit(noise, y)`: the selector saw **every row**, validation folds included
- It kept the 10 columns that best matched the labels it would later be graded on

::: notes
The result. Four of the five folds sit between 0.92 and 0.98: close agreement,
which is exactly what a trustworthy result looks like.

Then the bug, sel.fit(noise, y): SelectKBest was shown EVERY row,
including the rows that would later serve as validation folds. It searched two
thousand columns for the ones that best matched labels it had already seen, and
handed the winners to cross-validation.

The framing worth using: cross-validation did not fail. It was lied to. It
faithfully measured a procedure that had already peeked.

This is the single most common serious error in applied machine learning, and it
is committed in published work regularly.
:::

# Fix it, and look what happens

- Selection moved **inside** the pipeline, refitted in every round: `Pipeline([("select", SelectKBest(f_classif, k=10)), ("scale", StandardScaler()), ("model", LogisticRegression())])`
- 20 different fold seeds, 0 to 19: **0.658** on average, 0.576 to 0.759
- The truth is still 0.500

::: notes
The fix is one Pipeline: selection becomes part of the model, so it is refitted
on the training rows of each round.

The leak is mostly gone - 0.93 down to 0.66. And it has NOT come back to 0.5.
Twenty different fold seeds all agree that a signal-free dataset supports a
model better than chance.

Let that be uncomfortable for a moment before explaining. Most treatments of
this topic stop at "put it in a Pipeline" and imply the problem is solved.
:::

# Why it does not return to 0.5

- Some of 2,000 columns match the label **by accident**: in the **sample**, so in every fold
- Helps: a test set set aside **before** any search; its chance matches sit in other columns
- Fewer candidates: the luckiest of 20 is less lucky than the luckiest of 2,000
- More rows: a noise column's mean over 29 failures wobbles by 0.19; over 290, by 0.06

::: notes
This is the most useful idea in the lesson, so take the time.

With two thousand columns and eight hundred rows, some columns correlate with
the label purely by chance across the whole dataset. That accident is a property
of this particular sample. It is present in every subset of it - every training
fold and every validation fold alike.

So a selector fitted honestly on four fifths finds those columns, and they still
work on the remaining fifth, because the spurious correlation was never
fold-specific in the first place.

The statement to write on the board: cross-validation protects you from leaking
between folds. It cannot protect you from having searched a large space of
possibilities on a small sample. No rearrangement of the same 800 rows removes a
pattern that is genuinely in those 800 rows and absent from the world.

What helps, one bullet each, and none of them is another cross-validation:
cross-validation reshuffles the same 800 rows, where the accident lives.
A test set held out before any of this and used once: in new rows the chosen
columns' correlation is a fresh coin toss, on average zero; the test set has
chance matches of its own, but in other columns, which the model does not use.
Atom 11 shows it. Fewer candidates: the best of many lottery tickets is luckier
than the best of a few. More rows: what matters is the count in the rarer class -
a noise column's mean over 29 failures wobbles by 1 ÷ √29 = 0.19, over 290 by
1 ÷ √290 = 0.06 - so chance matches shrink and the selector finds nothing
convincing.
:::

# GridSearchCV: try all, keep the best

- **k** = columns kept: 2, 5, 10, 20, 40 (**columns, not folds**)
- **C** = penalty, smaller is stronger: 0.01, 0.1, 1, 10, 100
- 5 values of k × 5 values of C = **25 pairs**; each pair: 5-fold cross-validation, 25 × 5 = **125 fits**
- Best mean AUC wins: `best_params_` (k = 20, C = 10); `best_score_` = that mean
- Then refitted on all 800, **not scored again**

::: notes
Slide 45's honest pipeline still has two settings someone has to choose: how
many columns the selector keeps, and how strong the logistic regression's
penalty is. GridSearchCV is the loop that chooses them, the same loop as slide
10's recipe, written once for you.

In notebook 3 it is one line: GridSearchCV(honest, grid, cv=inner,
scoring="roc_auc").fit(noise, y), where grid is a dictionary with "select__k"
and "model__C" as keys - the step's name, two underscores, the parameter's name -
and inner is StratifiedKFold(5, shuffle=True, random_state=1).

It tries all 25 combinations, scores each by 5-fold cross-validation, keeps the
one with the highest mean AUC, and refits it on all the data: best_estimator_
is that refitted model. The order matters: best_score_ comes from the folds,
before the refit. The refitted model is the one you ship, and it is not scored:
scored on the 800 drives it trained on, it would give a training score.

A second predictable mistake: "best_score_ is the AUC of the model refitted on
all the drives". Natural, because the refit is the last thing the slide lists;
but the score was computed first, on the folds.

A third: "25 values of C, times 5 columns, 125 fits" - the 125 is right and the
reasoning is not. Five values of k times five values of C make the 25 pairs; each
pair costs five fits; the refit is fit 126.

Each pair's five round AUCs are in notebook 3. The winner, k = 20 and C = 10,
scored 0.750, 0.754, 0.879, 0.740 and 0.877; their mean, 4.000 ÷ 5 = 0.800, is
best_score_. The runners-up are nearly tied with it: k = 20 with C = 1 at 0.798,
k = 20 with C = 100 at 0.798. Handout section 7.3 prints the table.

A predictable mistake: "k in the grid is the number of folds". The reasoning is
natural - k has meant the folds since slide 13 - but here k is the selector's
parameter, the number of columns kept. The folds are fixed at 5 by cv=inner.
:::

# The best of 25 scores is inflated: 0.800 for a worthless model

| `GridSearchCV`: k = 2, 5, 10, 20, 40 columns × C = 0.01, 0.1, 1, 10, 100 | AUC |
|---|---|
| best of the 25 candidates, `best_score_` | **0.800** |
| average candidate | 0.726 |
| worst candidate | 0.684 |
| the luck of the winner: best − average = 0.800 − 0.726 | 0.074 |
| the truth, by construction | 0.500 |

::: notes
Choosing between configurations is choosing, and choosing on data costs the same
honesty. This is the last slide's search, run on the same signal-free table.

Two inflations, one on top of the other. The average candidate, 0.726, is
already far above the truth: that is slide 46's accident of the sample, present
even with the selection inside the pipeline. The winner adds its own luck on
top: 0.800 - 0.726 = 0.074, the fourth row.

The number a practitioner reports is best_score_, and it is the MAXIMUM of 25
noisy estimates. The maximum of a set of noisy numbers is biased upward even
when every one measures the same quantity - and here they nearly do, since the
average candidate is 0.726 and the worst is 0.684.

So 0.800 is not the performance of the chosen configuration. It is the
performance of whichever configuration got luckiest.

Hyperparameter search is the same problem as the selection of the last slides:
choosing on the data you then report. How much the choosing costs, measured
honestly, is what the next two slides build.
:::

# Nested cross-validation: slide 10's recipe, five times

- `GridSearchCV`'s folds both **choose and score**: `best_score_` is the luckiest of 25
- Set aside **160 drives** as the outer test
- Run the **whole search** on the other 640 (its own 5 inner folds), refit the winner on the 640
- Score it **once** on the 160; rotate, 5 times; mean of the 5 outer scores: **0.670**

::: notes
Start from what they already trust. Ordinary cross-validation measures a fixed
model, say k = 10 and C = 1: nothing is chosen while it measures, so its mean is
honest. GridSearchCV gives its folds two jobs at once - choose among 25
candidates, and score the winner - and the winner is whichever got luckiest on
those very folds.

The analogy that works: a student allowed to pick which exam to sit after seeing
all the questions. The mark is real, and it says little about the next exam.

Nested cross-validation separates the two jobs, and the bridge is the title:
it is slide 10's recipe - test aside, choose on the rest, refit, score the test
once - with cross-validation as the way of choosing, run five times so that
every drive is the outer test once. The whole search, choosing included, is
treated as the model, and cross_val_score measures it like any other model. outer is StratifiedKFold(5, shuffle=True, random_state=2): five
outer folds of 160. In each outer round the search runs on the other 640 drives
with its own five inner folds - train on 512, score on 128 - picks a winner,
refits it on the 640, and that model is scored once on the 160 it never saw.

What it costs to run: per outer round, 125 fits for the search plus 1 refit; 5
× 126 = 630 fits. The mean of the five outer scores is 0.670.
:::

# Nested cross-validation: it measures what choosing costs

![](nested_cv_diagram.png)

::: notes
Measure the search, not the winner. The figure is the last slide, drawn.

In code, one line: cross_val_score(GridSearchCV(honest, grid, cv=inner), noise,
y, cv=outer) - the search goes in where the model usually goes.

The figure's bottom lines carry the score that the last slide's table left out:
0.670, against best_score_'s 0.800. The difference, 0.800 - 0.670 = 0.130, is
the optimism of the search - larger than most differences anyone reports between
competing methods.

The inner loop may overfit its own data as much as it likes; the outer block was
never part of it.

The figure shows outer round 1 of 5, and inside it the 5-fold search over the
25 candidates (inner seed 1, outer seed 2 in notebook 3). Point at the arrow: the
640 drives pass to the search as one block and are re-split into 5 new inner
folds of 128; the inner folds are not the 4 outer blocks of 160. Over the 5
rounds every drive is in the outer test exactly once.

The nested figure is honest but itself noisy: its five outer folds range widely
(handout section 7.4). Say so in one sentence; the numbers stay in the handout.

One clarification that always comes up, so pre-empt it: nested cross-validation
is NOT a way to choose hyperparameters - it produces k possibly different
winners. It is a way to estimate what choosing costs. You choose on all the data
afterwards, and you report the nested figure.

A predictable slip: "0.670 is the score of the model I ship". The search ends
with a refit and the procedure returns one number, so it reads as a model and
its score; but the model refitted on all 800 drives is none of the five
outer-round models that produced 0.670. The figure measures the procedure,
search included.
:::

# Each piece of discipline costs score: 0.931, 0.800, 0.670, 0.500

![](leakage_ladder.png)

::: notes
The summary of the whole last hour, on one axis. Same data, containing nothing.

The rule that reads the ladder: a score is honest to the extent that the rows
that measure it took part in no decision. At 0.931 they chose the columns; at
0.800 they chose the configuration; at 0.670 they chose nothing. Handout section
7.4 tells it as an exam: answers seen in advance, the best of 25 mock exams, an
exam never seen - from the same question bank.

0.500 is the truth. 0.670 is nested cross-validation. 0.800 is what the search
reports. 0.931 is selection before cross-validation.

Say clearly: nobody in this picture wrote dishonest code. The difference between
0.93 and 0.50 is entirely a matter of which choices were made while looking at
which data.

Each step to the left is a piece of discipline, and each one costs a chunk of
the score you would otherwise have reported.

Even the nested 0.670 stays 0.17 above the truth: the chance correlations belong
to the 800-drive sample (slide 46), so no re-split removes them. Only a test set
set aside before any search, fewer candidates or more rows expose it. That is what discipline feels like
from the inside, and it is why it is unpopular.
:::

# The debt, paid: the honest threshold costs 1,400 EUR more

| Threshold (one model, fitted on 450 drives, for all three rows) | missed failures × 2,600 EUR + false alarms × 140 EUR | Cost on the 200 test drives (7 failures) |
|---|---|---|
| 0.50, the default: nothing chosen | 4 × 2,600 + 1 × 140 | 10,540 EUR |
| 0.050: the cheapest of 200 thresholds from 0.01 to 0.9, tried on the 150 **validation** drives | 1 × 2,600 + 15 × 140 | **4,700 EUR** |
| 0.171: the same sweep, on the 200 **test** drives (lesson 4's way) | 1 × 2,600 + 5 × 140 | 3,300 EUR |

::: notes
Lesson 4's threshold, done properly - but on slide 6's 800 drives, not lesson
4's 8,000, so do not compare these euros with lesson 4's: a test set of 200
drives with 7 failures has a smaller bill. Split three ways: fit on 450 drives,
choose the threshold on 150, report once on the remaining 200. ONE model, fitted
once on the 450, serves all three rows; they differ only in the threshold, and
all three bills are counted on the same 200 test drives. Row 1 chooses nothing.
Row 2 sweeps 200 thresholds from 0.01 to 0.9 on the validation drives'
probabilities and keeps the cheapest; row 3 runs the same sweep on the test set.

A predictable mistake: reading the first header as row 1's label ("fitted on 450,
chosen on 150"). It describes the model every row shares; row 1 chooses nothing.

The middle column is the bill, counted: the default misses 4 of the 7 failures; both other
thresholds miss 1, and differ only in false alarms, 15 against 5.

Then point at 0.050. The validation set chose almost exactly lesson 4's t* =
140 ÷ 2,740 = 0.051, the threshold the costs alone gave before anyone looked at
data. With 5 failures in the validation set part of that is luck, but it is the
point to land: the formula was already the honest answer; what needed paying was
the threshold lesson 4 picked by looking at its test set.

The honest threshold is WORSE on the test set than the cheating one. It is
supposed to be - and this is the slide where that finally makes sense to them.
The 1,400 euros between them are 10 false alarms that this particular test set
happened to let 0.171 avoid.

A predictable slip: "the debt is paid, so the number improves". The reasoning is
natural - doing things properly ought to be rewarded - but paying the debt makes
the reported number WORSE: it removes the luck the cheating number had borrowed.

The cheating figure describes a threshold tuned to this particular test set, and
there is no reason it survives contact with the next two hundred drives. The
honest figure is an unbiased estimate of what happens next.

Note the cost: we now fit on 450 drives instead of 600, because the validation
set has to come from somewhere. On a small dataset that hurts - which
is the argument for choosing by cross-validation on the training portion and
keeping the test set whole.
:::

# Three sets, three jobs

- **Train**: fit the parameters
- **Validation** (a held-out set, or the cross-validation folds): make every choice: model, hyperparameters, threshold
- **Test**: report. Touched **once**
- Change anything after looking at the test set, and it has become a validation set: **say so in the report**, or get fresh data

::: notes
The organising principle of the whole lesson, on one slide. Put it on the board
and leave it there.

Every choice belongs to the validation set. Which model, which penalty, which
features, which threshold, when to stop training, whether to try a different
approach entirely - all of it.

The test set has exactly one job and one use. The last bullet is the rule that
breaks most often: the moment you look at it and change something in response,
it has become a validation set, and you no longer have a test set at all - the
cheating row of the last slide.

The honest thing to say aloud, because they will do it anyway: everyone looks
twice sometimes. The requirement is that you SAY SO in the report. A stated
weakness costs a little credibility; an unstated one costs all of it when
somebody else fails to reproduce your number.
:::

# A fixed seed buys repeatability, not stability (optional)

- 30 seeds of the train/test split (600 drives to fit, 200 to test), same model: **0.913 to 1.000**
- Every run is perfectly reproducible
- `AUC 1.000 (random_state=3)` is reproducible **and** misleading
- Report the seed **and** the spread: AUC 0.951 ± 0.019 over 5-fold cross-validation, seed 0

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

Every one of those thirty runs is exactly reproducible by anyone with the same
code. They disagree with each other by nearly nine points.

So reproducibility is necessary and nowhere near sufficient. Report the seed AND
the spread.

Contrast the two ways of writing the same experiment up. "AUC 1.000,
random_state=3" - fully reproducible, thoroughly misleading. "AUC 0.951 plus or
minus 0.019 over 5-fold cross-validation, seed 0" - reproducible and honest; those
are slide 14's five round AUCs at seed 0: mean 0.9514, standard deviation 0.0187.

The second one is also the way to write up every exercise from here on - it is
what the exam asks them to defend.
:::

# What to fix, and what to write down (optional)

- Fix every `random_state` (the split, the folds), and seed your own NumPy draws with `np.random.default_rng(seed)`: a rerun gives the same number
- **Library versions**: defaults change between releases
- What you could **not** fix: parallel computations that finish in a different order, graphics processing unit (GPU) non-determinism

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

Practical checklist. The container exists for the second item: it pins the
environment so that their result and the examiner's are the same result.

Worth saying that a result nobody can reproduce in two years was not really
reproducible - and that scikit-learn genuinely does change defaults between
minor versions, which has invalidated published comparisons.

Commit the code that produced the number, not a tidied-up version of it.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| the catastrophe, 42-44 | `SelectKBest(f_classif, k=10)` on every row, then `cross_val_score` → 0.931 |
| selection inside the round, 45-46 | `Pipeline` with the selector as its first step → 0.658 |
| the search, 47-48 | `GridSearchCV`, `best_score_` → 0.800 |
| nested, 49-51 | `cross_val_score(GridSearchCV(honest, grid, cv=inner), noise, y, cv=outer)` → 0.670 |
| the threshold, honestly, 52-53 | `train_test_split` twice → 4,700 EUR against 10,540 |
| seeds, 54-55 | 30 values of `random_state` → 0.913 to 1.000 |


::: notes
The table pairs each section with its calls and its number. On 6 November: run atoms 9 and 10 of Consolidation/05_lesson_5_atoms.ipynb
instead (GridSearchCV's 25 candidates; nested cross-validation one outer round at
a time), after showing 0.931 appear in the notebook's first cell if there is a
minute, and leave the rest to self-study. With a full three hours: run notebook
03, fifteen minutes.

The cell to protect is the first one - let them watch 0.931 appear on empty data
before any explanation. Nothing else in the lesson changes behaviour as
reliably.
:::

# What to put in a report

- The metric, its **spread** (standard deviation of the fold scores, slide 16), and how it was estimated (e.g. stratified 5-fold, seed 0)
- How many **positives** the test set held
- Every seed, and every library version
- Which choices were made on which data

::: notes
A checklist they can apply to every remaining exercise and to anything they read.

The first item in code: scores = cross_val_score(...) with the folds written
out, then scores.mean() and scores.std(); in the report, "AUC 0.951 ± 0.019
(standard deviation over 5 folds), stratified 5-fold, seed 0". Not s ÷ √5, not a
confidence interval: slide 17 showed the real spread is twice that. With a single
test set there is one number and no spread of its own, which is why the second
item asks for the positives.

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

- **A score is a measurement**: 0.885 to **1.000**
- **Choosing on one split chooses noise**
- **High training error: bias; a gap: variance**
- **Choose on validation, report on test once**
- **What cross-validation cannot remove**:
    - chance correlations in the sample: 0.658 on pure noise
    - the luck of choosing: 0.800 reported, 0.670 nested
    - dependent rows, unless split with `GroupKFold`

::: notes
Close on the number they opened with. Seed 3, AUC 1.000, on a model whose honest
performance is 0.951.

The third bullet is slides 25 to 40 in one line: bias² + variance + noise, read
without the truth; and more data cures only the variance, the gap. Say "too high
for the task" aloud - the slide had no room for it: a training error is high or
low only against something outside it (slide 36). The fourth bullet is slides 52 and 53, the debt
of lesson 4 paid: the honest number is the worse one. The fifth bullet is slides
42 to 51.

The fifth bullet, said in full: cross-validation is honest only about what runs
inside it. Its three limits, each with its cure. The sample's chance
correlations: no re-split of the same rows removes them; a test set set aside
before any search, fewer candidates or more rows do. The luck of choosing: report
the nested figure, not best_score_. Rows that are not independent: keep each
drive's readings in one fold with GroupKFold, or cross-validation itself leaks
between folds.

If they remember one habit from today, make it this: before believing any
reported score, ask what the base rate is, how many positives are in the test
set, and which choices were made while looking at which data.

That third question is the one that separates a result from a story, and it is the
one they are most likely to be asked at the exam about whichever exercise is drawn.
:::

# Homework

- **Exercise 5**, discussed at the start of **Friday 13 November**
- A result that is too good. Find out why, then produce the honest number

::: notes
Set it explicitly and say out loud when it comes back: the first ten
minutes of next Friday, 13 November.

The shape of it: they are given a notebook that reports an excellent score, and
the score is wrong. They have to find the reason, fix it, report the honest
number, and say how much honesty cost.

Warn them that there is more than one problem planted, and that at the exam
what counts is the diagnosis, not the final figure - which will be much worse
than the one they start with, and is supposed to be.

Next week: k-nearest neighbours, Naive Bayes and support vector machines. Three
new model families, all of them evaluated with what we built today.
:::

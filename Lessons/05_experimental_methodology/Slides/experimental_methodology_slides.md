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
from the material: the slides marked (optional) - 19 and 20, 33 and 34, 48 and
49 - and notebook 2 are left for self-study, and the handout covers them all.
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

Section 8 of the handout pays it off, and slide 46 shows the number.
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
measured score that is systematically too high or too low. On slides 25 to 34 it
is the bias of a MODEL: a model too simple to follow the truth, wrong in the same
way on every training set. A predictable slip is to merge them - "cross-validation
removes bias" - when it removes neither: it makes the estimate more precise, which
is a third thing.
:::

# A test score is a measurement

- You would not report a poll of forty people to three decimals
- Holding data out makes the estimate **unbiased**: right on average, over all the splits you might have drawn
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

# Choosing a hyperparameter: two recipes

![](two_recipes.png)

::: notes
Two recipes, and the difference is where the hyperparameter acts.

Left, a hyperparameter fixed before the fit. Lambda sits inside the cost being
minimised, so one fit cannot serve every value: each candidate needs its own fit.
With cross-validation (slide 13) every candidate costs k fits: three values of C
times five folds is 15 fits. The best mean wins, the winner is refitted once on
all the data that is not the test set - fit number 16 - and the test set is
scored once. scikit-learn's GridSearchCV does exactly this loop (slide 43): with
refit=True, its default, best_estimator_ is that sixteenth model.

With a fixed validation set instead of cross-validation the loop is the same with
one fit per candidate: three fits on the training part, three scores on the
validation set, then a refit on training plus validation, then the test once.

Right, a hyperparameter chosen after the fit. The threshold acts on probabilities
the model has already produced, so one fit serves every candidate: fit once,
predict on the validation set, try every threshold on those probabilities, keep
the cheapest. Slide 46 does it for lesson 4's costs.

Both end the same way: the test set is touched once, at the end, after every
choice has been made. Handout section 3.
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
Left: what each winner wins by - on each of the 200 splits, the best AUC minus
the second best, 0 on a tie. The median margin is 0.0015; the dashed line is the
0.024 that a single measurement wobbles by (slide 7), sixteen times wider. The
decision is made inside the noise. Right: the 200 test AUCs of each model, as a box plot - say how to
read one, it is the first in the course: the box holds the middle half of the 200
scores, the line inside it is the median, the whiskers reach the rest, and the
circles are the few scores unusually far out. The three boxes sit at the same
height.

The right panel is the truth about these three models. The left panel is how
little separates the winner a single experiment reports.
:::

# k-fold cross-validation

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
800 drives, no test set needed. To CHOOSE (slide 10's left recipe): set the test
set aside first, run it on the rest, refit the winner, score the test set once.

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

- The performance of a **procedure** (the whole recipe), not of one fitted model
- Slightly **pessimistic**: with 5 folds, each model trains on 4/5 = 80% of the data
- Seed 0: the mean of the five round AUCs is **0.9514**
- In code: `cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True),
  scoring="roc_auc")`, one score per round

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

When nothing is being chosen, this is the measurement: the model you ship is the
same recipe refitted on all 800 drives, and 0.9514 is its estimate. A test set is
needed only when something was chosen (slide 43).

Second: every round's model trains on four fifths of the data, so it estimates the
performance of a model trained on less data than the one you will eventually
ship. The bias is in the safe direction, and it shrinks as k grows. That is the
argument for leave-one-out, where k equals the number of rows.

Handout section 4.2 has both in symbols.
:::

# Five rounds, five different models

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
:::

# Report the spread, but not as a confidence interval

- 95% **confidence interval**: estimate ± 1.96 × standard error; it would contain the truth in 95 of 100 repeated experiments
- s ÷ √k (s: standard deviation of the k round scores) is **too small**: training sets share 3 of 4 folds, and s rests on only 5 scores
- Report the mean and s, call s a spread, treat small differences as unresolved

::: notes
Recall the interval first, from lesson 1 and slide 5's poll: the standard error
is how far the estimate would move if the whole experiment were repeated on a new
dataset, and estimate plus or minus 1.96 standard errors is a 95% interval.

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

Practical instruction: quote the mean and the spread, call the spread a spread,
and treat small differences between models as unresolved.
:::

# The real spread is twice the naive one

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
example is not a measurement of anything - and nothing in the code warns you.

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
still predicted once, and all 29 failures count.

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

What it does NOT fix is the subject of slides 38 to 45.
:::

# How many folds? (optional)

- **k = 5 or 10** in practice
- Larger k: less pessimistic, more fits to run, round scores can be more correlated
- k = m (m the number of rows) is **leave-one-out**: every row is a validation fold on its own

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

The question always comes, so answer it before it does.

Larger k means each round's model trains on more data - 80% at k = 5, 90% at
k = 10 - so the estimate is less pessimistic: the bias shrinks. A predictable slip
is "larger k is more optimistic": cross-validation errs low, and a larger k
errs less. It also means more fits, and the training sets overlap even more
heavily, so the round scores can be more strongly correlated (on this fleet the
correlation is small, slide 16).

Five and ten are the conventional choices and the difference between them rarely
matters. Leave-one-out is the extreme: nearly unbiased, m fits, and notoriously
high variance for the estimate itself. With AUC it cannot work at all: a fold of
one row holds no failing-healthy pair, so its AUC is undefined - cross_val_score
with LeaveOneOut returns nan for every round.

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
:::

# When the rows are not independent

- Suppose the table held ten readings per drive (lesson 4's has one row per drive)
- A random split puts nine of a drive's readings in training and one in test
- The model recognises the drive, not the failure

::: notes
Everything so far assumed rows are independent draws. That assumption is
violated constantly and silently, and random k-fold gives no warning.

The concrete case, which they will meet - a hypothetical one here, since the
fleet table has one row per drive: suppose it had ten telemetry readings per
drive. A random split puts nine of a drive's readings in training and one in
test. The model does not need to learn anything about failure - it learns to
recognise that drive, and the test row is nearly a duplicate of a training row.

The score is excellent and the model is worthless on a drive it has never seen,
which is the only situation anyone cares about. Why the test row is a near
duplicate: a drive's ten readings have nearly the same values and the same label,
so the model can look the answer up instead of learning the pattern. It is a form
of leakage: the test row is not independent of the training rows.

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

- If one drive (or patient, or customer) has several rows: **GroupKFold** keeps all its rows in the same fold
- If the model must predict the future: **TimeSeriesSplit** trains on the past and validates on what comes after it

::: notes
The rule that covers both: split along the axis you need to generalise across.
If the model must work on a drive it has never seen, no drive may appear in both
training and test. GroupKFold takes a group label and guarantees that.

Time is the other case, and it is stricter. With a random split the model trains
on Wednesday and tests on Tuesday, which is not the problem anyone has. Real
deployment always predicts forward, so validation must too: TimeSeriesSplit
trains on a prefix and tests on what comes next.

How it is called: GroupKFold needs a group label per row, passed apart from the
features - cross_val_score(model, X, y, groups=drive_id, cv=GroupKFold(n_splits=5)).
The label comes from the data, a serial number or a patient id, and it is NOT a
feature: a model that sees it can recognise the drive again. Groups may differ in
size - one drive with ten readings among drives with one each is fine; the
predictable slip is "GroupKFold needs equal-size groups". What unequal groups
change is weight: that drive counts ten times in training and in the score, so
aggregate to one row per drive or weight its readings 1/10.

Two cautions. GroupKFold is not stratified: with 29 failing drives a fold can get
few of them; StratifiedGroupKFold does both. And the test set must be grouped too:
train_test_split takes no groups, GroupShuffleSplit does.

The tell that you needed one of these and did not use it is the same in both
cases - a validation score that is excellent and a production score that is not.
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

Run notebook 01. Fifteen minutes.

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

# The experiment: 300 training sets

- **Goal**: split the squared error on a new day into three parts, and measure each
- Daily energy at t °C, written by us: f(t) = 240 + 0.55 (t − 18)² + 12 sin(t/4) kWh
- 300 training sets of 25 random days, −5 to 38 °C, noise of standard deviation 22 kWh
- Each fitted with `make_pipeline(PolynomialFeatures(degree), LinearRegression())`, degree 1 to 12

::: notes
Say the goal first, because the next eight slides serve it. A model trained on
some days is asked about a new day; its error is measured squared, as lesson 3's
mean squared error (MSE). We will split that squared error into three parts -
two that belong to the model, one to the data - and measure each with numbers.
The formula two slides on is the split; slides 29 and 30 do the measuring.

Then the data. One observation One observation
is one day: an outdoor temperature and that day's energy use.

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
on different days. In real life you would have one of them. Every error that
follows is computed from these 300 models, at 220 test temperatures spaced
evenly from −4 to 37 °C.
:::

# Three hundred parallel universes

![](many_universes.png)

::: notes
Before any definition, look at the models. Each faint line is a model fitted to a
DIFFERENT training set of 25 days from the same curve. Forty of the 300 are drawn, so the panel stays readable; the red
average is taken over all 300. In real life you drew exactly one of them and
never knew about the rest.

Left, degree 1: every line misses the curvature in the same way, and the red
average is nowhere near the black truth. That gap, between red and black, is
bias. The lines also fan out, more than degree 3's: that fan is variance, not
bias. A predictable mistake is to read bias off the fan - "the lines diverge,
so the bias is high" - which is natural, since both are error. But bias is
about where the average sits, and variance about how far the lines sit from it.
Slide 31 gives the numbers, bias² 5,250.8 and variance 723.8, and its notes
explain why a straight line pivots.

Middle, degree 3: average tracks the truth, spread is modest. Both small.

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

# Expected error = bias² + variance + noise

$$\mathbb{E}\left[(y - \hat{f})^2\right] = \text{bias}^2 + \text{variance} + \sigma^2$$

::: notes
This is what the experiment measures. The left side is the goal of slide 25:
the squared error of a model on a new day, averaged. The right side is the
split, three terms the next slide defines and slides 29 and 30 measure.

State it, do not derive it - handout section 5.2 does that in five lines, and it
is worth reading because the derivation is short and the cross terms vanishing
is the whole trick.

Read the symbols on the previous slide's figure. Pick one test temperature x,
say 30 °C, and imagine a vertical line through it in each panel.
- f-hat is where ONE faint line crosses that vertical: the prediction of one
  model, fitted to one random training set of 25 days.
- y is a new day at 30 °C: the truth plus fresh noise.
- E averages over everything that is random: which of the 300 training sets the
  model came from, and the noise in the new day.
So the left side is the average squared miss of a model trained on random days,
predicting a new day at x. On the right, bias² is the squared gap between the red
and black curves at x, variance the spread of the faint lines around the red one
at x, and sigma squared the noise variance, 484. The next slide says what each
term means and why bias appears squared; slide 29 does it with numbers.

The important claim is that this is an IDENTITY, not an approximation or a
useful metaphor. Notebook 2 checks it on 300 independently drawn training sets
and the two sides agree to 2.7 times ten to the minus twelve - floating-point
equality.

That matters pedagogically: they have seen this picture in every textbook drawn
as a schematic. Here it is measured.
:::

# What each term means, and why it is squared

- **Bias** = average prediction − truth, in kWh: it can be negative
- **Squared** because the error is measured squared: bias² is in kWh², like the other two terms, so the three add
- **Variance** = average of (prediction − average prediction)²: how much the 300 models disagree
- **Noise** σ² = 22² = 484: the days scatter around the curve; we know it because we added it

::: notes
The three terms of the formula, in words, now that the picture has shown them.

Bias first. It is a gap with a sign: the average of the 300 predictions minus
the truth, in kWh. At 30 °C, for the straight line, 303.6 − 330.5 = −26.9 kWh:
the average line sits below the curve there. Why does it appear squared? Because
the error we set out to measure is squared, (y − f-hat)². Expand that square -
handout section 5.2 does it in five lines - and the gap between the average model
and the truth comes out squared, in kWh², the units of the other two terms. That
is what lets the three be added. A bias of +26.9 and one of −26.9 cost the same.

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

# From 300 models to three numbers, at 30 °C

| Step, degree 1 | Formula | Value |
|---|---|---|
| The truth | f(30), known because we wrote f | 330.5 |
| The 300 predictions | one per model | 300.8, 317.1, 304.8, and so on, from 215.6 to 364.7 |
| Their average | sum ÷ 300 | 303.6 |
| **bias²** | (303.6 − 330.5)² | **723.9** |
| **Variance** | sum of (prediction − 303.6)² ÷ 300 | **856.1** |
| **Noise** | 22² | **484.0** |
| Expected error | bias² + variance + noise, unrounded | **2,063.9** |

::: notes
The bridge from the picture to the table on slide 31: the formula holds at one
test temperature at a time, so do it at one. The 300 degree-1 models of slide 26,
each asked for its prediction at 30 °C. The numbers are printed by notebook 2,
section 2, "First at one temperature".

Read the last row as one sentence: a straight line trained on 25 random days,
asked about a new day at 30 °C, misses by a squared error of about 2,064 kWh² on
average - 724 because the line cannot bend, 856 because it depends on which 25
days it saw, 484 because the new day is itself noisy.

Rounding, before someone checks: every number is computed from unrounded
values. Squaring the printed gap gives (−26.9)² = 723.6, not 723.9, because the
unrounded average is 303.55 and the truth 330.46. And the printed parts add to
723.9 + 856.1 + 484.0 = 2,064.0, printed 2,063.9 - the same last-digit effect
as slide 31's 614.4 against 614.3.

The variance is the spread of the 300 predictions: a standard deviation of 29.3
kWh, so squared, 856.1.

Two checks, if someone asks. Degree 3 at the same temperature: average 321.0,
bias² 89.8, variance 98.1, noise 484.0, total 671.8 (the printed parts add to
671.9) - three times smaller. And a
check that does not use the formula: give each of the 300 models 2,000 new days
at 30 °C, truth plus fresh noise, and average the squared errors. The notebook
gets 2,061.0 for degree 1 and 670.0 for degree 3, within sampling error of the
formula's 2,063.9 and 671.8.

Noise is the one term nobody estimates here: we added it, so we know it. On real
data it cannot be separated out.
:::

# From one temperature to the table

![](error_by_temperature.png)

::: notes
The same three numbers as the previous slide, now at every test temperature from
−4 to 37 °C, for degree 1. Notebook 2 draws it, right after the 30 °C example.

Red is bias²: zero where the straight line happens to cross the curve, near 5
and 28 °C, and large at the two ends and in the middle, where the line
passes above the bottom of the curve. Its average over the 220 temperatures, the
dashed line, is 5,250.8. So 30 °C was a lucky spot for degree 1: its bias²
there, 723.9, is far below the average.

Blue is variance: lowest in the middle, where the days are dense, and rising
toward both ends, where the line pivots. Its average is 723.8.

The dotted line is the noise, 484 everywhere.

The averages make the first row of the next slide: 5,250.8 + 723.8 + 484.0 =
6,458.6. Every row of that table is built the same way, one degree at a time.

And the caveat, once more: we can split the error into three only because we
know the truth and the noise. On real data you see only the total, on a test
set. Slide 35's learning curves are the tool for real data.
:::

# Measured, on 25 observations

| Degree | bias² | Variance | Noise | Total = bias² + variance + noise |
|---|---|---|---|---|
| 1 | 5,250.8 | 723.8 | 484.0 | 6,458.6 |
| 2 | 59.6 | 70.8 | 484.0 | **614.3** |
| 3 | 44.5 | 125.8 | 484.0 | 654.4 |
| 5 | 8.9 | 368.9 | 484.0 | 861.8 |
| 12 | 153,537.3 | 32,174,344.9 | 484.0 | 32,328,366.2 |

::: notes
Each row is averaged over the 300 training sets of 25 days and the 220 test
temperatures, as slide 30 showed for degree 1. The totals add the unrounded parts, so a student adding the printed
ones may be off in the last digit: 59.6 + 70.8 + 484.0 = 614.4, printed 614.3.

Noise is 484 in every row - point at the column and say that it never
moves, because it is a property of the data and not of the model.

Degree 1 is a bias problem: bias² of 5,250, two orders of magnitude above the
next row. Wrong everywhere, in the same way.

Degree 3 is the "about right" panel of slide 26: 654.4, close to degree 2's
614.3, the minimum.

Do not say it has the smallest variance - it does not, 723.8 against degree 2's
70.8, and it is on the slide. Every universe draws its own 25 temperatures, and
a straight line that cannot follow the curve pivots according to where they
fell. Handout section 5.3 measures what happens when the temperatures are held
fixed instead: the variance column then climbs from the first row.

Degree 12 is a variance problem, and the magnitude deserves a pause: 32 million,
four orders of magnitude larger than anything else. On 25 points a degree-12
polynomial nearly interpolates, and an interpolating model is a machine for
amplifying noise.

Someone will notice that degree 12 also has the worst bias², which contradicts
"flexible models have low bias". It does not: with variance this large the
average of the fitted curves is not a meaningful curve, so the bias term
inherits the instability. Read the two together.
:::

# Bias, variance and total error against degree, measured

![](bias_variance_tradeoff.png)

::: notes
The textbook picture of this trade-off, measured instead of drawn. Many of them
will meet the drawn version later, in every machine-learning text. The textbook version: bias falling
monotonically, variance rising, the total a U. Two of those three are not what
the measurement gives.

Bias² falls only as far as degree 5 - 5,250 down to 8.9 - and then climbs to
153,537, for the reason on the previous slide: at that variance the average of
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
errors nothing else catches.

Connects forward to the last hour, where we manufacture exactly that symptom
deliberately.
:::

# The best model depends on how much data you have (optional)

| Degree; total error at training-set size m | m = 25 | m = 60 | m = 150 |
|---|---|---|---|
| 2 | **614.3** | 570.6 | 555.3 |
| 5 | 861.8 | **540.9** | **506.9** |
| 12 | 32,328,366.2 | 7,626.9 | 899.0 |

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

The same experiment at three sample sizes. The best degree moves from 2 to 5 as
the data grows, because there is now enough of it to pin down the extra
coefficients and the variance has fallen accordingly.

Degree 12 improves by a factor of four thousand between 25 and 60 observations
(32,328,366.2 ÷ 7,626.9 = 4,239), and of thirty-six thousand by 150 (÷ 899.0 =
35,960), and is still the worst of the three rows shown. Not of all six: at 150
the straight line, degree 1, is worse, 5,861.5 (handout section 5.4).

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

Say that the right-hand setup is not contrived. It is what a wide feature table
looks like when most columns carry nothing, which is most feature tables.
:::

# Which fix to buy

- A learning curve: both scores against training-set size,
  `learning_curve(model, X, y, cv=folds, scoring="roc_auc")`
- Curves **meet low**: bias. More data will **not** help
- Wide gap, validation **still rising**: more data **will** help
- Wide gap, validation **flat**: regularise, drop features. **Slide 35, right, is this one**

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
:::

# Notebook 2, self-study (optional)

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| 300 parallel universes, 25-26 | `PolynomialFeatures` then `LinearRegression`, in `make_pipeline`, fitted 300 times |
| the three parts measured, 27-33 | NumPy on the 300 predictions → sum and measured error agree to 2.73e-12 |
| more data moves the answer, 34 | the same at 25, 60 and 150 days → best degree 2, then 5 |
| learning curves, 35-36 | `learning_curve` with `StratifiedKFold` → 0.715 against 0.718; 1.000 against 0.847 |


::: notes
The table maps each section to its calls and its number. On 6 November this
notebook is left for self-study: show the table, say what each section measures,
and that its numbers are on slides 26 to 36, which were just shown.

When there is a full three hours, run it live, for twenty minutes.

The cell to protect is the decomposition table where the sum equals the measured
error to twelve decimal places. Let them see the identity hold rather than
telling them it does.

If time is short, skip the three-sample-size comparison and assign it as reading
 -  the table is in the handout.
:::

# Now: what cross-validation does not protect you from

- 2,000 columns of **pure random noise**, 800 rows, 29 positives
- No relationship to the label whatsoever
- The honest score of any model on it is **0.500**
- The plan: keep the 10 columns whose F-test against the label scores highest,
  `SelectKBest(f_classif, k=10)`, then cross-validate

::: notes
Set this up carefully, because the demonstration only lands if they believe the
data is genuinely empty.

Two thousand columns drawn from a standard normal, independent of the label and
of each other. There is nothing to find. Any model's honest AUC is one half.

Tell them what is coming so they can watch for it: we are going to write code
that looks completely reasonable, and get 0.93.

The last bullet names the selector. f_classif scores each column by how far apart
its means are for the two classes, relative to its spread (an analysis-of-variance
F-test), and SelectKBest keeps the k best. Point out the clash of letters: this k
counts columns, not folds.
:::

# The code almost everyone writes

```python
sel = SelectKBest(f_classif, k=10)
sel.fit(noise, y)
X = noise.loc[:, sel.get_support()]
cross_val_score(model, X, y, cv=cv)
```

::: notes
Read it aloud as the sentence a practitioner would say: "two thousand features
is too many, so let me keep the ten most predictive ones, then cross-validate a
model on those."

Every line is defensible in isolation. Feature selection is normal. Ten from two
thousand is reasonable. Cross-validating afterwards is exactly what the first half of
today recommended.

Do not reveal the bug yet. Ask them to find it before the next slide.
:::

# 0.931

- Cross-validated AUC **0.931** on data with no signal
- Four of five folds between 0.92 and 0.98

::: notes
The result. The folds are 0.928, 0.979, 0.975, 0.943 and 0.832: close
agreement, which is exactly what a trustworthy result looks like.

Now the bug, which is on the first line: SelectKBest was shown EVERY row,
including the rows that would later serve as validation folds. It searched two
thousand columns for the ones that best matched labels it had already seen, and
handed the winners to cross-validation.

The framing worth using: cross-validation did not fail. It was lied to. It
faithfully measured a procedure that had already peeked.

This is the single most common serious error in applied machine learning, and it
is committed in published work regularly.
:::

# Fix it, and look what happens

- Selection moved **inside** the pipeline, refitted in every round:
  `Pipeline([("select", SelectKBest(f_classif, k=10)), ("scale", ...), ("model", ...)])`
- 20 different fold seeds: **0.658** on average, 0.576 to 0.759
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

- With 2,000 columns and 800 rows, some columns match the label **by accident**
- That accident belongs to the **sample**, not to the split
- Every fold contains it

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

What helps: a test set held out before any of this and used once; fewer
candidates; or more rows. Nothing else.
:::

# Hyperparameter search is the same problem

| `GridSearchCV`, 5 values of k × 5 of C | AUC |
|---|---|
| best of 25 candidates, `best_score_` | **0.7999** |
| average candidate | 0.7265 |
| nested cross-validation, next slide | 0.6699 |
| the truth, by construction | 0.5 |

::: notes
Choosing between configurations is choosing, and choosing on data costs the same
honesty. GridSearchCV tries every combination of five values of k (columns kept)
and five of C, 25 in all, each scored by 5-fold cross-validation, on the same
signal-free table. The nested row is explained on the next slide; read the first
two rows now.

The number a practitioner reports is best_score_, and it is the MAXIMUM of 25
noisy estimates. The maximum of a set of noisy numbers is biased upward even
when every one measures the same quantity - and here they nearly do, since the
average candidate is 0.7265 and the worst is 0.6840.

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

0.500 is the truth. 0.6699 is nested cross-validation. 0.7999 is what the search
reports. 0.931 is selection before cross-validation.

Say clearly: nobody in this picture wrote dishonest code. The difference between
0.93 and 0.50 is entirely a matter of which choices were made while looking at
which data.

Each step to the left is a piece of discipline, and each one costs a chunk of
the score you would otherwise have reported. That is what discipline feels like
from the inside, and it is why it is unpopular.
:::

# The debt, paid

| Slide 6's 800 drives: fit on 450, choose on 150, report on 200 (7 failures) | Missed × 2,600 + false alarms × 140 | Cost on the test set |
|---|---|---|
| 0.50, the default | 4 × 2,600 + 1 × 140 | 10,540 EUR |
| 0.050, chosen on **validation** | 1 × 2,600 + 15 × 140 | **4,700 EUR** |
| 0.171, chosen on **test** | 1 × 2,600 + 5 × 140 | 3,300 EUR |

::: notes
Lesson 4's threshold, done properly - but on slide 6's 800 drives, not lesson
4's 8,000, so do not compare these euros with lesson 4's 87,620 and 45,540: a
test set of 200 drives with 7 failures has a smaller bill. Split three ways, as
the table header says: fit on 450 drives (56%), choose the threshold on 150
(19%, with 5 failures), report once on the remaining 200 (25%). The middle column
is the bill, counted: the default misses 4 of the 7 failures; both other
thresholds miss 1, and differ only in false alarms, 15 against 5.

Then point at 0.050. The validation set chose almost exactly lesson 4's t* =
140 ÷ 2,740 = 0.051, the threshold the costs alone gave before anyone looked at
data. With 5 failures in the validation set part of that is luck, but it is the
point to land: the formula was already the honest answer; what needed paying was
the 0.078 lesson 4 picked by looking at its test set.

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

# A fixed seed buys repeatability, not stability (optional)

- 30 seeds, all perfectly reproducible: **0.913 to 1.000**
- `AUC 1.000 (random_state=3)` is reproducible **and** misleading

::: notes
Optional on 6 November: skip in class if time is short, and say it is left for
self-study, with the handout.

Every one of those thirty runs is exactly reproducible by anyone with the same
code. They disagree with each other by nearly nine points.

So reproducibility is necessary and nowhere near sufficient. Report the seed AND
the spread.

Contrast the two ways of writing the same experiment up. "AUC 1.000,
random_state=3" - fully reproducible, thoroughly misleading. "AUC 0.951 plus or
minus 0.019 over 5-fold cross-validation, seed 0" - reproducible and honest.

The second one is also the way to write up every exercise from here on - it is
what the exam asks them to defend.
:::

# What to fix, and what to write down (optional)

- Every `random_state`, and `np.random.default_rng(seed)`
- **Library versions**: defaults change between releases
- What you could **not** fix: threads, graphics processing unit (GPU) non-determinism

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
| the catastrophe, 38-40 | `SelectKBest(f_classif, k=10)` on every row, then `cross_val_score` → 0.931 |
| selection inside the round, 41-42 | `Pipeline` with the selector as its first step → 0.658 |
| the search, 43 | `GridSearchCV`, `best_score_` → 0.7999 |
| nested, 44-45 | `cross_val_score(GridSearchCV(...), cv=outer)` → 0.6699 |
| the threshold, honestly, 46-47 | `train_test_split` twice → 4,700 EUR against 10,540 |
| seeds, 48-49 | 30 values of `random_state` → 0.913 to 1.000 |


::: notes
The table pairs each section with its calls and its number. Run notebook 03.
Fifteen minutes.

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
A checklist they can apply to every remaining exercise and to anything they read.

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

Warn them that there is more than one problem planted, and that the marks are
for the diagnosis rather than for the final figure - which will be much worse
than the one they start with, and is supposed to be.

Next week: k-nearest neighbours, Naive Bayes and support vector machines. Three
new model families, all of them evaluated with what we built today.
:::

---
title: "Lesson 4: Classification and Evaluation Metrics"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "23 October 2026"
---

# Before we start

- Exercise 3, which we discuss now
- Ridge, Lasso, and the choice of the penalty strength

::: notes
Nothing to collect. Ask who chose which penalty and why, and name the recurring
mistake rather than praising the good answers - it is worth more to the room.

The one to watch for: choosing the penalty by looking at the test error. It is
the natural thing to do because the test error is the number they were told to
care about. It is exactly the mistake lesson 1 named: once the test set has
helped choose anything, its score is no longer an honest estimate of how the
model does on new data, because the choice was fitted to it. Today's lesson ends up in
the same place from a different direction, on slide 49, when we choose a decision
threshold by looking at the test set.
Lesson 5 gives them the machinery to do both honestly.
:::

# Today: from a number to a decision

- Logistic regression: the model and why the cost had to change
- The confusion matrix, and why accuracy lies
- Precision, recall, F1, and the threshold nobody chose
- Receiver operating characteristic (ROC) curves, and where they mislead
- Choosing a threshold from money

::: notes
The agenda. Point out the proportion, because it is the point of the lesson:
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
- Answer with a probability: $0 \leq P(y = 1 \mid x) \leq 1$
- $y = 1$ if the drive failed, $y = 0$ if not; $x$ holds its six counters

::: notes
Read the third bullet aloud: "the probability that y equals 1, given x". The
vertical bar is read "given": everything to its right is what we know about the
drive, everything to its left is what we want to know. The two inequalities are
the point of writing it this way - whatever the model outputs has to stay
between 0 and 1, and slide 7 shows a straight line failing exactly that.

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
- `train_test_split(..., stratify=y)`: 6,000 to train, 2,000 to test

::: notes
SMART is the diagnostic data a drive keeps about itself: reallocated sectors
(bad patches of the disk the drive has had to replace), spin retries, read error
rate, temperature, hours in service, and seek error rate. Expand the acronym
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
consequence yet - it lands far harder on slide 28, after the break, when they see
96.2% printed as a model's accuracy score.
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
Left: the curve. Its horizontal axis is z, the number the linear model
w-transpose-x plus b produces for a drive - think of it as the total evidence
that the drive will fail, all six counters added together with their weights.
Its vertical axis is the probability. Slide 10 shows that z is exactly the
log-odds; for now, z is "evidence".

Steep in the middle, flat at both ends. Make the case that the flatness is the
right behaviour, not a compromise - and be precise about what is diminishing.
It is the total evidence z, not any one counter. Two more units of z starting
from z = 0 take the probability from 0.50 to 0.88; the same two units starting
from z = 6 take it from 0.9975 to 0.9997, 178 times less. Once the model is
nearly sure, more evidence barely changes its answer.

Expect someone to object that the previous figure says otherwise: there the
curve is flat at the START, where drives have few reallocated sectors, not only
at the ends. The answer is the horizontal axis. That curve is drawn against raw
reallocated sectors, not against z, so it is steepest at 13.2 sectors and the
first four sectors move it by only 0.03. Both pictures are right. The model
does not use the raw count: section 1.1 of the handout explains that four of
the counters enter it as log(1 + count), so the first few sectors are a big
step on the model's own scale. On that scale four sectors add +4.7 to z: a drive
with average telemetry and no reallocated sectors has a 0.012% chance of
failing, and four sectors take it to 1.3%, about a hundred times more.

Right panel: the same relationship inverted - probability on the horizontal
axis, log-odds on the vertical one. This is the panel that explains WHY this
particular curve and not some other S-shape; slide 9 says it in words. A probability is trapped in [0,1].
The odds - p over 1 minus p - remove the ceiling. The log of the odds removes
the floor too, leaving something that runs over all of the real line.

And that is exactly what a linear model can safely predict. So the sigmoid is
not a squashing trick bolted onto linear regression: it is what you get when you
model the log-odds linearly and then ask for the probability back.
:::

# Why the log-odds

- A line gives any number; a probability, only [0, 1]
- Odds, p ÷ (1 − p): no ceiling, from 0 to ∞
- Log of the odds: no floor either, −∞ to +∞
- p = 0.01, 0.5, 0.99 → log-odds −4.60, 0, +4.60
- Bonus: coefficients multiply the odds (slide 12)

::: notes
This slide puts in words what the right panel of slide 8 shows. Go through the
two steps slowly; each removes one limit. Odds: a probability of 0.2 is odds of
0.25, and as the probability approaches 1 the odds run off to infinity - no
ceiling. But odds are never negative, so there is still a floor at 0. The
logarithm sends 0 to minus infinity, and the floor is gone too.

Nothing is lost on the way: each probability has exactly one log-odds and back.
The fourth bullet makes the stretch concrete - the scale is symmetric around a
half, and the extremes are pushed outwards.

The question a sharp student asks: why this function and not some other one
that also maps (0, 1) onto the whole line? There are others; the probit, built
on the inverse of the normal distribution, is the common one, and it gives
almost the same S-shape. The log-odds wins on three counts, and the shape is
not one of them:

- the coefficients read as odds multipliers, which is slide 12;
- it is the scale on which Bayes' rule adds evidence: the log-odds after
  seeing a drive's counters equals the log-odds before seeing them (from the
  base rate alone) plus one term per counter, if the counters are independent
  given the class - and a sum of one term per feature is exactly what a linear
  model is;
- with the cross-entropy cost of slide 21 the gradient comes out as
  (prediction minus label) times x, and the cost is convex - handout sections
  3.4 and 4.3.

The predictable mistake: students think the sigmoid was picked because it looks
like the right S-shape. A reasonable reading, because the shape is what slide 8
shows first. But many curves have that shape; what singles this one out is the
log-odds behind it. Handout section 2.3.
:::

# Log-odds of p = P(y = 1 | x): a line

$$\log\frac{p}{1 - p} = w^\top x + b \quad\Longrightarrow\quad p = \frac{1}{1 + e^{-(w^\top x + b)}}$$

::: notes
This is the step from the log-odds of slide 9 to the sigmoid of the next slide,
done on the board.
p is shorthand for P(y = 1 | x), the probability from slide 4.

The left-hand equation is the modelling decision. The log-odds runs over the
whole real line - the right panel of slide 8 - so it is the one
version of the probability a line can predict without ever leaving the range.
So put the linear model of lesson 3 there, unchanged.

The right-hand side is algebra, three lines. Write z for w-transpose-x plus b.
Exponentiate both sides: p over 1 minus p equals e to the z. Multiply out and
collect p: p times (1 plus e to the z) equals e to the z, so p is e to the z
over 1 plus e to the z. Divide top and bottom by e to the z: p is 1 over 1 plus
e to the minus z. That function has a name, and the next slide gives it.

Check it with a number from the handout's table in section 2.3: a log-odds of
+2.20 gives back 1 over (1 plus e to the minus 2.2), which is 0.90. A log-odds
of 0 gives back exactly a half.

The handout runs the same algebra in the other direction, from the sigmoid to
the log-odds, in section 2.3.
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
it decays to zero at both ends. That one fact does two jobs later. It is why the
sigmoid behaves well: flat ends mean that once the model is sure, extra
evidence changes little (slide 8). And it is why squared error fails with it:
slide 23 shows that the same factor ends up inside squared error's gradient and
switches learning off exactly when the model is confidently wrong.
:::

# What the coefficients mean

- The model is linear in the log-odds, log(p ÷ (1 − p)), so `exp(coefficient)`
  multiplies the **odds**, p ÷ (1 − p)
- Features standardised, so one unit = one standard deviation
- A coefficient of 1.80 means **six times the odds** of failing
- In code: `make_pipeline(StandardScaler(), LogisticRegression())`

::: notes
This is the interpretability that keeps logistic regression in production in
medicine and credit scoring. Standardising first (StandardScaler, lesson 3)
puts every feature in the same unit, one standard deviation, so the
coefficients can be compared with each other.

1.80 is the true coefficient of reallocated_sectors, the one the data was
generated with (next slide). It means: a drive one standard deviation worse on
that counter has exp(1.80), about six times, the odds of failing. The fitted
coefficient on the next slide is 1.66, a multiplier of 5.28.

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
Notebook 1, last section. This is the payoff of using synthetic data: we do not just look at the
estimates, we check them against the truth.

The columns: True is the coefficient disk_data.py generated the labels with,
Fitted is what LogisticRegression estimated from 6,000 training drives, and
"Odds ×" is exp(Fitted), the multiplier of slide 12.

Go down the column. Everything is close. Nothing is exact, and say why - 6,000
training drives carrying 230 failures is a modest amount of evidence, and 230 is
the number that matters, not 6,000.

Then the last row, which is the one to pause on. seek_error_rate was generated
with a coefficient of exactly zero: a real-looking counter with no connection to
failure. The model estimates 0.02, an odds multiplier of 1.03 (the table rounds the
coefficient; the multiplier is computed from the unrounded value). It did not fall
for it.

Then ask the room the harder question: would you have noticed, if I had not told
you? Leave it hanging. It is lesson 5.
:::

# The intercept is the average drive's chance

- Fitted intercept: **−6.09**
- $\sigma(-6.09) = 0.0023$: a 0.2% chance for a drive average on every counter
- The fleet fails at 3.8%: the failures come from the few drives far above average

::: notes
This is the slide that quietly explains everything in the second half of the
lesson, so do not rush it.

Say first what the intercept is NOT: it is not the base rate. The fleet fails
at 3.8%; the intercept describes a drive whose every counter sits at the
average, and such a drive almost never fails. The failures come from the small
group of drives far above average on one or more counters. Handout section 2.4.

An average drive has a 0.2% chance of failing in thirty days. For the model to
reach an even bet - probability one half, log-odds zero - the counters have to
add more than 6.09 to the intercept's -6.09. Only badly degraded drives manage
that.

So the model will assign small probabilities to almost everything. Introduce
the word now: a model that outputs a probability still has to decide "fail" or
"healthy", and it does so with a threshold - flag the drive if its probability
is at least the threshold. scikit-learn's predict() uses 0.5. With these
probabilities, a threshold of 0.5 will flag almost nothing. That is not the model being timid. It
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

# Step 1: each drive gets a p

| Drive | What happened | $y$ | Setting 1: $p = \sigma(w^\top x + b)$, the P(fail) |
|---|---|---|---|
| A | failed | 1 | 0.8 |
| B | stayed healthy | 0 | 0.1 |
| C | stayed healthy | 0 | 0.2 |

::: notes
The next five slides answer the question of slide 15 - what should fitting
minimise? - one step at a time, on three drives. The numbers are made up for
the example; drives A, B and C are not from the dataset.

Start from what the model does. Fitting means choosing the coefficients w and
the intercept b. Call one particular choice of w and b a "setting". Given a
setting, the model of slide 11 turns each drive's counters x into a
probability of failing: p = sigma(w-transpose-x + b). This slide shows the p
that one setting, setting 1, gives to each of the three drives.

Next to it, what actually happened, and its label y: 1 for a drive that
failed, 0 for one that stayed healthy - the convention of slide 4.

Ask the room: is setting 1 a good setting? Let them argue it from the table.
It gave the drive that failed a high p, and the two that survived a low one,
so it looks good. The next slide turns that impression into a number.
:::

# Step 2: how likely was the outcome?

| Drive | $y$ | $p$ | Probability of what happened: $p$ if $y = 1$, $1 - p$ if $y = 0$ |
|---|---|---|---|
| A | 1 | 0.8 | 0.8 |
| B | 0 | 0.1 | 1 − 0.1 = 0.9 |
| C | 0 | 0.2 | 1 − 0.2 = 0.8 |
| All three: multiply | | | 0.8 × 0.9 × 0.8 = **0.576** |

::: notes
The idea: judge a setting by how probable it made what really happened.

p is the probability of failing. For drive A, which did fail, that is
exactly the probability of what happened: 0.8. For drives B and C, which
stayed healthy, the probability of what happened is the probability of
staying healthy, which is 1 minus p: 1 - 0.1 = 0.9 and 1 - 0.2 = 0.8. That is
the rule in the last column header: p if y = 1, 1 - p if y = 0.

Then the three drives together. If the drives fail or survive independently
of each other, the probability that all three outcomes happen is the product
of the three: 0.8 times 0.9 times 0.8, which is 0.576. Under setting 1, what we
observed had a 57.6% chance of happening.

That product has a name: the likelihood of the setting. The next slide writes
the same two rules as formulas.
:::

# Step 2 as formulas: the likelihood

$$P(y \mid x) = p^{\,y}\,(1 - p)^{1 - y} \qquad\qquad \mathcal{L}(w, b) = \prod_{i=1}^{m} \left(p^{(i)}\right)^{y^{(i)}} \left(1 - p^{(i)}\right)^{1 - y^{(i)}}$$

::: notes
Two formulas, and both are the previous slide written in symbols.

Left: the probability of what happened to one drive. It is the rule "p if
y = 1, 1 - p if y = 0" in a single expression. Check it with y = 1: p to the
power 1 is p, and (1 - p) to the power 0 is 1, so it gives p. With y = 0 it
gives 1 times (1 - p). Do drive B on the board: 0.1 to the power 0 times 0.9 to
the power 1 is 0.9, the number in the table. The exponents are just a way of
writing an if-statement.

Right: the likelihood, script L, of a setting (w, b). The capital pi means
"multiply over all the drives", i from 1 to m, the way a capital sigma means
"add". Each drive contributes its factor from the left formula; the superscript
(i) says which drive. For our three drives, m = 3 and the product is
0.8 times 0.9 times 0.8 = 0.576.

The p inside depends on w and b, through p = sigma(w-transpose-x + b). That is
why L is written as a function of w and b: change the setting, every p changes,
and the likelihood changes with them. Handout section 3.2.
:::

# Step 3: two settings, the likelier wins

| Drive | $y$ | Setting 1: $p$ | Setting 1: probability of what happened | Setting 2: $p$ | Setting 2: probability of what happened |
|---|---|---|---|---|---|
| A | 1 | 0.8 | 0.8 | 0.3 | 0.3 |
| B | 0 | 0.1 | 0.9 | 0.6 | 0.4 |
| C | 0 | 0.2 | 0.8 | 0.5 | 0.5 |
| Likelihood: the product | | | **0.576** | | **0.060** |

::: notes
Now a second setting, a different choice of w and b. It gives the three drives
p = 0.3, 0.6 and 0.5: a low p to the drive that failed, a high p to one that
survived. It looks worse, and the likelihood says how much worse.

Apply the same rule, column by column. A failed, so its factor is p = 0.3. B
and C survived, so their factors are 1 - 0.6 = 0.4 and 1 - 0.5 = 0.5. Multiply:
0.3 times 0.4 times 0.5 = 0.060.

Compare: 0.576 against 0.060. What actually happened is 9.6 times more probable
under setting 1, so setting 1 explains the data better. Keep it.

That comparison is maximum likelihood. Fitting a logistic regression is the
same comparison made over every possible setting rather than two: choose the w
and b with the largest likelihood. Nobody tries them one by one - gradient
descent does the searching, as in lesson 3.
:::

# Step 4: minus the log, a cost

| Drive | $y$ | Setting 1: −log of probability of what happened | Setting 2: −log of probability of what happened |
|---|---|---|---|
| A | 1 | −log 0.8 = 0.22 | −log 0.3 = 1.20 |
| B | 0 | −log 0.9 = 0.11 | −log 0.4 = 0.92 |
| C | 0 | −log 0.8 = 0.22 | −log 0.5 = 0.69 |
| Sum = −log of the likelihood | | **0.55** | **2.81** |

::: notes
The last step turns "largest likelihood" into "smallest cost", so it can be
minimised as in lesson 3. Natural logarithms throughout.

Take minus the logarithm of each drive's factor and add them up. Because the
log of a product is the sum of the logs, the sum is minus the log of the
likelihood: 0.55 is minus the log of 0.576, and 2.81 is minus the log of 0.060.
The logarithm keeps the order and the minus sign reverses it, so the better
setting now has the SMALLER number - 0.55 against 2.81. Setting 1 still wins.

Read the per-drive numbers too. Minus the log of a probability is called the
surprise of an outcome: 0 for something you were certain of, large for
something you said was unlikely. Setting 2 is most surprised by drive A, 1.20,
because it gave A's failure only 0.3. Maximum likelihood and least total
surprise are the same choice.

Two reasons we always work with the logarithm, worth saying aloud. A sum
differentiates one term at a time. And the product itself is too small to
store: gradient descent starts with all weights at zero, so z is 0 for every
drive and every p is 0.5; 0.5 multiplied by itself 6,000 times, once per
training drive, is about 10 to the minus 1806, while the smallest number a
64-bit float holds is about 10 to the minus 308. The computer would store
exactly zero.

Divide the sum by the number of drives and you have the average surprise: that
is the log loss, and the next slide writes it as a formula. Handout sections
3.1 to 3.3.
:::

# Cross-entropy, also called log loss

$$L = -\left[\, y \log p + (1 - y)\log(1 - p) \right]$$

::: notes
This is the loss on ONE example, L, not the cost over the training set; p is
the predicted probability, the same number the next slide writes as y-hat. Say
the other half aloud rather than showing it: the cost J is the average of this over the
m examples, one over m times the sum. Written out in full the formula reached
the slide edge and came out at 26pt, smaller than the bullets on the slide
before it.

Do not derive it here - the handout, section 3, does it in full. Show it and
read it.

The reading that makes it stick: the cost of an outcome is the SURPRISE of
seeing it. Something you called certain costs nothing. Something you gave a
probability of one in a thousand costs log 1000, about 6.9 (natural logarithm,
as everywhere in this lesson). Something you
declared impossible costs infinity.

Only one of the two terms is ever active, because y is 0 or 1. The bracket is a
way of writing an if-statement in algebra, nothing more.

And the headline: nobody designed this function. It fell out of asking for the
parameters that make the data least surprising.
:::

# Lesson 3's gradient, $\hat{y} = \sigma(w^\top x + b)$

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x_j^{(i)}$$

::: notes
What the formula says, in words. To find how the cost J changes when the
coefficient w_j is nudged up: take each training drive, compute its error - the
predicted probability y-hat minus the true label y, which is 0 or 1 - multiply
that error by the drive's value of feature j, and average over the m drives.
Gradient descent then steps w_j in the opposite direction.

What is new, and what is not. The formula is written with exactly the symbols
of lesson 3's slide 23. The difference is hidden in y-hat: in lesson 3 it was
the straight line w-transpose-x plus b, and here it is the sigmoid of that line,
the probability of slide 11 - which is why the title says so. The cost is
different too: squared error there, log loss (slide 21) here. Different model,
different cost, and yet the same gradient.

A concrete drive. A drive that failed, y = 1, to which the model gives
y-hat = 0.2: its error is 0.2 minus 1, which is minus 0.8. A negative term means
the cost falls if w_j rises, so this drive pulls w_j up, and pulls harder the
larger its value of feature j. A healthy drive, y = 0, given the same 0.2 has
error plus 0.2 and pulls the other way, four times more gently. This is lesson
3's weighted vote: each drive pulls in proportion to how wrong it was and how
large the feature was for it.

Why it comes out the same - worth the board if there is appetite, handout
section 3.4 has it in full. Follow one drive through the chain rule, in three
links, with p for y-hat and z for w-transpose-x plus b:
- how the loss changes with p: (p - y) divided by p times (1 - p) - the
  denominator comes from the logarithms in the log loss;
- how p changes with z: p times (1 - p), the sigmoid's derivative from slide 11;
- how z changes with w_j: x_j.
Multiply the three: the p times (1 - p) cancels, and what is left is (p - y)
times x_j.

Why that cancellation matters, and it is the bridge to the next slide: with
squared error in place of log loss, nothing cancels the factor p times (1 - p),
and slide 23 shows what it does to a drive the model is confidently wrong about.

If someone asks whether the match is a coincidence: no, and one sentence is
enough. Both costs come from maximum likelihood - squared error when the noise
on y is Gaussian (lesson 3, handout section 2.2), log loss when y is a 0/1
outcome - and each paired with its own model gives this same form. Statisticians
call the family "generalised linear models"; the name is not examinable. The
practical consequence is: the same gradient descent code trains both models,
and only the line that computes y-hat changes.
:::

# So why not squared error?

![](loss_comparison.png)

::: notes
Both panels take a drive that failed, y = 1, and move the model's probability
p for it along the horizontal axis. Squared error here is (p - 1) squared, the
handout's convention in section 4.

Left panel, the costs. A drive failed and the model said 0.0001. Squared error
charges 0.9998. An honest "don't know" - 0.5 - costs 0.25. So the worst possible
answer is four times worse than a shrug. Log loss charges 9.2 against 0.69, and
the ratio grows without bound.

Right panel is the real argument, so spend the time here. It shows the size
of the gradient with respect to z, the push gradient descent gives the model.
Log loss's gradient is p - 1 (slide 22). Squared error's is 2(p - 1) times
p times (1 - p): the chain rule through the sigmoid brings in its derivative,
p times one-minus-p, from slide 11, and nothing cancels it. That factor goes to
zero as p goes to zero.

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

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| a line on a 0/1 label, 7 | `LinearRegression` → 3,606 negative probabilities |
| odds, and why not squared error, 8-24 | NumPy only: the sigmoid, the two losses, their gradients |
| gradient descent, 22 | `train_test_split`, `StandardScaler`, fifteen lines of NumPy → log loss 0.6931 to 0.0692 |
| the decision boundary, 26 | `make_pipeline(StandardScaler(), LogisticRegression())` on two features |
| the truth, recovered, 13-14 | `LogisticRegression`, `.coef_`, `.intercept_` → seek_error_rate 0.02, intercept −6.09 |

::: notes
Fifteen lines of NumPy, then one line of scikit-learn, and the same answer. Run
notebook 01. Seventeen minutes, leaving three for the decision boundary on the
next slide. The parts worth stopping on:

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
The threshold of 0.5 is a habit. Telling those two apart is the rest of the
lesson.
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

The general statement, worth putting on the board: if a share pi of the cases
are positive (the positive rate), the trivial always-negative classifier scores
1 minus pi. Here 96.2%. In some fraud
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
single-number metric knows that. Accuracy treats them as identical. F1, the
metric of slide 34, treats them as identical. Only a cost, supplied from outside the data, distinguishes
them.

Ask for a domain where the ratio goes the other way - where a false positive is
the expensive one. Medical screening that triggers an invasive biopsy is the
usual answer; content moderation that removes a legitimate post is another.
:::

# Precision and recall: look at the denominator

![](metric_denominators.png)

::: notes
This picture prevents the most common confusion of the lesson.
Both metrics put the true positives - failed drives we flagged - on top of the
fraction. The difference is entirely in what goes
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

- True and false positives and negatives: TP, FP, FN, TN
- Precision = TP ÷ (TP + FP) = 43 ÷ 56 = **0.768**: of the drives flagged
- Recall = TP ÷ (TP + FN) = 43 ÷ 76 = **0.566**: of the drives that failed
- Specificity = TN ÷ (TN + FP) = 1,911 ÷ 1,924 = 0.993
- In code: `precision_score`, `recall_score`

::: notes
Read each line as an English sentence with the counts in it, not as a ratio.
That is the habit worth building: a metric that cannot be said as a sentence
about actual objects has not been understood.

Specificity is recall for the negative class: of the healthy drives, how many
did we leave alone. One minus it is the false positive rate, which comes back in
the receiver operating characteristic (ROC) section. Mention it now so the term
is not new later.

If someone asks which of the three to report: the answer is that it depends on
who is asking, which is the honest answer and also the lesson.
:::

# F1: why the harmonic mean

| Model | Precision | Recall | Arithmetic mean | Harmonic, F1 = 2PR ÷ (P + R) |
|---|---|---|---|---|
| Flag every drive | 0.038 | 1.000 | 0.519 | **0.073** |

::: notes
Take the stupidest possible model - flag every single drive - and score it.
Recall is a perfect 1.0. Precision is 0.038.

The ordinary average gives that model 0.519, which looks respectable. Anyone
scanning a results table would not
blink at 0.519.

The harmonic mean gives it 0.073, which is the truth.

The reason: the harmonic mean is dominated by the smaller of the two numbers. A
model is only as good as its weaker side, and F1 refuses to let a perfect score
on one axis buy a pass on the other.

Mention F-beta in one sentence: a version of F1 with a weight beta, where beta
above 1 counts recall more than precision. That is what you want here, where a
miss costs about nineteen times a false alarm (2,600 against 140 euros). F2 is the
usual choice.
:::

# Read the minority class, not the average

- `weighted avg` F1: **0.975** (dominated by the healthy class)
- `macro avg` F1: **0.820** (both classes get an equal say)
- In code: `classification_report(y_test, y_pred)` prints both rows

::: notes
classification_report is scikit-learn's summary table, and they will print it
themselves in notebook 2 in about five minutes: one row per class, then three
summary rows - accuracy, macro avg, weighted avg. The two numbers on this slide
are the f1-score column of the last two.

Anticipating it by one slide is deliberate. The rule for choosing between those
rows is what they need in their heads while they run the notebook, not
afterwards - left to themselves they quote whichever is biggest.

The weighted average weights each class by how common it is, so on this problem
it is essentially the healthy class talking, and it reads close to accuracy  - 
inheriting exactly the blindness slides 28 to 31 were about.

The macro average gives the two classes equal weight, so the rare class actually
registers. 0.820 against 0.975 for the same model.

The rule: on an imbalanced problem, quote the macro average or the minority
class directly. Reporting the weighted average is not incorrect - it is the
number that makes a mediocre model look finished, and choosing it without saying
so is how a result gets oversold.
:::

# Notebook 2, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| the number that should worry you, 28 | `accuracy_score` → 96.20% for "always healthy" |
| the confusion matrix, 30 | counts by hand, then `confusion_matrix`, `ConfusionMatrixDisplay` → TP 43 |
| precision and recall, 32-35 | `precision_score`, `recall_score`, `f1_score`, `classification_report` |
| the threshold sweep, 37-38 | `predict_proba`, then the metrics at each threshold |
| the precision-recall curve, 39 | `precision_recall_curve`, `average_precision_score` → 0.717 against 0.038 |

::: notes
The confusion matrix by hand, then from scikit-learn, then the threshold sweep.
Run notebook 02. Twenty minutes.

Have them build the four counts by hand before calling confusion_matrix. It is
four comparisons and it takes two minutes, and it is what makes the jargon stop
being jargon.

Spend time on the threshold sweep table. Do not explain it first  - 
let them read the columns and say what they see. Somebody will notice that
accuracy barely moves. That is the moment to make the next slide.
:::

# The threshold is a choice nobody made

![](threshold_sweep.png)

::: notes
The title of that figure is the point: nothing about the model changes along the
x-axis. Same coefficients, same probabilities, same drives. All that moves is
where we draw the line between "leave it" and "replace it".

That 0.5 is predict()'s default and nothing more. It is the right choice only
when the two errors cost the same and the probabilities are trustworthy - slide
48 derives the threshold from the costs, and it is 0.5 exactly when they are
equal. Here a miss costs about nineteen times a false alarm, and in most real
problems the costs differ too.

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
it moves by three and a half percentage points, 0.942 to 0.977, while recall
falls from 0.803 to 0.184. And
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
equal to the positive rate - 0.038 here - at every recall. That flat dashed line,
not 0.5, is what "no skill" looks like on an imbalanced problem.

That is why this curve is so much more informative than accuracy when positives
are rare: the floor is where the difficulty actually is. Our model's area under
it, the average precision, is 0.717 against a floor of 0.038.

The three marked dots are thresholds 0.5, 0.3 and 0.05. The first two are rows
of the table on the previous slide, so they can check the correspondence: at 0.5,
recall 0.566 and precision 0.768.
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

The two axes: the true positive rate (TPR), which is recall under another name -
the share of failures we catch. The false positive rate (FPR) - the share of
healthy drives we condemn.

The intuition worth saying slowly: at threshold 1 both are zero, flag nothing.
At threshold 0 both are one. The entire content of the curve is the ORDER in
which they get there. A good model raises TPR fast while FPR is still low,
because its highest scores really are the failures. A useless model raises them
together and traces the diagonal.

So the curve asks: as we grow more willing to raise alarms, do we catch failures
faster than we annoy technicians?
:::

# What the area means

- **Area under the curve (AUC) = the probability that a random failing drive scores above a random healthy one**
- Ours: 0.949
- In code: `roc_auc_score(y_test, scores)`

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
costs, as slide 48 does - AUC is not the check you want.

Good place to mention that calibration - whether a predicted 0.7 really comes
true 70% of the time - is a topic of its own, without going into it.
:::

# Where ROC misleads

![](roc_vs_pr_imbalance.png)

::: notes
Same model. Not retrained. Notebook 3 generates a new, larger fleet of 60,000
drives from the same simulator, keeps every healthy drive, and throws away
failures at random until they are 0.4% instead of 3.8% - the direction reality
moves in fraud, disease, hardware failure.

Someone will notice that at 3.8% the AUC reads 0.968, not the 0.949 of slide
42. Different drives: 0.949 rests on the 76 failures of our 2,000-drive test
set, 0.968 on a fleet thirty times larger. Both are estimates of the same
ranking ability.

Left panel: the three ROC curves sit on top of each other. AUC 0.968, 0.965, 0.974. It
reports an excellent model throughout, and at the rarest setting a marginally
better one, which is sampling noise on 231 positives.

Right panel: the precision-recall curves are in three different worlds. Average
precision falls from 0.706 to 0.361, and precision at the same threshold falls
from 0.79 to 0.26 - three alarms in four now false.

Three curves on the left, one model, no problem. Three curves on the right, same
model, a disaster. Let that sit for a second before explaining.
:::

# Why the two disagree

- False positive rate (FPR) divides by **all healthy drives**: an enormous, growing denominator
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
- Its area is the **average precision**, `average_precision_score`: 0.717 here,
  against a floor of 0.038

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
can supply: what the errors cost. We wrote those down on slide 31 - €140 for a
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

State the condition, because the derivation used it: p must be an honest
probability - among drives given 0.05, about 5% really fail. That property is
called calibration (slide 43). Divide every probability by ten, as slide 43
did, and the formula would still say 0.051 while flagging far too few drives.

If the errors cost the same, t-star is 0.5 and the default was right all along.
As misses grow more expensive, it falls towards zero.

Empirically we found 0.08 against a theoretical 0.051, and the curve is flat
between them. Two likely reasons for the gap: the test set holds only 76
failures, and the probabilities are good but not perfectly calibrated.
:::

# The caveat you must state

- We chose the threshold **by looking at the test set**
- That is precisely what lesson 1 forbade

::: notes
Say this out loud rather than leaving it in the handout, because it is the
same mistake you may have named at the start of today's lesson, in the discussion
of exercise 3.

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
| Balanced class weights, 0.50 | 0.895 | 0.249 | **0.950** | 49,500 |

::: notes
class_weight="balanced" makes each failed drive count, in the cost, as much as
about 25 healthy ones - the ratio of healthy to failed drives in the training set.
It changes the training rather than the decision.

Point at the AUC column: 0.949 against 0.950. Reweighting taught the model
essentially nothing new about disk failure. What it did was move where the model
puts its 0.5 line - the same lever as the threshold, pulled at a different
moment.

That matters because class weighting is usually presented as a REMEDY for
imbalance. It is not a remedy; it is another way of setting the same decision.
The remedy, where one exists, is more positive examples.

Where it earns its place: when a downstream tool insists on predict() and gives
you no threshold to move. Then this is how you shift the operating point.

The same reading holds for resampling and for SMOTE (synthetic minority
oversampling technique, which builds artificial minority examples): they move the
operating point too. And both must be applied to the training rows only - inside
each cross-validation fold, once lesson 5 introduces folds - for the reasons
lesson 2 gave about imputation.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| the ROC curve by hand, 41 | `roc_curve`, `roc_auc_score` → 0.9493 |
| what AUC means, 42 | 200,000 random (failing, healthy) pairs → 0.9488 |
| the rare-positive trap, 44-45 | `average_precision_score` beside `roc_auc_score` → AP 0.706 to 0.361, AUC unmoved |
| the threshold from costs, 47-49 | `confusion_matrix` at each threshold → 45,540 EUR at 0.08, against 87,620 at 0.50 |
| class weights, 50 | `LogisticRegression` with balanced class weights → AUC 0.950 |
| three classes, 52-53 | `classification_report` → macro F1 0.707, weighted 0.883 |

::: notes
The ROC curve by hand, the 200,000-pair experiment, the cost curve, then the
multiclass matrix. Run notebook 03. Fifteen minutes, and the two multiclass slides
that follow (52 and 53) are shown when the notebook reaches its last section, so
they run alongside it rather than after it.

If time is short, keep the pairs experiment: it turns a definition they would
otherwise memorise into something they watched happen.
:::

# More than two classes

![](multiclass_confusion.png)

::: notes
Real fleets are not healthy-or-dead. A drive can be degraded: readable, but
reporting errors, and worth replacing at the next maintenance window rather than
at three in the morning.

For K classes the model produces one score per class and turns them into K
probabilities that add up to 1 - the softmax, which for two classes reduces
exactly to the sigmoid. The cost
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
threshold has an answer - C-FP over C-FP plus C-FN, for honest probabilities -
that comes from the business, not from the data.
:::

# Homework

- **Exercise 4**, discussed at the start of **Friday 30 October**
- Build a classifier, choose a threshold from a stated cost ratio, defend it

::: notes
Set it explicitly and say out loud when it comes back: the first ten
minutes of next Friday, 30 October.

Emphasise which part matters at the exam, where one of the ten exercises is
drawn and they talk through their own notebook. The classifier is a few lines
and everyone will get it. The paragraph defending the threshold choice is where
the learning is, and it is what they will be asked to defend.

Warn them about the trap deliberately planted: the dataset has a decoy column,
as this one did, and the exercise asks them to report AUC and average precision
together. Anyone reporting AUC alone will be reporting a number that looks
better than the model is.

Next week: experimental methodology. Cross-validation, bias-variance, and the
honest way to make every choice we made by hand today.
:::
